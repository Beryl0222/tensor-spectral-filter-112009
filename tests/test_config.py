"""配置身份、缓存隔离与作废语义。"""

import unittest

import numpy as np

from helpers import load
from freqmix.contract import (
    ConfigRegistry,
    OpticalConfig,
    StaleConfig,
    ORTHOGONAL,
)
from freqmix.operator import RAISE_STALE, VOID_STALE


def clone_config(config: OpticalConfig, **overrides) -> OpticalConfig:
    return OpticalConfig(
        config_id=overrides.get("config_id", config.config_id),
        shape=overrides.get("shape", (config.height, config.width)),
        grid=overrides.get("grid", [list(r) for r in config.grid]),
        missing=overrides.get("missing", [list(m) for m in config.missing]),
        stencils=config.stencils,
        input_order=config.input_order,
        output_order=config.output_order,
        weight_ids=config.weight_ids,
    )


class ConfigIdentityTests(unittest.TestCase):
    def setUp(self):
        self.session, self.registry, self.weights, self.mixer, self.inputs = load()

    def test_same_shape_different_configs_keep_distinct_maps(self):
        plan_a = self.registry.plan("A", (5, 6), ORTHOGONAL)
        plan_b = self.registry.plan("B", (5, 6), ORTHOGONAL)
        self.assertNotEqual(plan_a.version, plan_b.version)
        sig_a = [g.signature for g in plan_a.groups]
        sig_b = [g.signature for g in plan_b.groups]
        self.assertNotEqual(sig_a, sig_b)
        # 同一图像在两种配置下给出不同输出
        item = self.inputs["change"]
        out_b, _ = self.mixer.forward(item["image"], "B", mode=ORTHOGONAL)
        out_a, _ = self.mixer.forward(item["image"], "A", mode=ORTHOGONAL)
        self.assertFalse(np.allclose(out_a, out_b))

    def test_replacing_config_voids_old_tape_in_raise_mode(self):
        item = self.inputs["e"]
        _, tape = self.mixer.forward(item["image"], "A", mode=ORTHOGONAL)
        old_version = tape.version

        old = self.registry.get_config("A", (5, 6))
        # 同身份同形状、改动一个频点的标定 -> 新版本
        new_grid = [list(r) for r in old.grid]
        new_grid[0] = [0, 0, "q", 1]
        new_cfg = clone_config(old, grid=new_grid)
        self.registry.register(new_cfg)
        self.assertNotEqual(old_version, new_cfg.version)

        with self.assertRaises(StaleConfig):
            self.mixer.backward(item["grad_output"], tape,
                                stale_policy=RAISE_STALE)

    def test_void_policy_returns_zero_gradient_with_report(self):
        item = self.inputs["e"]
        _, tape = self.mixer.forward(item["image"], "A", mode=ORTHOGONAL)
        old = self.registry.get_config("A", (5, 6))
        new_grid = [list(r) for r in old.grid]
        new_grid[1] = [1, 0, "q", 1]
        self.registry.register(clone_config(old, grid=new_grid))

        dx, gr, gi, report = self.mixer.backward(
            item["grad_output"], tape, stale_policy=VOID_STALE
        )
        self.assertTrue(report["void"]["voided"])
        self.assertTrue(np.allclose(dx, 0.0))
        for wid in gr:
            self.assertTrue(np.allclose(gr[wid], 0.0))
            self.assertTrue(np.allclose(gi[wid], 0.0))

    def test_revoked_config_missing_original_info_voids_gradient(self):
        item = self.inputs["e"]
        _, tape = self.mixer.forward(item["image"], "A", mode=ORTHOGONAL)
        self.registry.revoke("A", (5, 6))
        with self.assertRaises(StaleConfig):
            self.mixer.backward(item["grad_output"], tape,
                                stale_policy=RAISE_STALE)
        dx, _, _, report = self.mixer.backward(
            item["grad_output"], tape, stale_policy=VOID_STALE
        )
        self.assertTrue(report["void"]["voided"])
        self.assertTrue(np.allclose(dx, 0.0))

    def test_reload_same_content_keeps_versions_stable(self):
        before = self.registry.plan("A", (5, 6), ORTHOGONAL).version
        registry2 = ConfigRegistry.from_session(self.session)
        cfg = registry2.get_config("A", (5, 6))
        self.registry.register(cfg)
        after = self.registry.plan("A", (5, 6), ORTHOGONAL).version
        self.assertEqual(before, after)

    def test_physical_channel_reordering_is_honored(self):
        # 相机交来的物理通道顺序与逻辑顺序相反时，矩阵寻址按逻辑名称重排
        from freqmix.weights import WeightStore
        swapped = WeightStore(
            self.session["weights"],
            tuple(self.session["input_order"]),
            tuple(self.session["output_order"]),
            physical_input_order=tuple(reversed(self.session["input_order"])),
            physical_output_order=tuple(reversed(self.session["output_order"])),
        )
        logical = swapped.complex_matrix("p")
        # 物理存储里的 (0,0) 应等于逻辑矩阵的 (1,1)
        physical = swapped.real_part("p") + 1j * swapped.imag_part("p")
        self.assertTrue(np.allclose(physical[0, 0], logical[1, 1]))


if __name__ == "__main__":
    unittest.main()
