"""前向正确性：与朴素全谱参考实现逐条记录比对。"""

import unittest

import numpy as np

from helpers import load, naive_forward
from freqmix.contract import ORTHOGONAL, STRICT, IllegalSpectrum
from freqmix.budget import ScratchBudget, WorkspaceExceeded


class ForwardTests(unittest.TestCase):
    def test_matches_naive_reference_all_records(self):
        session, registry, weights, mixer, inputs = load()
        matrices = {wid: weights.complex_matrix(wid)
                    for wid in weights.weight_ids}
        for rec_id, item in inputs.items():
            config = registry.get_config(item["config"], item["shape"])
            ref, ref_resid = naive_forward(
                item["image"], matrices, config, mode=item["mode"]
            )
            out, tape = mixer.forward(
                item["image"], item["config"], mode=item["mode"]
            )
            self.assertEqual(out.shape, item["image"].shape)
            self.assertTrue(np.allclose(out, ref, atol=1e-12, rtol=1e-12),
                            f"{rec_id} 与参考实现不一致")
            self.assertTrue(np.isrealobj(out))
            self.assertLess(tape.imaginary_residual, 1e-12)

    def test_output_is_real_valued_and_shape_preserved(self):
        _, _, _, mixer, inputs = load()
        for item in inputs.values():
            out, _ = mixer.forward(item["image"], item["config"],
                                   mode=item["mode"])
            # 原尺寸返回：通道、高、宽全部保留
            self.assertEqual(out.shape, item["image"].shape)
            self.assertEqual(out.dtype, np.float64)

    def test_missing_band_contributes_nothing(self):
        session, registry, weights, mixer, inputs = load()
        item = inputs["e"]  # config A, 5x6, missing (0,2)
        out1, tape = mixer.forward(item["image"], "A", mode=ORTHOGONAL)
        self.assertIn([0, 2], [list(b) for b in tape.missing_bins])
        # 把缺测频带的输入谱能量任意改掉，输出不应变化
        image2 = item["image"].copy()
        spec = np.fft.rfft2(image2, axes=(-2, -1))
        spec[:, 0, 2] += 1000 + 500j
        image2 = np.fft.irfft2(spec, s=image2.shape[-2:], axes=(-2, -1))
        out2, _ = mixer.forward(image2, "A", mode=ORTHOGONAL)
        self.assertTrue(np.allclose(out1, out2, atol=1e-10))

    def test_observed_zero_is_distinct_from_missing(self):
        # 未列出的格是“标定零”，有观测语义；权重非零后应立刻产生贡献，
        # 而 missing 格无论权重如何都不贡献——二者在计划里分类不同。
        session, registry, weights, mixer, inputs = load()
        plan = registry.plan("A", (5, 6), ORTHOGONAL)
        flat = {(b[0], b[1]) for b in plan.missing_bins}
        zero_flat = {(b[0], b[1]) for b in plan.zero_bins}
        self.assertIn((0, 2), flat)
        self.assertNotIn((0, 2), zero_flat)
        # 5x6 半谱共 5x4=20 格，列出 7、缺测 1，其余为标定零
        self.assertEqual(len(plan.bins) + len(plan.missing_bins)
                         + len(plan.zero_bins), 5 * 4)

    def test_strict_rejects_illegal_pair_orthogonal_projects(self):
        session, registry, weights, mixer, inputs = load()
        # A/5x6 的 (1,3)<->(4,3)：q 与 conj(p) 一般不共轭，strict 拒绝
        with self.assertRaises(IllegalSpectrum):
            registry.plan("A", (5, 6), STRICT)
        plan = registry.plan("A", (5, 6), ORTHOGONAL)
        kinds = {p["kind"] for p in plan.projected_bins}
        self.assertIn("pair", kinds)
        self.assertIn("self", kinds)

    def test_strict_catches_self_conjugate_imaginary_before_projection(self):
        # 回归：自共轭点非零虚部不能被 real_only 投影提前抹掉而逃过 strict
        from freqmix.contract import OpticalConfig
        session, registry, weights, mixer, inputs = load()
        cfg = registry.get_config("A", (5, 6))
        bad = OpticalConfig(
            "SELF", (5, 6), [[0, 0, "d", 1]], [], cfg.stencils,
            cfg.input_order, cfg.output_order, cfg.weight_ids,
        )
        registry.register(bad)
        with self.assertRaises(IllegalSpectrum):
            mixer.forward(inputs["e"]["image"], "SELF", mode=STRICT)
        # orthogonal 仍按许可的投影规则交付，并报告该频点
        out, tape = mixer.forward(
            inputs["e"]["image"], "SELF", mode=ORTHOGONAL
        )
        self.assertTrue(np.isrealobj(out))
        self.assertIn(
            [0, 0], [p["bin"] for p in tape.projected if p["kind"] == "self"]
        )

    def test_even_nyquist_and_odd_last_column_are_different_identities(self):
        # 旧 bug：偶数裁边 Nyquist 项与奇数裁边最后一列被当作同一种量。
        # 这里两者频率身份不同：(5,6) 的 kx=3 是特殊列，(5,7) 的 kx=3 不是。
        _, registry, _, _, _ = load()
        plan_even = registry_plan(registry, "A", (5, 6))
        plan_odd = registry_plan(registry, "A", (5, 7))
        special_even = {b.kx for b in plan_even.bins if b.is_special}
        special_odd = {b.kx for b in plan_odd.bins if b.is_special}
        self.assertIn(3, special_even)
        self.assertNotIn(3, special_odd)

    def test_forward_respects_budget(self):
        _, _, _, mixer, inputs = load()
        # 最小频域中间量（半谱）就要 2*5*4*16=640 字节，256 字节必被拦截
        budget = ScratchBudget(256)
        item = inputs["e"]
        with self.assertRaises(WorkspaceExceeded):
            mixer.forward(item["image"], "A", mode=ORTHOGONAL, budget=budget)
        budget_big = ScratchBudget(1 << 20)
        mixer.forward(item["image"], "A", mode=ORTHOGONAL, budget=budget_big)
        self.assertTrue(budget_big.summary()["within_budget"])


def registry_plan(registry, cid, shape):
    return registry.plan(cid, shape, ORTHOGONAL)


if __name__ == "__main__":
    unittest.main()
