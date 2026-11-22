"""混合尺寸成批处理与大规格负载的工作区预算交付。"""

import unittest

import numpy as np

from helpers import load
from freqmix.batch import BatchItem, run_batch
from freqmix.budget import WorkspaceExceeded
from freqmix.contract import ConfigRegistry, ORTHOGONAL
from freqmix.operator import FrequencyMixer
from freqmix.synthetic import build_large_load, large_profile_weights
from freqmix.weights import WeightStore


class BatchTests(unittest.TestCase):
    def test_mixed_size_batch_and_accumulation(self):
        _, registry, weights, mixer, inputs = load()
        items = [
            BatchItem(
                image=v["image"],
                config_id=v["config"],
                mode=v["mode"],
                grad_output=v["grad_output"],
                record_id=k,
            )
            for k, v in inputs.items()
        ]
        weights.zero_grad()
        result = run_batch(mixer, items, byte_limit=1 << 20)
        self.assertEqual(len(result.outputs), 3)
        self.assertTrue(all(o.shape == it.image.shape
                            for o, it in zip(result.outputs, items)))
        # 存储中的梯度等于批量合计
        for wid in weights.weight_ids:
            self.assertTrue(np.allclose(
                weights.grad_real[wid], result.weight_grad_real[wid]))
        self.assertTrue(result.ok)
        self.assertTrue(result.budget_summary["within_budget"])

    def test_budget_rejects_too_small_workspace(self):
        _, registry, weights, mixer, inputs = load()
        item = BatchItem(
            image=inputs["e"]["image"], config_id="A",
            mode=ORTHOGONAL, grad_output=inputs["e"]["grad_output"])
        with self.assertRaises(WorkspaceExceeded):
            run_batch(mixer, [item], byte_limit=2048)

    def test_large_profile_within_scratch_budget(self):
        from inspect_fixture import read_session
        profile = read_session()["load_profile"]
        load_pack = build_large_load(profile)
        weights_dict = large_profile_weights(profile)
        store = WeightStore(
            weights_dict,
            tuple(load_pack["channel_names"]),
            tuple(load_pack["channel_names"]),
        )
        registry = ConfigRegistry()
        for cfg in load_pack["configs"]:
            registry.register(cfg)
        mixer = FrequencyMixer(store, registry, chunk_bins=128)

        items = [
            BatchItem(
                image=it["image"],
                config_id=it["config_id"],
                mode=it["mode"],
                grad_output=it["grad_output"],
                record_id=str(it["index"]),
            )
            for it in load_pack["items"]
        ]
        result = run_batch(
            mixer, items,
            byte_limit=profile["scratch_bytes"],
        )
        peak = result.budget_summary["peak_live_bytes"]
        self.assertLessEqual(peak, profile["scratch_bytes"])
        self.assertEqual(len(result.outputs), profile["count"])
        self.assertTrue(result.ok)
        # 输出保持原尺寸、实值（含通道维 [C,H,W]）
        c = profile["output_channels"]
        for it, out in zip(load_pack["items"], result.outputs):
            self.assertEqual(out.shape, (c,) + tuple(it["shape"]))
            self.assertTrue(np.isrealobj(out))
        # 工作区没有出现“所有样本通道外积铺满”的规模：
        # 峰值应远小于 12 张完整复谱之和
        full_spectrum_one = (
            profile["input_channels"]
            * profile["shapes"][0][0]
            * (profile["shapes"][0][1] // 2 + 1) * 16
        )
        self.assertLess(peak, full_spectrum_one * profile["count"])

    def test_no_full_batch_outer_product_in_events(self):
        from inspect_fixture import read_session
        profile = read_session()["load_profile"]
        pack = build_large_load(profile)
        store = WeightStore(
            large_profile_weights(profile),
            tuple(pack["channel_names"]), tuple(pack["channel_names"]))
        registry = ConfigRegistry()
        for cfg in pack["configs"]:
            registry.register(cfg)
        mixer = FrequencyMixer(store, registry, chunk_bins=64)
        result = run_batch(
            mixer,
            [BatchItem(image=it["image"], config_id=it["config_id"],
                       mode=ORTHOGONAL, grad_output=it["grad_output"])
             for it in pack["items"]],
            byte_limit=profile["scratch_bytes"],
        )
        # 单个分块临时张量不得超过 chunk 对应的频点数规模
        max_block = max(size for note, size, _live
                        in result.budget_summary["events"]
                        if "block" in note)
        c = profile["input_channels"]
        # 块字节 <= dM 块(c*c*chunk*16) 与 dX/carrier 块(c*chunk*16) 的上界
        self.assertLessEqual(
            max_block, c * c * 64 * 16 + c * 64 * 16
        )


if __name__ == "__main__":
    unittest.main()
