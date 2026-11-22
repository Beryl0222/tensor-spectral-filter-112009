"""实值扰动导数检查（契约门槛 atol=1e-7, rtol=1e-5，双精度）。

覆盖：

* 输入图像实值扰动 -> 输入图梯度；
* 每个独立权重的实部/虚部扰动 -> 权重梯度；
* stencil 插值把梯度按 0.75/0.25 分摊，不把插值矩阵当独立参数；
* 自共轭位置虚部自由度被投影移除，不产生双倍计数；
* 同一权重被偶数/奇数两种裁边引用时，梯度归并到同一物理权重。
"""

import unittest

import numpy as np

from helpers import load
from freqmix.contract import ORTHOGONAL
from freqmix.operator import RAISE_STALE

ATOL, RTOL, H = 1e-7, 1e-5, 1e-6


def scalar_loss(mixer, image, grad_output, config_id, mode):
    out, tape = mixer.forward(image, config_id, mode=mode)
    return float(np.sum(grad_output * out)), tape


class GradientTests(unittest.TestCase):
    def setUp(self):
        self.session, self.registry, self.weights, self.mixer, self.inputs = load()

    def _analytic(self, item):
        mixer = self.mixer
        loss, tape = scalar_loss(
            mixer, item["image"], item["grad_output"],
            item["config"], item["mode"]
        )
        dx, gr, gi, report = mixer.backward(
            item["grad_output"], tape, stale_policy=RAISE_STALE
        )
        self.assertIsNone(report["void"])
        return loss, dx, gr, gi

    def test_input_gradient_finite_difference(self):
        item = self.inputs["e"]
        loss, dx, _, _ = self._analytic(item)
        x = item["image"]
        rng = np.random.default_rng(1)
        probes = rng.choice(x.size, size=12, replace=False)
        for flat in probes:
            idx = np.unravel_index(flat, x.shape)
            xp, xm = x.copy(), x.copy()
            xp[idx] += H
            xm[idx] -= H
            lp, _ = scalar_loss(self.mixer, xp, item["grad_output"],
                                item["config"], item["mode"])
            lm, _ = scalar_loss(self.mixer, xm, item["grad_output"],
                                item["config"], item["mode"])
            fd = (lp - lm) / (2 * H)
            self.assertTrue(
                np.isclose(fd, dx[idx], atol=ATOL, rtol=RTOL),
                f"输入梯度不符 @ {idx}: fd={fd}, analytic={dx[idx]}"
            )

    def test_weight_gradient_finite_difference_all_ids(self):
        for rec_id in ("e", "o", "change"):
            item = self.inputs[rec_id]
            snapshot = self.weights.snapshot()
            loss, dx, gr, gi = self._analytic(item)
            rng = np.random.default_rng(hash(rec_id) & 0xFFFF)
            for wid in self.weights.weight_ids:
                re = self.weights.real_part(wid)
                im = self.weights.imag_part(wid)
                picks = rng.choice(re.size, size=min(4, re.size), replace=False)
                for f in picks:
                    o, i = np.unravel_index(f, re.shape)
                    # 实部扰动
                    self.weights.restore(snapshot)
                    re[o, i] += H
                    lp, _ = scalar_loss(self.mixer, item["image"],
                                        item["grad_output"], item["config"],
                                        item["mode"])
                    self.weights.restore(snapshot)
                    re[o, i] -= H
                    lm, _ = scalar_loss(self.mixer, item["image"],
                                        item["grad_output"], item["config"],
                                        item["mode"])
                    fd = (lp - lm) / (2 * H)
                    self.assertTrue(
                        np.isclose(fd, gr[wid][o, i], atol=ATOL, rtol=RTOL),
                        f"{rec_id}/{wid} 实部梯度不符"
                    )
                    # 虚部扰动
                    self.weights.restore(snapshot)
                    im[o, i] += H
                    lp, _ = scalar_loss(self.mixer, item["image"],
                                        item["grad_output"], item["config"],
                                        item["mode"])
                    self.weights.restore(snapshot)
                    im[o, i] -= H
                    lm, _ = scalar_loss(self.mixer, item["image"],
                                        item["grad_output"], item["config"],
                                        item["mode"])
                    fd = (lp - lm) / (2 * H)
                    self.assertTrue(
                        np.isclose(fd, gi[wid][o, i], atol=ATOL, rtol=RTOL),
                        f"{rec_id}/{wid} 虚部梯度不符"
                    )
            self.weights.restore(snapshot)

    def test_stencil_split_ratios(self):
        # 奇数裁边 (0,3) 用 pq = 0.75 p + 0.25 q
        item = self.inputs["o"]
        _, _, gr, gi = self._analytic(item)
        # 单独构造只激活 (0,3) 的空间梯度，验证 3:1 分摊
        grad = np.zeros_like(item["grad_output"])
        h, w = item["shape"]
        # 常值梯度只激活 ky=0,kx=0；改选只在目标频点有能量的扰动：
        # 用频域设计一个实空间梯度，其半谱仅 (0,3) 非零
        half = np.zeros((grad.shape[0], h, w // 2 + 1), dtype=np.complex128)
        half[:, 0, 3] = 1.0 + 0.5j
        grad[:] = np.fft.irfft2(half, s=(h, w), axes=(-2, -1))
        _, tape = self.mixer.forward(item["image"], item["config"],
                                     mode=item["mode"])
        _, gr2, gi2, report = self.mixer.backward(grad, tape)
        # p 的该频点贡献与 q 的比值应为 0.75:0.25 = 3（同为 2x2 矩阵结构）
        ratio = np.linalg.norm(gr2["p"]) / max(np.linalg.norm(gr2["q"]), 1e-30)
        self.assertAlmostEqual(ratio, 3.0, places=5)

    def test_no_double_count_at_self_conjugate_bins(self):
        # 自共轭 (0,0) 用 d 的实部：虚部扰动必须零导数，实部只计一次
        item = self.inputs["e"]
        snapshot = self.weights.snapshot()
        _, tape = self.mixer.forward(item["image"], "A", mode=ORTHOGONAL)
        grad = np.zeros_like(item["grad_output"])
        half = np.zeros((grad.shape[0], 5, 4), dtype=np.complex128)
        half[:, 0, 0] = 2.0
        grad[:] = np.fft.irfft2(half, s=(5, 6), axes=(-2, -1))
        _, gr, gi, _ = self.mixer.backward(grad, tape)
        self.assertTrue(np.allclose(gi["d"], 0.0, atol=1e-12))
        # 与单点 FD 比较实部（系数 1/N 而非 2/N）
        for o in range(2):
            for i in range(2):
                self.weights.restore(snapshot)
                self.weights.real_part("d")[o, i] += H
                lp, _ = self.mixer.forward(item["image"], "A", mode=ORTHOGONAL)
                lp = float(np.sum(grad * lp))
                self.weights.restore(snapshot)
                self.weights.real_part("d")[o, i] -= H
                lm, _ = self.mixer.forward(item["image"], "A", mode=ORTHOGONAL)
                lm = float(np.sum(grad * lm))
                self.weights.restore(snapshot)
                fd = (lp - lm) / (2 * H)
                self.assertTrue(np.isclose(fd, gr["d"][o, i],
                                           atol=ATOL, rtol=RTOL))

    def test_shared_weight_across_crops_accumulates_once_per_use(self):
        # 偶数与奇数裁边都引用 p；两图各反传一次，p 的梯度是各自贡献之和，
        # 且每个频点只计一次（自共轭/内部缩放不同但都不翻倍）。
        even = self.inputs["e"]
        odd = self.inputs["o"]
        self.weights.zero_grad()
        total = {wid: np.zeros_like(self.weights.real_part(wid))
                 for wid in self.weights.weight_ids}
        total_i = {wid: np.zeros_like(self.weights.imag_part(wid))
                   for wid in self.weights.weight_ids}
        for item in (even, odd):
            _, tape = self.mixer.forward(item["image"], item["config"],
                                         mode=item["mode"])
            _, gr, gi, _ = self.mixer.backward(item["grad_output"], tape)
            for wid in total:
                total[wid] += gr[wid]
                total_i[wid] += gi[wid]
        # 与逐图 FD 求和比较 p 的若干元素
        snapshot = self.weights.snapshot()
        rng = np.random.default_rng(7)
        for wid in ("p",):
            for f in rng.choice(4, size=3, replace=False):
                o, i = np.unravel_index(f, (2, 2))
                fd_sum = 0.0
                for item, part in ((even, "real"), (odd, "real")):
                    self.weights.restore(snapshot)
                    self.weights.real_part(wid)[o, i] += H
                    lp, _ = self.mixer.forward(item["image"], item["config"],
                                               mode=item["mode"])
                    lp = float(np.sum(item["grad_output"] * lp))
                    self.weights.restore(snapshot)
                    self.weights.real_part(wid)[o, i] -= H
                    lm, _ = self.mixer.forward(item["image"], item["config"],
                                               mode=item["mode"])
                    lm = float(np.sum(item["grad_output"] * lm))
                    fd_sum += (lp - lm) / (2 * H)
                self.weights.restore(snapshot)
                self.assertTrue(np.isclose(fd_sum, total[wid][o, i],
                                           atol=ATOL, rtol=RTOL))

    def test_missing_band_has_zero_weight_gradient(self):
        item = self.inputs["e"]
        _, tape = self.mixer.forward(item["image"], "A", mode=ORTHOGONAL)
        # 构造只在 missing (0,2) 有能量的空间梯度
        grad = np.zeros_like(item["grad_output"])
        half = np.zeros((grad.shape[0], 5, 4), dtype=np.complex128)
        half[:, 0, 2] = 3.0 + 1j
        grad[:] = np.fft.irfft2(half, s=(5, 6), axes=(-2, -1))
        before = self.weights.snapshot()
        _, gr, gi, _ = self.mixer.backward(grad, tape)
        for wid in gr:
            self.assertTrue(np.allclose(gr[wid], 0.0))
            self.assertTrue(np.allclose(gi[wid], 0.0))


if __name__ == "__main__":
    unittest.main()
