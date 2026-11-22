"""单次前向/反向频域通道混合算子。

正向链路（契约口径，双精度，前向 DFT 无缩放、逆变换除以面积）::

    x [C,H,W]  --rfft2-->  X [C,H,K]
    Y[o,ky,kx] = sum_i M[o,i](ky,kx) * X[i,ky,kx]     （半谱逐频点）
    y = irfft2(Y, s=(H,W))

特殊列上的输出由计划保证共轭对称（见 :mod:`freqmix.contract`），因此
``irfft2`` 之前不会出现“直接吞掉不匹配虚部”的情况；如确有残差，
:meth:`assert_legal_real_spectrum` 明确报错而不是静默丢弃。

反向从实值空间梯度出发，先经 ``irfft2`` 的伴随（逐频点缩放
``1/N`` 自共轭、``2/N`` 其余）再做矩阵转置共轭，并把权重梯度沿配方
分摊回**独立物理权重**的实部/虚部变量。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .budget import ScratchBudget
from .contract import (
    ORTHOGONAL,
    STRICT,
    ConfigRegistry,
    IllegalSpectrum,
    Plan,
    StaleConfig,
)
from .weights import WeightStore

VOID_STALE = "void"      # 配置失效：该次梯度明确作废（返回零并报告）
VOID_MISSING = "missing"
RAISE_STALE = "raise"    # 配置失效：直接抛 StaleConfig


@dataclass
class MixerTape:
    """前向记录。只持有配置身份/版本与形状，不缓存计划对象。

    这样当光学配置被撤销或替换后，反向无法借同形缓存取得错误映射；
    缺原版信息时按策略作废。
    """

    config_id: str
    shape: tuple[int, int]
    mode: str
    version: str
    plan_key: tuple
    n_in: int
    n_out: int
    # 只读输入影像本身不计预算；保存引用以在反向重算 rfft
    input_ref: np.ndarray
    projected: list[dict] = field(default_factory=list)
    missing_bins: list[tuple[int, int]] = field(default_factory=list)
    zero_bins: list[tuple[int, int]] = field(default_factory=list)
    imaginary_residual: float = 0.0


class FrequencyMixer:
    """可训练频域算子：一份权重存储 + 一个配置注册表。"""

    def __init__(
        self,
        weights: WeightStore,
        registry: ConfigRegistry,
        strict_tol: float = 1e-10,
        chunk_bins: int = 256,
    ):
        self.weights = weights
        self.registry = registry
        self.strict_tol = strict_tol
        self.chunk_bins = int(chunk_bins)

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def forward(
        self,
        image: np.ndarray,
        config_id: str,
        shape: tuple[int, int] | None = None,
        mode: str = ORTHOGONAL,
        budget: ScratchBudget | None = None,
    ) -> tuple[np.ndarray, MixerTape]:
        image = np.asarray(image, dtype=np.float64)
        if image.ndim != 3:
            raise ValueError("输入图像顺序为 [通道,高,宽]")
        c, h, w = image.shape
        if c != len(self.weights.input_order):
            raise ValueError("输入通道数与权重存储不一致")
        if shape is None:
            shape = (h, w)
        if tuple(shape) != (h, w):
            raise ValueError("shape 必须与图像实际高宽一致")
        plan = self.registry.plan(config_id, shape, mode, self.strict_tol)
        if plan.n_in != c:
            raise ValueError("计划输入通道数与图像不一致")

        budget = budget or ScratchBudget(None)
        kh = plan.half_width
        matrices = {
            wid: self.weights.complex_matrix(wid) for wid in self.weights.weight_ids
        }

        # 1) 前向 DFT（频域中间变量，计入预算）
        spectrum = budget.track(
            np.fft.rfft2(image, axes=(-2, -1)), "rfft2(X)"
        )

        # 2) strict 模式的数值合法性复核（结构检查在计划构建时完成）
        residual = 0.0
        if mode == STRICT:
            residual = self._strict_residual(plan, matrices)
            if residual > self.strict_tol:
                raise IllegalSpectrum(
                    f"特殊列标定不满足实值逆变换约束，残差 {residual:.3e}"
                    f" 超过容差 {self.strict_tol:.3e}"
                )

        # 3) 逐频点矩阵混合，按配方分组、分块执行
        y_spectrum = budget.track(
            np.zeros((plan.n_out, h, kh), dtype=np.complex128), "Y"
        )
        for group in plan.groups:
            matrix = group.recipe.assemble(matrices)
            idx = group.indices
            for start in range(0, len(idx), self.chunk_bins):
                sel = idx[start:start + self.chunk_bins]
                rows = sel[:, 0]
                cols = sel[:, 1]
                block = budget.track(
                    np.einsum(
                        "oi,ib->ob",
                        matrix,
                        spectrum[:, rows, cols],
                        optimize=True,
                    ),
                    "einsum Y block",
                )
                y_spectrum[:, rows, cols] = block
                budget.release(block)

        # 缺测频带与未列出零格：Y 保持零即可；再做一次实值合法性核对
        legality = self._check_real_spectrum(plan, y_spectrum)
        if legality > self.strict_tol:  # 理论上投影后不应发生
            raise IllegalSpectrum(
                f"输出半谱不满足共轭对称，逆变换虚部残差 {legality:.3e}"
            )

        # 4) 逆 DFT（输出本身不计预算，但 irfft 的内部临时量按一次谱规模估算）
        inverse_tmp = budget.track(
            np.empty_like(y_spectrum), "irfft2 temp (估算)"
        )
        output = np.fft.irfft2(y_spectrum, s=(h, w), axes=(-2, -1))
        budget.release(inverse_tmp)
        budget.release(y_spectrum)
        budget.release(spectrum)

        tape = MixerTape(
            config_id=config_id,
            shape=(h, w),
            mode=mode,
            version=plan.version,
            plan_key=plan.key,
            n_in=plan.n_in,
            n_out=plan.n_out,
            input_ref=image,
            projected=list(plan.projected_bins),
            missing_bins=list(plan.missing_bins),
            zero_bins=list(plan.zero_bins),
            imaginary_residual=float(residual),
        )
        return output, tape

    # ------------------------------------------------------------------
    # 反向
    # ------------------------------------------------------------------
    def backward(
        self,
        grad_output: np.ndarray,
        tape: MixerTape,
        budget: ScratchBudget | None = None,
        stale_policy: str = RAISE_STALE,
    ) -> tuple[np.ndarray, dict, dict, dict]:
        """返回 ``(输入图梯度, 权重实部梯度, 权重虚部梯度, 报告)``。

        配置失效时：``raise`` 直接抛 :class:`StaleConfig`；``void`` 返回
        零梯度并在报告 ``void`` 字段列出作废原因，梯度不写入任何物理权重。
        """
        budget = budget or ScratchBudget(None)
        void_report = None
        try:
            plan = self.registry.plan(
                tape.config_id, tape.shape, tape.mode, self.strict_tol
            )
        except (StaleConfig, KeyError) as exc:
            if stale_policy == RAISE_STALE:
                raise StaleConfig(str(exc)) from exc
            plan = None
            void_report = self._void_report(tape, str(exc))
        if plan is not None and (
            plan.version != tape.version or plan.key != tape.plan_key
        ):
            msg = (
                f"配置 {tape.config_id} {tape.shape} 的映射已变更"
                f"（前向版本 {tape.version}，当前 {plan.version}），"
                "缺原版信息，本次梯度作废"
            )
            if stale_policy == RAISE_STALE:
                raise StaleConfig(msg)
            plan = None
            void_report = self._void_report(tape, msg)

        if plan is None:
            out = self._zero_grad(tape)
            return (out["grad_input"], out["grad_real"],
                    out["grad_imag"], {"void": void_report})

        grad_output = np.asarray(grad_output, dtype=np.float64)
        h, w = tape.shape
        kh = plan.half_width
        if grad_output.shape != (tape.n_out, h, w):
            raise ValueError("空间梯度形状与前向输出不一致")

        matrices = {
            wid: self.weights.complex_matrix(wid) for wid in self.weights.weight_ids
        }

        # 1) 空间梯度 -> 半谱。两条反向路径缩放不同（全谱口径严格推出）：
        #    输入图梯度：carrier_x = rfft2(g)，最后 irfft2，全程无额外缩放；
        #    权重梯度：  carrier_w = scale * rfft2(g)，自共轭 1/N、其余 2/N。
        g_spectrum = budget.track(
            np.fft.rfft2(grad_output, axes=(-2, -1)), "rfft2(grad)"
        )

        # 2) 输入谱梯度 dX = M^H G'（缺测/零格自然不贡献）
        dx_spectrum = budget.track(
            np.zeros((tape.n_in, h, kh), dtype=np.complex128), "dX"
        )
        # 输入谱重算一次即可，分块只取切片（避免每块做一次完整 rfft）
        x_spectrum = budget.track(
            np.fft.rfft2(tape.input_ref, axes=(-2, -1)), "recomputed rfft2(X)"
        )
        grad_real = {
            wid: np.zeros((tape.n_out, tape.n_in), dtype=np.float64)
            for wid in self.weights.weight_ids
        }
        grad_imag = {
            wid: np.zeros((tape.n_out, tape.n_in), dtype=np.float64)
            for wid in self.weights.weight_ids
        }

        # 3) 逐组反传：矩阵梯度与输入谱梯度
        for group in plan.groups:
            matrix = group.recipe.assemble(matrices)
            idx = group.indices
            for start in range(0, len(idx), self.chunk_bins):
                sel = idx[start:start + self.chunk_bins]
                rows = sel[:, 0]
                cols = sel[:, 1]
                gx_block = g_spectrum[:, rows, cols]        # [o,B] 无缩放
                scale_block = plan.scale[rows, cols]        # [B]
                gw_block = budget.track(
                    gx_block * scale_block, "carrier weight block"
                )
                x_block = x_spectrum[:, rows, cols]       # [i, B]
                # dM = carrier_w @ X^H（复矩阵欧氏梯度，按实/虚变量解释）
                dm = budget.track(
                    np.einsum("ob,ib->oi", gw_block,
                              np.conj(x_block), optimize=True),
                    "dM block",
                )
                group.recipe.accumulate(dm, grad_real, grad_imag)
                dx = budget.track(
                    np.einsum("oi,ob->ib",
                              np.conj(matrix), gx_block, optimize=True),
                    "dX block",
                )
                dx_spectrum[:, rows, cols] += dx
                budget.release(dm)
                budget.release(dx)
                budget.release(gw_block)
        budget.release(x_spectrum)

        # 4) 半谱 -> 实值空间梯度（伴随的另一半：irfft 伴随作用于 dX）
        #    dX 在特殊列上已具共轭对称性，这里不再静默吞虚部
        residual = self._check_real_spectrum(plan, dx_spectrum)
        if residual > self.strict_tol:
            raise IllegalSpectrum(
                f"输入梯度半谱不满足共轭对称，残差 {residual:.3e}"
            )
        back_tmp = budget.track(np.empty_like(dx_spectrum), "irfft grad temp")
        grad_input = np.fft.irfft2(dx_spectrum, s=(h, w), axes=(-2, -1))
        budget.release(back_tmp)
        budget.release(dx_spectrum)
        budget.release(g_spectrum)

        # 5) 逻辑通道顺序 -> 物理通道顺序写回权重存储
        phys_gr, phys_gi = self._to_physical_order(grad_real, grad_imag, plan)
        report = {
            "void": None,
            "config_id": tape.config_id,
            "shape": list(tape.shape),
            "version": plan.version,
            "mode": tape.mode,
            "missing_bins": [list(b) for b in plan.missing_bins],
            "zero_bins": [list(b) for b in plan.zero_bins],
            "projected_bins": plan.projected_bins,
            "imaginary_residual": tape.imaginary_residual,
        }
        return grad_input, phys_gr, phys_gi, report

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _to_physical_order(self, grad_real, grad_imag, plan: Plan):
        out_perm = np.array(self.weights._out_perm)
        in_perm = np.array(self.weights._in_perm)
        phys_r, phys_i = {}, {}
        for wid in self.weights.weight_ids:
            phys_r[wid] = np.zeros_like(self.weights.real_part(wid))
            phys_i[wid] = np.zeros_like(self.weights.imag_part(wid))
            phys_r[wid][np.ix_(out_perm, in_perm)] = grad_real[wid]
            phys_i[wid][np.ix_(out_perm, in_perm)] = grad_imag[wid]
        return phys_r, phys_i

    def _zero_grad(self, tape: MixerTape) -> dict:
        h, w = tape.shape
        return {
            "grad_input": np.zeros((tape.n_in, h, w), dtype=np.float64),
            "grad_real": {
                wid: np.zeros_like(self.weights.real_part(wid))
                for wid in self.weights.weight_ids
            },
            "grad_imag": {
                wid: np.zeros_like(self.weights.imag_part(wid))
                for wid in self.weights.weight_ids
            },
        }

    def _void_report(self, tape: MixerTape, reason: str) -> dict:
        return {
            "voided": True,
            "config_id": tape.config_id,
            "shape": list(tape.shape),
            "version": tape.version,
            "reason": reason,
        }

    def _check_real_spectrum(self, plan: Plan, half: np.ndarray) -> float:
        """检查半谱在特殊列上的共轭对称残差（Frobenius 归一化）。"""
        h, w = plan.height, plan.width
        residual = 0.0
        # 自共轭点虚部必须为零
        if plan.self_mask.any():
            residual += float(np.sum(np.abs(half[:, plan.self_mask].imag) ** 2))
        # 特殊列非自反行对：Y[-ky] == conj(Y[ky])
        bin_set = {(b.ky, b.kx) for b in plan.bins}
        for kx in plan_bins_special_columns(plan):
            for ky in range(h):
                partner = (-ky) % h
                if partner <= ky:
                    continue
                if (ky, kx) not in bin_set and (partner, kx) not in bin_set:
                    continue
                diff = half[:, ky, kx] - np.conj(half[:, partner, kx])
                residual += float(np.sum(np.abs(diff) ** 2))
        scale = float(np.sum(np.abs(half) ** 2)) + 1e-30
        return float(np.sqrt(residual / scale))

    def _strict_residual(self, plan: Plan, matrices: dict) -> float:
        """strict 模式下基于**原始标定配方**复核特殊列合法性。

        不能用执行用的投影配方（自共轭点已取实部、行对已被平均），否则
        非法虚部会在检查前被抹掉。raw_recipe 保留投影前的标定结构。
        """
        residual = 0.0
        h, w = plan.height, plan.width
        raw_lookup = {(b.ky, b.kx): b.raw_recipe for b in plan.bins}
        for kx in plan_bins_special_columns(plan):
            for ky in range(h):
                partner = (-ky) % h
                if partner == ky:
                    raw = raw_lookup.get((ky, kx))
                    if raw is None or not raw.terms:
                        continue
                    m = raw.assemble(matrices)
                    residual += float(np.sum(np.abs(m.imag) ** 2))
                elif partner > ky:
                    r1 = raw_lookup.get((ky, kx))
                    r2 = raw_lookup.get((partner, kx))
                    if r1 is None and r2 is None:
                        continue
                    if r1 is None or r2 is None:
                        continue  # 单边标定：共轭补出本身合法
                    m1 = r1.assemble(matrices)
                    m2 = r2.assemble(matrices)
                    residual += float(
                        np.sum(np.abs(m1 - np.conj(m2)) ** 2)
                    )
        return float(np.sqrt(residual))


def plan_bins_special_columns(plan: Plan) -> tuple[int, ...]:
    cols = [0]
    if plan.width % 2 == 0:
        cols.append(plan.width // 2)
    return tuple(cols)
