"""独立物理通道权重的存储。

契约：权重 ID 指向“可训练的物理通道矩阵”，矩阵按逻辑通道名称寻址，
每项是 ``[实部, 虚部]``。插值（stencil）与方向共轭都只是这些独立矩阵的
线性组合，本身**不是**可独立更新的参数。

因此这里把每个权重拆成两个双精度实变量（实部、虚部），梯度也按两个
实变量分别累计，和契约“回传按实部与虚部各自的实变量欧氏梯度”一致。
权重与梯度属于持久参数，不计入调用方的临时工作区预算。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WeightSpec:
    weight_id: str
    n_out: int
    n_in: int


class WeightStore:
    """按权重 ID 持有独立物理通道矩阵。"""

    def __init__(
        self,
        weights: dict[str, np.ndarray],
        input_order: tuple[str, ...],
        output_order: tuple[str, ...],
        physical_input_order: tuple[str, ...] | None = None,
        physical_output_order: tuple[str, ...] | None = None,
    ):
        self.input_order = tuple(input_order)
        self.output_order = tuple(output_order)
        # 相机实际交来的通道顺序可能与逻辑顺序不同；记录物理排列，
        # 取矩阵时按逻辑名称重排。
        self.physical_input_order = tuple(
            physical_input_order or input_order
        )
        self.physical_output_order = tuple(
            physical_output_order or output_order
        )
        if sorted(self.physical_input_order) != sorted(self.input_order):
            raise ValueError("物理输入通道顺序必须是逻辑名称的一个排列")
        if sorted(self.physical_output_order) != sorted(self.output_order):
            raise ValueError("物理输出通道顺序必须是逻辑名称的一个排列")

        self.specs: dict[str, WeightSpec] = {}
        self._real: dict[str, np.ndarray] = {}
        self._imag: dict[str, np.ndarray] = {}
        for weight_id, matrix in weights.items():
            m = np.asarray(matrix, dtype=np.float64)
            if m.shape != (len(self.output_order), len(self.input_order), 2):
                raise ValueError(
                    f"权重 {weight_id} 形状应为 "
                    f"({len(self.output_order)}, "
                    f"{len(self.input_order)}, 2)，实际为 {m.shape}"
                )
            self.specs[weight_id] = WeightSpec(
                weight_id, m.shape[0], m.shape[1]
            )
            # copy 保证外部改动不会污染参数；用独立的实变量存放
            self._real[weight_id] = np.array(m[..., 0], dtype=np.float64)
            self._imag[weight_id] = np.array(m[..., 1], dtype=np.float64)

        self._in_perm = tuple(
            self.physical_input_order.index(name) for name in self.input_order
        )
        self._out_perm = tuple(
            self.physical_output_order.index(name) for name in self.output_order
        )
        self.zero_grad()

    # ---- 基本属性 ------------------------------------------------------
    @property
    def weight_ids(self) -> tuple[str, ...]:
        return tuple(self._real.keys())

    def complex_matrix(self, weight_id: str) -> np.ndarray:
        """返回按逻辑通道顺序寻址的复矩阵（只读语义的视图）。"""
        if weight_id not in self._real:
            raise KeyError(f"未知权重 ID: {weight_id}")
        m = self._real[weight_id] + 1j * self._imag[weight_id]
        # 物理顺序 -> 逻辑顺序
        return m[np.ix_(self._out_perm, self._in_perm)]

    def real_part(self, weight_id: str) -> np.ndarray:
        return self._real[weight_id]

    def imag_part(self, weight_id: str) -> np.ndarray:
        return self._imag[weight_id]

    # ---- 梯度 ----------------------------------------------------------
    def zero_grad(self) -> None:
        self.grad_real: dict[str, np.ndarray] = {
            wid: np.zeros_like(part) for wid, part in self._real.items()
        }
        self.grad_imag: dict[str, np.ndarray] = {
            wid: np.zeros_like(part) for wid, part in self._imag.items()
        }

    def add_grad(self, weight_id: str, grad_real: np.ndarray,
                 grad_imag: np.ndarray) -> None:
        """累计物理权重格上的欧氏梯度（已按物理通道顺序写回）。"""
        if weight_id not in self._real:
            raise KeyError(f"未知权重 ID: {weight_id}")
        self.grad_real[weight_id] += grad_real
        self.grad_imag[weight_id] += grad_imag

    def parameters(self) -> dict[str, np.ndarray]:
        """供优化器使用的实变量视图。"""
        return {
            wid: (self._real[wid], self._imag[wid]) for wid in self.weight_ids
        }

    def gradients(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {
            wid: (self.grad_real[wid], self.grad_imag[wid])
            for wid in self.weight_ids
        }

    def snapshot(self) -> dict[str, np.ndarray]:
        """参数快照，用于有限差分等场景。"""
        return {
            wid: np.stack([self._real[wid], self._imag[wid]], axis=-1).copy()
            for wid in self.weight_ids
        }

    def restore(self, snapshot: dict[str, np.ndarray]) -> None:
        for wid, value in snapshot.items():
            self._real[wid][...] = value[..., 0]
            self._imag[wid][...] = value[..., 1]
