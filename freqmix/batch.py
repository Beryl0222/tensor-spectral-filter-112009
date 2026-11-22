"""混合尺寸成批处理。

输入样本可以有不同高宽；每张图按自己的（配置, 形状, 模式）取计划，
顺序处理而不拼全批复谱——临时张量峰值由单个样本决定，受调用方
:class:`~freqmix.budget.ScratchBudget` 约束。权重梯度在物理权重格上
跨样本累计；同一独立权重被两份裁边引用时自然归并，不会重复计数。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .budget import ScratchBudget
from .operator import FrequencyMixer, MixerTape, RAISE_STALE


@dataclass
class BatchItem:
    image: np.ndarray
    config_id: str
    mode: str = "orthogonal"
    shape: tuple[int, int] | None = None
    grad_output: np.ndarray | None = None
    record_id: str | None = None


@dataclass
class BatchResult:
    outputs: list[np.ndarray]
    tapes: list[MixerTape]
    grad_inputs: list[np.ndarray]
    weight_grad_real: dict
    weight_grad_imag: dict
    reports: list[dict]
    budget_summary: dict
    voided: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.voided


def run_batch(
    mixer: FrequencyMixer,
    items: list[BatchItem],
    byte_limit: int | None,
    backward: bool = True,
    accumulate: bool = True,
    stale_policy: str = RAISE_STALE,
) -> BatchResult:
    """成批执行前向（可选反向）。

    参数
    ----
    byte_limit:
        所有频域中间变量共享的字节预算；逐样本复用同一个记账器，
        高水位即本批真实工作区峰值。
    accumulate:
        True 时把权重梯度累计进 ``mixer.weights`` 并返回累计值；
        False 时只返回本批合计、不改动存储中的梯度。
    """
    budget = ScratchBudget(byte_limit)
    outputs, tapes, grad_inputs, reports = [], [], [], []
    voided: list[dict] = []

    grad_real_total = {
        wid: np.zeros_like(mixer.weights.real_part(wid))
        for wid in mixer.weights.weight_ids
    }
    grad_imag_total = {
        wid: np.zeros_like(mixer.weights.imag_part(wid))
        for wid in mixer.weights.weight_ids
    }

    for i, item in enumerate(items):
        output, tape = mixer.forward(
            item.image,
            item.config_id,
            shape=item.shape,
            mode=item.mode,
            budget=budget,
        )
        outputs.append(output)
        tapes.append(tape)

        if backward:
            if item.grad_output is None:
                raise ValueError(f"第 {i} 个样本缺少空间梯度 grad_output")
            dx, gr, gi, report = mixer.backward(
                item.grad_output,
                tape,
                budget=budget,
                stale_policy=stale_policy,
            )
            grad_inputs.append(dx)
            reports.append(report)
            if report["void"] is not None:
                entry = dict(report["void"])
                entry["item"] = item.record_id if item.record_id is not None else i
                voided.append(entry)
                continue
            for wid in grad_real_total:
                grad_real_total[wid] += gr[wid]
                grad_imag_total[wid] += gi[wid]
        else:
            grad_inputs.append(None)
            reports.append({"void": None})

    if accumulate and backward:
        for wid in mixer.weights.weight_ids:
            mixer.weights.add_grad(
                wid, grad_real_total[wid], grad_imag_total[wid]
            )

    return BatchResult(
        outputs=outputs,
        tapes=tapes,
        grad_inputs=grad_inputs,
        weight_grad_real=grad_real_total,
        weight_grad_imag=grad_imag_total,
        reports=reports,
        budget_summary=budget.summary(),
        voided=voided,
    )
