"""调用方字节预算与工作区高水位统计。

契约规定：预算不计原始只读影像和已交付输出，但包括正反向频域中间变量。
本模块提供一个显式的工作区记账器；算子内部所有临时数组都通过
:meth:`ScratchBudget.track` 登记，超预算直接抛出 :class:`WorkspaceExceeded`，
而不是先铺满内存再失败。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import numpy as np


class WorkspaceExceeded(RuntimeError):
    """临时张量超过调用方给出的字节预算。"""

    def __init__(self, needed: int, limit: int, peak: int, note: str = ""):
        super().__init__(
            f"工作区超预算：本次需要 {needed} 字节，预算 {limit} 字节，"
            f"当前高水位 {peak} 字节。{note}".rstrip()
        )
        self.needed = needed
        self.limit = limit
        self.peak = peak


class ScratchBudget:
    """记录频域中间变量占用的工作区。

    参数
    ----
    byte_limit:
        调用方允许的临时张量总字节上限（高水位口径）。``None`` 表示不限。
    """

    def __init__(self, byte_limit: int | None):
        self.byte_limit = None if byte_limit is None else int(byte_limit)
        self.peak = 0          # 历史高水位（含已释放的并发量估算）
        self.live = 0          # 当前存活的临时张量字节
        self.peak_live = 0     # 同时存活字节的真实高水位
        self.events: list[tuple[str, int, int]] = []

    @staticmethod
    def nbytes(shape: Iterable[int], dtype: np.dtype) -> int:
        itemsize = np.dtype(dtype).itemsize
        total = itemsize
        for dim in shape:
            total *= int(dim)
        return total

    def track(self, array: np.ndarray, note: str = "") -> np.ndarray:
        """登记一个临时数组；超预算立即拒绝，不返回该数组。"""
        size = array.nbytes
        if self.byte_limit is not None and self.live + size > self.byte_limit:
            raise WorkspaceExceeded(size, self.byte_limit, self.peak_live, note)
        self.live += size
        self.peak_live = max(self.peak_live, self.live)
        self.peak = max(self.peak, self.peak_live)
        self.events.append((note or "array", size, self.live))
        return array

    def release(self, array: np.ndarray) -> None:
        """标记一个临时数组不再存活。"""
        self.live -= int(array.nbytes)
        if self.live < 0:  # 防御性记账错误
            self.live = 0

    @contextmanager
    def scoped(self, note: str = ""):
        """上下文：退出时把块内仍存活的登记数组全部释放。"""
        before = len(self.events)
        live_before = self.live
        try:
            yield self
        finally:
            # 块内未显式 release 的部分按事件回滚
            while len(self.events) > before:
                _, size, _ = self.events.pop()
                self.live -= size
            self.live = live_before

    def reset(self) -> None:
        self.peak = 0
        self.live = 0
        self.peak_live = 0
        self.events.clear()

    def summary(self) -> dict:
        return {
            "byte_limit": self.byte_limit,
            "peak_live_bytes": self.peak_live,
            "current_live_bytes": self.live,
            "within_budget": self.byte_limit is None
            or self.peak_live <= self.byte_limit,
            "events": list(self.events),
        }
