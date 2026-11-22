"""可训练的频域通道混合算子（光学半谱交换）。

模块划分：

* :mod:`freqmix.budget`    —— 调用方字节预算与工作区高水位统计
* :mod:`freqmix.contract`  —— 资料/契约解析、光学配置注册与频率身份判定
* :mod:`freqmix.weights`   —— 独立物理通道矩阵（实部/虚部两个实变量）
* :mod:`freqmix.operator`  —— 单次前向/反向算子与配置缓存
* :mod:`freqmix.batch`     —— 混合尺寸成批处理
* :mod:`freqmix.audit`     —— 现场核对：生效配置、独立权重、受影响频带
* :mod:`freqmix.synthetic` —— 契约规定的可重复合成负载
"""

from .budget import ScratchBudget, WorkspaceExceeded
from .contract import (
    ContractError,
    IllegalSpectrum,
    StaleConfig,
    OpticalConfig,
    ConfigRegistry,
)
from .weights import WeightStore
from .operator import FrequencyMixer, MixerTape
from .batch import BatchResult, run_batch
from .audit import plan_report, operator_report

__all__ = [
    "ScratchBudget",
    "WorkspaceExceeded",
    "ContractError",
    "IllegalSpectrum",
    "StaleConfig",
    "OpticalConfig",
    "ConfigRegistry",
    "WeightStore",
    "FrequencyMixer",
    "MixerTape",
    "BatchResult",
    "run_batch",
    "plan_report",
    "operator_report",
]
