# 频域通道混合算子（freqmix）

可训练的光学半谱交换算子：实值图像 → `rfft2` 半谱 → 逐频点复矩阵通道混合
→ 合法共轭对称的半谱 → `irfft2` 回到原尺寸实值空间。全程双精度，
DFT 采用 numpy 约定（前向无缩放、逆变换除以面积）。

## 模块

| 文件 | 职责 |
| --- | --- |
| `freqmix/budget.py` | 调用方字节预算与工作区高水位，超预算在分配前拒绝 |
| `freqmix/contract.py` | 资料解析、频率身份判定、共轭合法性、strict/orthogonal 计划 |
| `freqmix/weights.py` | 独立物理通道矩阵（实部/虚部两个实变量）与梯度累计 |
| `freqmix/operator.py` | 单次前向/反向、配置版本缓存与作废 |
| `freqmix/batch.py` | 混合尺寸成批处理，峰值由单样本决定 |
| `freqmix/audit.py` | 现场核对：生效配置、独立权重、受影响频带 |
| `freqmix/synthetic.py` | 契约规定的可重复小图与大规格合成负载 |

## 契约语义落地

- **频率身份**由 `(配置 ID, 高, 宽)` 共同决定，并带内容版本哈希。偶数裁边的
  Nyquist 列（`kx=W/2`）是特殊列；奇数裁边没有该列，其最后一列是普通内部
  频点。二者绝不会复用同一条缓存映射。
- **三类频点分开记账**：
  - `missing`：缺测频带，输入谱不贡献，特殊列上整对作废；
  - `zero_calibration_bins`：未列出的半谱格，标定明确不传递的零；
  - 正常列出的格：即使组装出的矩阵数值恰好为零，也仍是有观测的权重。
- **特殊列实值合法性**：
  - 自共轭点（`-ky≡ky`）矩阵必须为实；orthogonal 取实部，strict 校验数值。
  - 非自反行按 `(ky,-ky)` 共轭配对；两行都有标定时，orthogonal 投影为
    `(M_ky + conj(M_partner))/2`，单边标定时按共轭补出；strict 在无法
    组成合法实图权重时直接抛 `IllegalSpectrum`。
  - 正向采用的独立参数映射与反向完全一致（同一份配方）。
- **方向 `dir=-1`**：组合值逐元素共轭，只在配方解析时应用一次。
- **stencil 插值**（如 `pq=0.75p+0.25q`）是独立权重的实线性组合，梯度按
  系数分摊回 `p/q`，插值矩阵本身不是可更新参数。

## 反向口径（与双精度扰动一致，atol 1e-7 / rtol 1e-5）

从实值空间梯度 `g` 出发，两条路径缩放不同：

- 输入图梯度：`carrier = rfft2(g)`，`dX = Mᴴ·carrier`，`dx = irfft2(dX)`，
  全程无额外缩放；
- 权重梯度：`carrier_w = scale·rfft2(g)`，`dM = carrier_w·Xᴴ`，其中
  内部频点 `scale=2/N`，特殊列（自共轭点与配对行）`scale=1/N`。

自共轭点的虚部自由度被投影移除，该位置虚部梯度恒为零，不会在自共轭位置
加倍。同一物理权重被多个频点/两种裁边引用时，各贡献按真实配置累加到同一
独立权重。

## 配置变更与作废

`MixerTape` 只记录配置身份、形状与版本哈希，不持有计划对象。换光学配置后
同数组索引可对应不同物理频率：

- 注册同身份同形状但内容不同的配置会产生新版本；旧 tape 反向时
  `stale_policy="raise"` 抛 `StaleConfig`，`"void"` 返回零梯度并在
  报告 `void` 字段说明受影响样本，绝不写入错误梯度；
- `revoke()` 模拟现场卸载配置，缺原版信息时同样作废。

## 工作区预算

预算只计频域中间变量，不含原始只读影像与已交付输出。反向把权重 carrier
按块即时缩放、不单独占一份完整半谱，峰值约为 3 份半谱加小分块。
fixture 的 12 张 6 通道 384/385 混合尺寸负载：

- 预算 24 MiB，实测峰值约 20.4 MiB；
- 不构造完整复谱外积，单 einsum 分块约 0.01 MiB 量级。

## 最小用法

```python
from freqmix import (ConfigRegistry, WeightStore, FrequencyMixer,
                     ScratchBudget, run_batch, BatchItem)

registry = ConfigRegistry.from_session(session)
weights = WeightStore(session["weights"],
                      tuple(session["input_order"]),
                      tuple(session["output_order"]),
                      physical_input_order=("VIS", "IR"))  # 相机实际通道顺序
mixer = FrequencyMixer(weights, registry)

budget = ScratchBudget(24 * 1024 * 1024)
y, tape = mixer.forward(image, config_id="A", mode="orthogonal", budget=budget)
dx, grad_real, grad_imag, report = mixer.backward(grad_output, tape, budget)
```

## 工程师导数自检

`tests/test_gradient.py` 用实值中心差分（h=1e-6）核对输入图与每个权重的
实部/虚部梯度；`tests/test_forward.py` 用逐频点朴素全谱参考实现交叉验证
前向。运行：

```
python3 -m unittest discover -s tests -v
```
