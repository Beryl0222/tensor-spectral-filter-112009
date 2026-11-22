"""契约规定的可重复合成负载。

小图规则（session.json 的 image_rule）::

    pixel[i]     = (i*rule[0] % rule[1] - floor(rule[1]/2)) / 2**rule[2]
    cotangent[i] = (i*7 % 11 - 5) / 8

索引 ``i`` 沿 ``[C, H, W]`` 的 C 顺序计数。

大规格负载（contract.md 第 5 段）：

* 第 i 个样本 ``rule = [17+2*i, 101, 4]``，通道沿 C 顺序计数；
* 物理矩阵的 ``(o, i)`` 元素在 2×2 参数网格中按两轴通道编号模二寻址；
* 除特殊列外频点共享 ``p``；特殊列非自反行使用 ``p`` 并按行对选择
  方向，自反行使用 ``d``（取实部）。
"""

from __future__ import annotations

import numpy as np

from .contract import ORTHOGONAL, OpticalConfig


# --------------------------------------------------------------------------
# 小图（session.json 口径）
# --------------------------------------------------------------------------
def pixel_values(rule: tuple[int, int, int], size: int) -> np.ndarray:
    a, b, shift = rule
    idx = np.arange(size, dtype=np.int64)
    return ((idx * a) % b - b // 2) / (2 ** shift)


def cotangent_values(size: int) -> np.ndarray:
    idx = np.arange(size, dtype=np.int64)
    return ((idx * 7) % 11 - 5) / 8.0


def make_record_image(
    rule: tuple[int, int, int], channels: int, height: int, width: int
) -> np.ndarray:
    total = channels * height * width
    flat = pixel_values(rule, total)
    return flat.reshape(channels, height, width).astype(np.float64)


def make_record_grad(
    channels: int, height: int, width: int
) -> np.ndarray:
    total = channels * height * width
    flat = cotangent_values(total)
    return flat.reshape(channels, height, width).astype(np.float64)


def build_session_inputs(session: dict) -> dict:
    """按 records 生成每张训练图及其实值空间梯度。"""
    out = {}
    channels_in = len(session["input_order"])
    channels_out = len(session["output_order"])
    for rec in session["records"]:
        h, w = rec["shape"]
        out[rec["id"]] = {
            "image": make_record_image(
                tuple(rec["rule"]), channels_in, h, w
            ),
            "grad_output": make_record_grad(channels_out, h, w),
            "config": rec["config"],
            "shape": tuple(rec["shape"]),
            "mode": rec["mode"],
        }
    return out


# --------------------------------------------------------------------------
# 大规格可重复相机负载
# --------------------------------------------------------------------------
def _tiled_2x2(blocks: np.ndarray, n_out: int, n_in: int) -> np.ndarray:
    """把 2×2 通道块按通道编号模二铺成 n_out×n_in 矩阵。

    blocks[k, l] 是输出通道 o≡k (mod 2)、输入通道 i≡l (mod 2) 时
    使用的复标量（2 元素 [实, 虚]）。
    """
    matrix = np.empty((n_out, n_in), dtype=np.complex128)
    for o in range(n_out):
        for i in range(n_in):
            v = blocks[o % 2, i % 2]
            matrix[o, i] = complex(v[0], v[1])
    return matrix


def large_profile_weights(profile: dict) -> dict[str, np.ndarray]:
    """生成 2×2 参数网格 p、d（元素为 [实, 虚]，再按模二铺开）。"""
    n_out = profile["output_channels"]
    n_in = profile["input_channels"]
    rng = np.random.default_rng(20261122)

    def random_block():
        block = np.empty((2, 2, 2), dtype=np.float64)
        block[..., 0] = rng.standard_normal((2, 2))
        block[..., 1] = rng.standard_normal((2, 2))
        return block

    weights = {}
    for name in ("p", "d"):
        block = random_block()
        full = np.empty((n_out, n_in, 2), dtype=np.float64)
        matrix = _tiled_2x2(block, n_out, n_in)
        full[..., 0] = matrix.real
        full[..., 1] = matrix.imag
        weights[name] = full
    # 自共轭点使用 d 的实部：把 d 的虚部置零是该结构的合法标定方式
    weights["d"][..., 1] = 0.0
    return weights


def large_profile_grid(height: int, width: int) -> list[list]:
    """半谱格表：内部列共享 p；特殊列非自反行 p ±方向，自反行 d。"""
    kh = width // 2 + 1
    special = [0] + ([width // 2] if width % 2 == 0 else [])
    grid = []
    for ky in range(height):
        for kx in range(kh):
            if kx in special:
                partner = (-ky) % height
                if partner == ky:
                    grid.append([ky, kx, "d", 1])
                elif partner > ky:
                    # 行对只声明一次：本行 +1，配对行由共轭关系给出
                    grid.append([ky, kx, "p", 1])
            else:
                grid.append([ky, kx, "p", 1])
    return grid


def build_large_config(
    config_id: str, shape: tuple[int, int], channel_names: list[str]
) -> OpticalConfig:
    grid = large_profile_grid(shape[0], shape[1])
    return OpticalConfig(
        config_id=config_id,
        shape=shape,
        grid=grid,
        missing=[],
        stencils={},
        input_order=tuple(channel_names),
        output_order=tuple(channel_names),
        weight_ids=("p", "d"),
    )


def build_large_load(profile: dict) -> dict:
    """生成契约规定的整套大规格负载。"""
    names = [f"C{c}" for c in range(profile["input_channels"])]
    configs = []
    items = []
    count = profile["count"]
    for i in range(count):
        shape = tuple(profile["shapes"][i % len(profile["shapes"])])
        rule = (17 + 2 * i, 101, 4)
        image = make_record_image(
            rule, profile["input_channels"], shape[0], shape[1]
        )
        grad = make_record_grad(
            profile["output_channels"], shape[0], shape[1]
        )
        config_id = f"L-{shape[0]}x{shape[1]}"
        configs.append((config_id, shape))
        items.append(
            {
                "index": i,
                "config_id": config_id,
                "shape": shape,
                "mode": ORTHOGONAL,
                "rule": list(rule),
                "image": image,
                "grad_output": grad,
            }
        )
    unique_configs = sorted(set(configs))
    return {
        "items": items,
        "configs": [
            build_large_config(cid, shape, names)
            for cid, shape in unique_configs
        ],
        "channel_names": names,
        "scratch_bytes": profile["scratch_bytes"],
    }
