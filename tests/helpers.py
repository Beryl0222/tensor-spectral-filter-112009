"""测试共享夹具与朴素参考实现。

参考实现故意用最直白的逐频点 Python 循环 + 全谱重建，便于人工核对；
算子本体走分组/分块向量化路径，两者结果必须一致。
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freqmix.contract import ConfigRegistry, ORTHOGONAL, STRICT  # noqa: E402
from freqmix.weights import WeightStore  # noqa: E402
from freqmix.operator import FrequencyMixer  # noqa: E402
from freqmix.synthetic import build_session_inputs  # noqa: E402
from inspect_fixture import read_session  # noqa: E402


def load():
    session = read_session()
    registry = ConfigRegistry.from_session(session)
    weights = WeightStore(
        session["weights"],
        tuple(session["input_order"]),
        tuple(session["output_order"]),
    )
    mixer = FrequencyMixer(weights, registry)
    inputs = build_session_inputs(session)
    return session, registry, weights, mixer, inputs


def _resolve_matrix(config, matrices, wid):
    if wid in config.stencils:
        return sum(c * matrices[b] for b, c in config.stencils[wid])
    return matrices[wid]


def naive_forward(image, matrices, config, mode=ORTHOGONAL, tol=1e-10):
    """全谱显式重建的参考前向。matrices: id -> 逻辑顺序复矩阵。

    特殊列严格按契约处理：自反行取实部；行对在 orthogonal 下投影为
    ``(M1 + conj(M2))/2``，strict 下要求本来就共轭。
    """
    h, w = image.shape[-2:]
    kh = w // 2 + 1
    x_full = np.fft.fft2(image, axes=(-2, -1))
    y_full = np.zeros_like(x_full)
    grid = {(r[0], r[1]): r for r in config.grid}
    missing = config.missing
    special = set(config.special_columns())

    def half_matrix(ky, kx):
        row = grid.get((ky, kx))
        if row is None:
            return None
        m = _resolve_matrix(config, matrices, row[2])
        return np.conj(m) if row[3] == -1 else m

    # 先在半谱上确定每个非缺测频点的最终矩阵（含特殊列投影）
    half_matrix_map: dict[tuple[int, int], np.ndarray] = {}
    for ky in range(h):
        for kx in range(kh):
            if (ky, kx) in missing:
                continue
            m = half_matrix(ky, kx)
            if kx not in special:
                if m is not None:
                    half_matrix_map[(ky, kx)] = m
                continue
            partner = (-ky) % h
            if partner == ky:
                if m is not None:
                    half_matrix_map[(ky, kx)] = m.real + 0j
                continue
            if partner < ky:
                continue  # 行对只处理一次
            m2 = half_matrix(partner, kx)
            if m is None and m2 is None:
                continue
            if m is None:
                m1 = np.conj(m2)
            elif m2 is None:
                m1 = m.copy()
            else:
                if mode == STRICT:
                    assert np.allclose(m2, np.conj(m), atol=tol)
                else:
                    m1 = 0.5 * (m + np.conj(m2))
            half_matrix_map[(ky, kx)] = m1
            half_matrix_map[(partner, kx)] = np.conj(m1)

    for ky in range(h):
        for kx_full in range(w):
            kx = kx_full if kx_full < kh else w - kx_full
            ky_eff = ky if kx_full < kh else (-ky) % h
            m = half_matrix_map.get((ky_eff, kx))
            if m is None:
                continue
            if kx_full >= kh:
                m = np.conj(m)
            y_full[:, ky, kx_full] = m @ x_full[:, ky, kx_full]

    y = np.fft.ifft2(y_full, axes=(-2, -1))
    resid = float(np.max(np.abs(y.imag)))
    if resid > 1e-8 and mode == STRICT:
        raise AssertionError(f"参考实现检测到非法虚部 {resid}")
    return y.real, resid
