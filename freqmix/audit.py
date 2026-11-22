"""服务现场核对：实际生效的光学配置、独立权重与受影响频带。"""

from __future__ import annotations

import numpy as np

from .contract import ConfigRegistry, Plan
from .operator import FrequencyMixer


def plan_report(plan: Plan, registry: ConfigRegistry) -> dict:
    """单个计划的可核对报告。"""
    config = registry.get_config(plan.config_id, (plan.height, plan.width))
    used_weights = sorted({
        term.weight_id
        for group in plan.groups
        for term in group.recipe.terms
    })
    return {
        "config_id": plan.config_id,
        "config_version": plan.version,
        "shape": [plan.height, plan.width],
        "half_width": plan.half_width,
        "mode": plan.mode,
        "input_order": list(config.input_order),
        "output_order": list(config.output_order),
        "special_columns": list(config.special_columns()),
        "independent_weights": used_weights,
        "stencils": config.stencils,
        "active_bins": [
            [b.ky, b.kx, "self" if b.is_self else
             ("special-pair" if b.is_special else "interior")]
            for b in sorted(plan.bins, key=lambda b: (b.kx, b.ky))
        ],
        "missing_bins": [list(b) for b in plan.missing_bins],
        "zero_calibration_bins": [list(b) for b in plan.zero_bins],
        "projected_bins": plan.projected_bins,
        "violations": plan.violations,
        "n_groups": len(plan.groups),
    }


def operator_report(mixer: FrequencyMixer) -> dict:
    """算子级核对：注册中的配置、缓存计划、独立权重清单与值指纹。"""
    import hashlib

    plans = []
    for key in sorted(mixer.registry._plans.keys()):
        plan = mixer.registry._plans[key]
        plans.append(plan_report(plan, mixer.registry))

    weight_fingerprints = {}
    for wid in mixer.weights.weight_ids:
        m = mixer.weights.complex_matrix(wid)
        digest = hashlib.sha256(
            np.ascontiguousarray(m).view(np.float64).tobytes()
        ).hexdigest()[:16]
        weight_fingerprints[wid] = {
            "shape": list(m.shape),
            "sha256_16": digest,
            "frobenius": float(np.linalg.norm(m)),
        }

    return {
        "input_order": list(mixer.weights.input_order),
        "output_order": list(mixer.weights.output_order),
        "physical_input_order": list(mixer.weights.physical_input_order),
        "physical_output_order": list(mixer.weights.physical_output_order),
        "independent_weights": list(mixer.weights.weight_ids),
        "weight_fingerprints": weight_fingerprints,
        "active_configs": [list(k) for k in mixer.registry.active_keys],
        "cached_plans": plans,
        "strict_tol": mixer.strict_tol,
    }


def print_report(report: dict) -> str:
    """便于现场人工核对的纯文本渲染。"""
    lines = []
    for key, value in report.items():
        if isinstance(value, (dict, list)):
            import json
            rendered = json.dumps(value, ensure_ascii=False, indent=2,
                                  default=_json_default)
            lines.append(f"{key}:\n{rendered}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
