"""资料/契约解析、光学配置注册与频率身份判定。

频率身份判定遵循契约：

* 图像顺序 ``[通道, 高, 宽]``；只存 ``rfft2`` 的水平非负半谱，
  半谱宽 ``floor(宽/2)+1``。
* ``maps`` 以 ``[高, 宽]`` 顺序关联配置与格点，频点用 DFT 整数索引
  ``[ky, kx]`` 表示——*尺寸描述符和配置都参与频率身份判定*，因此换光学
  配置后同数组索引可以指向不同物理频率，缓存键必须同时含配置与形状。
* 未列出的半谱格解释为“标定明确不传递的零”，与 ``missing`` 缺测频带是
  两种不同的量；``missing`` 频带不参与正反向并单独报告。
* 特殊列（``kx=0`` 及偶数宽的 Nyquist 列）上的频点要满足共轭约束才能
  组成合法实图：自共轭点必须为实矩阵，其余行按 ``(ky, -ky)`` 共轭配对。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

# 交换模式
STRICT = "strict"
ORTHOGONAL = "orthogonal"
MODES = (STRICT, ORTHOGONAL)

# 特殊列上“没有合法对应系数”时的处理结果标记
OK = "ok"                  # 直接采用标定
PROJECT_REAL = "project_real"        # 自共轭点取实部（正交投影）
PROJECT_PAIR = "project_pair"        # 行对投影到共轭约束子空间
ILLEGAL = "illegal"        # strict 下拒绝
MISSING_BIN = "missing"    # 缺测频带，整对作废
ZERO_CALIBRATION = "zero_calibration"  # 标定明确不传递的零


class ContractError(ValueError):
    """资料不满足契约（字段缺失、形状不符、引用未知权重等）。"""


class IllegalSpectrum(ValueError):
    """标定无法组成合法实值逆变换，且当前模式不允许投影。"""


class StaleConfig(RuntimeError):
    """反向时原光学配置映射已被撤销/替换，缺原版信息，梯度作废。"""


# --------------------------------------------------------------------------
# 配方（recipe）：一个频点处输出谱关于独立物理权重的实线性组合
# --------------------------------------------------------------------------
@dataclass
class Term:
    """``coef * (M_wid 或 conj(M_wid))``，coef 为实系数。"""

    weight_id: str
    coef: float
    conjugate: bool

    def signature(self) -> tuple:
        return (self.weight_id, round(float(self.coef), 15), self.conjugate)


@dataclass
class Recipe:
    terms: list[Term] = field(default_factory=list)
    real_only: bool = False          # 自共轭点：组装/回传都丢弃虚部
    kind: str = OK                   # 见模块级标记
    note: str = ""

    def signature(self) -> tuple:
        return (
            tuple(sorted(t.signature() for t in self.terms)),
            self.real_only,
        )

    def assemble(self, matrices: dict[str, np.ndarray]) -> np.ndarray:
        """按配方组装复矩阵。空配方返回零矩阵（需要调用方给定形状）。"""
        if not self.terms:
            raise ValueError("空配方无法推断矩阵形状")
        total = None
        for term in self.terms:
            m = matrices[term.weight_id]
            value = np.conj(m) if term.conjugate else m
            part = term.coef * value
            total = part if total is None else total + part
        if self.real_only:
            total = total.real + 0.0j
        return total

    def accumulate(
        self,
        grad_complex: np.ndarray,
        grad_real: dict[str, np.ndarray],
        grad_imag: dict[str, np.ndarray],
    ) -> None:
        """把频点复矩阵梯度按配方分摊回独立物理权重的实/虚变量。

        与 :meth:`assemble` 互逆，保证正反向使用同一条独立参数映射。
        """
        g = grad_complex.real.copy()
        if not self.real_only:
            gi = grad_complex.imag.copy()
        for term in self.terms:
            wid = term.weight_id
            a = term.coef
            if term.conjugate:
                # dL/dRe(conj(M)) =  Re(dL/dC)*a；虚部反号
                grad_real[wid] += a * g
                if not self.real_only:
                    grad_imag[wid] -= a * gi
            else:
                grad_real[wid] += a * g
                if not self.real_only:
                    grad_imag[wid] += a * gi


# --------------------------------------------------------------------------
# 光学配置与计划
# --------------------------------------------------------------------------
@dataclass
class PlanBin:
    ky: int
    kx: int
    recipe: Recipe
    is_self: bool
    is_special: bool
    # 投影/取实部之前的原始标定配方，供 strict 数值复核；orthogonal 下
    # recipe 与 raw_recipe 可能不同，strict 必须基于 raw_recipe 判定。
    raw_recipe: "Recipe | None" = None


@dataclass
class Group:
    """配方相同的一组频点，向量化执行以控制临时张量规模。"""

    bins: list[PlanBin]
    signature: tuple
    recipe: Recipe

    @property
    def indices(self) -> np.ndarray:
        return np.array([(b.ky, b.kx) for b in self.bins], dtype=np.int64)


@dataclass
class Plan:
    config_id: str
    mode: str
    height: int
    width: int
    n_in: int
    n_out: int
    bins: list[PlanBin]
    groups: list[Group]
    missing_bins: list[tuple[int, int]]
    zero_bins: list[tuple[int, int]]
    projected_bins: list[dict]
    violations: list[dict]
    version: str
    scale: np.ndarray                 # irfft 伴随的逐频点缩放 1/N 或 2/N
    self_mask: np.ndarray              # 自共轭频点布尔图

    @property
    def half_width(self) -> int:
        return self.width // 2 + 1

    @property
    def key(self) -> tuple:
        return (self.config_id, self.height, self.width, self.mode, self.version)


class OpticalConfig:
    """一份相机光学配置（通道顺序 + 格表 + 缺测 + 插值 stencil）。"""

    def __init__(
        self,
        config_id: str,
        shape: tuple[int, int],
        grid: list[list],
        missing: Iterable[Iterable[int]],
        stencils: dict[str, list],
        input_order: tuple[str, ...],
        output_order: tuple[str, ...],
        weight_ids: Iterable[str],
    ):
        self.config_id = str(config_id)
        self.height, self.width = (int(shape[0]), int(shape[1]))
        self.grid = [tuple(row) for row in grid]
        self.missing = {tuple(int(v) for v in pair) for pair in missing}
        self.stencils = {
            name: [(str(wid), float(c)) for wid, c in spec]
            for name, spec in stencils.items()
        }
        self.input_order = tuple(input_order)
        self.output_order = tuple(output_order)
        self.weight_ids = tuple(weight_ids)
        self._validate()
        # 格表不可变，查找表与配方缓存在构造期建立，避免每次取配方重建
        self._grid_lookup = {(r[0], r[1]): r for r in self.grid}
        self._recipe_cache: dict[tuple[int, int], Recipe] = {}
        self.version = self._signature()

    # ---- 校验 ----------------------------------------------------------
    def _validate(self) -> None:
        h, w = self.height, self.width
        kh = w // 2 + 1
        seen: set[tuple[int, int]] = set()
        for row in self.grid:
            if len(row) != 4:
                raise ContractError(f"grid 条目必须是 [ky,kx,id,dir]：{row}")
            ky, kx, wid, direction = row
            if not (0 <= ky < h and 0 <= kx < kh):
                raise ContractError(f"频点 ({ky},{kx}) 超出半谱范围")
            if (ky, kx) in seen:
                raise ContractError(f"频点 ({ky},{kx}) 在格表中重复")
            seen.add((ky, kx))
            if int(direction) not in (1, -1):
                raise ContractError(f"方向必须是 ±1：{row}")
            self._resolve_id(str(wid))  # 检查引用可解析
        for ky, kx in self.missing:
            if not (0 <= ky < h and 0 <= kx < kh):
                raise ContractError(f"missing 频点 ({ky},{kx}) 超出半谱范围")

    def _resolve_id(self, wid: str) -> list[tuple[str, float, bool]]:
        """把权重 ID 或 stencil 名解析成 (wid, coef, conjugate) 列表。"""
        if wid in self.weight_ids:
            return [(wid, 1.0, False)]
        if wid in self.stencils:
            terms = []
            for base, coef in self.stencils[wid]:
                if base not in self.weight_ids:
                    raise ContractError(
                        f"stencil {wid} 引用未知权重 {base}"
                    )
                terms.append((base, coef, False))
            return terms
        raise ContractError(f"格表引用了未知的权重/stencil：{wid}")

    def _signature(self) -> str:
        payload = json.dumps(
            {
                "config": self.config_id,
                "shape": [self.height, self.width],
                "grid": self.grid,
                "missing": sorted(self.missing),
                "stencils": self.stencils,
                "input_order": self.input_order,
                "output_order": self.output_order,
                "weight_ids": self.weight_ids,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    # ---- 半谱几何 ------------------------------------------------------
    @staticmethod
    def conjugate_row(ky: int, height: int) -> int:
        """全谱中 -ky 对应的行索引（模 H）。"""
        return (-int(ky)) % height

    def special_columns(self) -> tuple[int, ...]:
        cols = [0]
        if self.width % 2 == 0:
            cols.append(self.width // 2)
        return tuple(cols)

    def is_self_conjugate(self, ky: int, kx: int) -> bool:
        return kx in self.special_columns() and \
            self.conjugate_row(ky, self.height) == ky

    def half_recipe(self, ky: int, kx: int) -> Recipe:
        """半谱格表条目 -> 配方（应用 stencil 与格表方向共轭）。

        格表 ``dir`` 是该条目自带的方向，只在这里应用一次；外部不要
        再把它当参数传入，否则 ``dir=-1`` 会被平方成 +1 而丢失共轭。
        结果按 ``(ky,kx)`` 记忆化，格表是不可变资料，可安全缓存。
        """
        key = (ky, kx)
        cached = self._recipe_cache.get(key)
        if cached is not None:
            return cached
        row = self._grid_lookup.get(key)
        terms: list[Term] = []
        if row is not None:
            wid = str(row[2])
            sign = int(row[3])
            if sign not in (1, -1):
                raise ContractError(f"方向必须是 ±1：{row}")
            for base, coef, _conj in self._resolve_id(wid):
                # 方向 -1：组合值逐元素共轭，即每个基矩阵取共轭
                terms.append(Term(base, coef, conjugate=(sign == -1)))
        recipe = Recipe(terms=terms)
        self._recipe_cache[key] = recipe
        return recipe

    # ---- 计划构建 ------------------------------------------------------
    def build_plan(self, mode: str, strict_tol: float = 1e-10) -> Plan:
        if mode not in MODES:
            raise ContractError(f"模式必须是 {MODES}，得到 {mode!r}")
        h, w = self.height, self.width
        kh = w // 2 + 1
        special = set(self.special_columns())

        bins: list[PlanBin] = []
        missing_bins: list[tuple[int, int]] = []
        zero_bins: list[tuple[int, int]] = []
        projected: list[dict] = []
        violations: list[dict] = []

        grid_lookup = self._grid_lookup

        for ky in range(h):
            for kx in range(kh):
                is_special = kx in special
                is_self = is_special and \
                    self.conjugate_row(ky, h) == ky
                listed = (ky, kx) in grid_lookup

                # 缺测频带：不参与正反向，整对作废
                if (ky, kx) in self.missing:
                    missing_bins.append((ky, kx))
                    continue

                if not is_special:
                    # 内部频点：无共轭约束，任意复矩阵；未列出即标定零
                    if not listed:
                        zero_bins.append((ky, kx))
                        continue
                    recipe = self.half_recipe(ky, kx)
                    bins.append(PlanBin(ky, kx, recipe, False, False))
                    continue

                if is_self:
                    recipe, kind, info = self._self_bin(
                        ky, kx, listed, grid_lookup, mode, strict_tol
                    )
                    if kind == MISSING_BIN:
                        missing_bins.append((ky, kx))
                        continue
                    if kind == ILLEGAL:
                        violations.append(info)
                        if mode == STRICT:
                            raise IllegalSpectrum(
                                f"自共轭频点 ({ky},{kx}) 含非零虚部，"
                                f"strict 模式拒绝：{info}"
                            )
                    if kind == PROJECT_REAL:
                        projected.append(info)
                    if kind == ZERO_CALIBRATION:
                        zero_bins.append((ky, kx))
                        continue
                    bins.append(PlanBin(ky, kx, recipe, True, True,
                                        raw_recipe=info.get("raw_recipe")))
                    continue

                # 特殊列的非自反行：按 (ky, -ky) 共轭配对，每对只处理一次
                partner = self.conjugate_row(ky, h)
                if partner < ky:
                    continue
                # 契约：缺测格成对表示时按两点均缺测处理
                if (ky, kx) in self.missing or (partner, kx) in self.missing:
                    missing_bins.append((ky, kx))
                    missing_bins.append((partner, kx))
                    continue
                recipe_p, kind, info = self._pair_bin(
                    ky, partner, kx, grid_lookup, mode, strict_tol
                )
                if kind == MISSING_BIN:
                    missing_bins.extend(tuple(b) for b in info["bins"])
                    continue
                if kind == ILLEGAL:
                    violations.append(info)
                    if mode == STRICT:
                        raise IllegalSpectrum(
                            f"特殊列行对 ({ky},{partner})@{kx} 不满足共轭"
                            f"约束，strict 模式拒绝：{info}"
                        )
                if kind == PROJECT_PAIR:
                    projected.append(info)
                if kind == ZERO_CALIBRATION:
                    zero_bins.extend(tuple(b) for b in info["bins"])
                    continue
                # 行对在半谱中占两个位置：本行用投影/锚点配方，
                # 配对行必须用其共轭配方，二者签名不同、各自成组，
                # 这样向量化执行不会互相覆盖。
                raw_a = info.get("raw_recipe_a")
                raw_b = info.get("raw_recipe_b")
                bins.append(PlanBin(ky, kx, recipe_p, False, True,
                                    raw_recipe=raw_a))
                bins.append(
                    PlanBin(partner, kx, _conjugate_recipe(recipe_p),
                            False, True, raw_recipe=raw_b)
                )

        # 逐频点的权重梯度缩放（全谱口径）：
        #   内部频点在全谱中有独立的负频率伙伴 -> 半谱系数 2/N；
        #   特殊列的自共轭点全谱只有自身 -> 1/N；
        #   特殊列的配对行两个半谱 bin 对应全谱两个独立频点，各 1/N，
        #   配对的另一 bin 由配方共轭处理，不能再用 2/N 否则重复计数。
        scale = np.full((h, kh), 2.0 / (h * w), dtype=np.float64)
        self_mask = np.zeros((h, kh), dtype=bool)
        for ky in range(h):
            for kx in special:
                scale[ky, kx] = 1.0 / (h * w)
                if self.conjugate_row(ky, h) == ky:
                    self_mask[ky, kx] = True

        groups = self._group_bins(bins)
        return Plan(
            config_id=self.config_id,
            mode=mode,
            height=h,
            width=w,
            n_in=len(self.input_order),
            n_out=len(self.output_order),
            bins=bins,
            groups=groups,
            missing_bins=sorted(missing_bins),
            zero_bins=sorted(zero_bins),
            projected_bins=projected,
            violations=violations,
            version=self.version,
            scale=scale,
            self_mask=self_mask,
        )

    def _self_bin(self, ky, kx, listed, grid_lookup, mode, tol):
        if not listed:
            return Recipe(), ZERO_CALIBRATION, {}
        row = grid_lookup[(ky, kx)]
        raw_recipe = self.half_recipe(ky, kx)       # 未取实部的原始标定
        recipe = Recipe(terms=list(raw_recipe.terms), real_only=True)
        info = {"bin": [ky, kx], "kind": "self", "grid": list(row),
                "raw_recipe": raw_recipe, "mode": mode}
        if mode == ORTHOGONAL:
            info["action"] = "取实部（投影到实子空间）"
            return recipe, PROJECT_REAL, info
        info["action"] = "要求标定虚部为零，数值在正向用 raw_recipe 复核"
        return recipe, OK, info

    def _pair_bin(self, ky, partner, kx, grid_lookup, mode, tol):
        bins = [(ky, kx), (partner, kx)]
        a_listed = (ky, kx) in grid_lookup
        b_listed = (partner, kx) in grid_lookup
        info = {"bins": [list(b) for b in bins], "kind": "pair"}

        if not a_listed and not b_listed:
            info["action"] = "两点均未列出，标定零"
            return Recipe(), ZERO_CALIBRATION, info

        # 只有一行有标定时：另一行按共轭约束补出（合法投影/严格接受）
        # 注意：half_recipe 内部只应用一次格表自带的 dir，不能再乘方向。
        row_a = grid_lookup.get((ky, kx))
        row_b = grid_lookup.get((partner, kx))
        if row_a is not None and row_b is None:
            raw_a = self.half_recipe(ky, kx)
            info.update(action="配对行缺失，按共轭补出", anchor=list(row_a),
                        raw_recipe_a=raw_a,
                        raw_recipe_b=_conjugate_recipe(raw_a))
            if mode == ORTHOGONAL:
                return raw_a, PROJECT_PAIR, info
            return raw_a, OK, info
        if row_b is not None and row_a is None:
            # 以 partner 为锚，本行取共轭
            raw_b = self.half_recipe(partner, kx)
            raw_a = _conjugate_recipe(raw_b)
            info.update(action="配对行缺失，按共轭补出", anchor=list(row_b),
                        raw_recipe_a=raw_a, raw_recipe_b=raw_b)
            if mode == ORTHOGONAL:
                return raw_a, PROJECT_PAIR, info
            return raw_a, OK, info

        # 两行都有标定：要求 M_partner == conj(M_ky)
        raw_a = self.half_recipe(ky, kx)
        raw_b = self.half_recipe(partner, kx)
        info.update(grid_a=list(row_a), grid_b=list(row_b),
                    raw_recipe_a=raw_a, raw_recipe_b=raw_b)
        if _recipes_are_conjugate(raw_a, raw_b):
            info["action"] = "标定满足共轭配对"
            return raw_a, OK, info
        # 结构上不共轭：orthogonal 取平均投影 M <- (A + conj(B))/2
        if mode == STRICT:
            info["action"] = "共轭约束不成立且禁止投影"
            return Recipe(), ILLEGAL, info
        projected_recipe = _average_conjugate(raw_a, raw_b)
        info["action"] = "投影为 (M_ky + conj(M_partner))/2"
        return projected_recipe, PROJECT_PAIR, info

    def _group_bins(self, bins: list[PlanBin]) -> list[Group]:
        """把配方相同且坐标互不重叠的频点并入同一向量化组。

        安全前提：一个组内的 ``(ky, kx)`` 坐标必须唯一，否则花式索引
        写回会在同一位置产生覆盖（特殊列共轭配对的两个 bin 坐标不同，
        但配方可能相同，需要先按坐标去重再分组）。
        """
        table: dict[tuple, dict[tuple, PlanBin]] = {}
        recipes: dict[tuple, Recipe] = {}
        for b in bins:
            sig = b.recipe.signature()
            bucket = table.setdefault(sig, {})
            coord = (b.ky, b.kx)
            if coord in bucket:
                # 同坐标同配方：语义相同，保留其一，不重复执行
                continue
            bucket[coord] = b
            recipes[sig] = b.recipe
        return [
            Group(bins=list(members.values()), signature=sig,
                  recipe=recipes[sig])
            for sig, members in table.items()
        ]


def _conjugate_recipe(recipe: Recipe) -> Recipe:
    """全谱共轭位置的配方：每个基矩阵换成它的共轭。"""
    terms = [
        Term(t.weight_id, t.coef, not t.conjugate) for t in recipe.terms
    ]
    return Recipe(terms=terms, real_only=recipe.real_only, kind=recipe.kind)


def _recipes_are_conjugate(a: Recipe, b: Recipe) -> bool:
    return _conjugate_recipe(a).signature() == b.signature()


def _average_conjugate(a: Recipe, b: Recipe) -> Recipe:
    """正交投影到 ``M2 = conj(M1)`` 子空间：``M1 <- (A + conj(B))/2``。"""
    terms: list[Term] = []
    for t in a.terms:
        terms.append(Term(t.weight_id, t.coef * 0.5, t.conjugate))
    for t in _conjugate_recipe(b).terms:
        terms.append(Term(t.weight_id, t.coef * 0.5, t.conjugate))
    return Recipe(terms=terms)


# --------------------------------------------------------------------------
# 配置注册与缓存
# --------------------------------------------------------------------------
class ConfigRegistry:
    """持有光学配置与按 (配置, 形状, 模式) 的计划缓存。

    换光学配置后，同数组索引对应不同物理频率：缓存键含配置身份与版本，
    旧计划不会被同形的新配置挤掉。:meth:`revoke` 模拟现场卸载配置——
    此前已完成前向的样本在反向时会因找不到原版映射而明确作废。
    """

    def __init__(self) -> None:
        self._configs: dict[str, OpticalConfig] = {}
        self._plans: dict[tuple, Plan] = {}
        self._revoked: dict[str, str] = {}

    @classmethod
    def from_session(cls, session: dict) -> "ConfigRegistry":
        registry = cls()
        input_order = tuple(session["input_order"])
        output_order = tuple(session["output_order"])
        weight_ids = tuple(session["weights"].keys())
        stencils = session.get("stencils", {})
        for record in session["maps"]:
            cfg = OpticalConfig(
                config_id=record["config"],
                shape=tuple(record["shape"]),
                grid=record["grid"],
                missing=record.get("missing", []),
                stencils=stencils,
                input_order=input_order,
                output_order=output_order,
                weight_ids=weight_ids,
            )
            registry.register(cfg)
        return registry

    def register(self, config: OpticalConfig) -> None:
        key = (config.config_id, config.height, config.width)
        old = self._configs.get(key)
        if old is not None and old.version != config.version:
            # 同身份同形状但内容变了：旧版本显式作废，避免同形缓存串台
            self._revoked[f"{key}:{old.version}"] = config.version
        self._configs[key] = config

    def get_config(self, config_id: str, shape: tuple[int, int]) -> OpticalConfig:
        key = (str(config_id), int(shape[0]), int(shape[1]))
        if key not in self._configs:
            revoked = [k for k in self._revoked if k.startswith(f"{key}:")]
            if revoked:
                raise StaleConfig(
                    f"配置 {config_id} 形状 {shape} 的原映射已被替换/撤销，"
                    f"缺少原版信息，本次梯度作废"
                )
            raise KeyError(f"未注册的光学配置：{key}")
        return self._configs[key]

    def plan(
        self,
        config_id: str,
        shape: tuple[int, int],
        mode: str,
        strict_tol: float = 1e-10,
    ) -> Plan:
        config = self.get_config(config_id, shape)
        cache_key = (
            config.config_id, config.height, config.width, mode, config.version
        )
        if cache_key not in self._plans:
            self._plans[cache_key] = config.build_plan(mode, strict_tol)
        return self._plans[cache_key]

    def revoke(self, config_id: str, shape: tuple[int, int]) -> None:
        key = (str(config_id), int(shape[0]), int(shape[1]))
        if key in self._configs:
            old = self._configs.pop(key)
            self._revoked[f"{key}:{old.version}"] = "revoked"
            dead = [k for k in self._plans if k[:4] == (*key, old.version)
                    or (k[0], k[1], k[2]) == key and k[4] == old.version]
            for k in dead:
                self._plans.pop(k, None)

    @property
    def active_keys(self) -> list[tuple]:
        return sorted(self._configs.keys())

    def load_session_file(self, path: str | Path) -> dict:
        data = Path(path).read_bytes()
        session = json.loads(data)
        registry = ConfigRegistry.from_session(session)
        for key, cfg in registry._configs.items():
            self.register(cfg)
        return session
