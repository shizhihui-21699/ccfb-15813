"""基线层：分层稳健统计。

核心思想：不要拿"深夜场"跟"黄金场"比，也不要拿"春节"跟"平日"比。
每条基线都在同一 (日期类型, 时段桶, 影厅层级) 分层内计算，并优先使用
"同城同档 + 同片"的对照组，再用稳健 z 分数衡量偏离。

为什么用中位数/MAD 而不是均值/标准差：
- 刷票房、偷换票房本身会产生极端值，均值和方差会被污染；
- MAD（中位绝对偏差）对长尾稳健，极端场次不会抬高"正常"基线；
- 小样本时用收缩（shrinkage）向同级总体均值靠拢，避免小影院噪声误报。
"""

from __future__ import annotations

import datetime as dt
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence


def median(xs: Sequence[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def mad(xs: Sequence[float], center: float | None = None) -> float:
    if not xs:
        return 0.0
    c = median(xs) if center is None else center
    return statistics.median([abs(x - c) for x in xs])


def robust_z(value: float, xs: Sequence[float], mad_floor: float = 1e-6) -> float:
    """稳健 z 分数：0.6745*(x-median)/MAD。MAD 过小时回退到 IQR/或 0。"""
    if not xs:
        return 0.0
    c = median(xs)
    m = mad(xs, c)
    if m < mad_floor:
        # 用极差/标准差兜底，避免常数序列导致 z 爆炸
        spread = statistics.pstdev([x - c for x in xs]) if len(xs) > 1 else 0.0
        if spread < mad_floor:
            return 0.0 if abs(value - c) < mad_floor else float("inf") if value > c else 0.0
        return (value - c) / spread
    return 0.6745 * (value - c) / m


@dataclass
class Baseline:
    """单条基线：分层键 + 样本分布。"""

    key: tuple
    values: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.values)

    def stats(self) -> dict[str, float]:
        if not self.values:
            return {"n": 0, "median": 0.0, "mad": 0.0, "mean": 0.0}
        c = median(self.values)
        return {
            "n": len(self.values),
            "median": c,
            "mad": mad(self.values, c),
            "mean": statistics.fmean(self.values),
            "q25": _percentile(self.values, 0.25),
            "q75": _percentile(self.values, 0.75),
        }


def _percentile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


class BaselineBook:
    """按任意键分层的基线集合，支持收缩。

    性能：分层统计（中位数/MAD/分位）只在 freeze() 时算一次，
    之后 score() 为 O(1) 字典查找。全国日批动辄数十万场次，
    若每次 score 都重排分组会退化为 O(n log n)/次，无法满足出报时效。
    """

    def __init__(self, min_samples: int = 20, shrinkage_k: float = 8.0):
        self._groups: dict[tuple, list[float]] = defaultdict(list)
        self.min_samples = min_samples
        self.shrinkage_k = shrinkage_k
        self._global: list[float] = []
        self._cache: dict[tuple, tuple[float, float, int]] = {}
        self._global_center = 0.0
        self._global_scale = 0.0
        self._frozen = False

    def add(self, key: tuple, value: float) -> None:
        self._groups[key].append(value)
        self._global.append(value)
        self._frozen = False

    def freeze(self) -> None:
        """预计算所有分层统计。建完基线后调用一次。"""
        if self._frozen:
            return
        self._global_center = median(self._global) if self._global else 0.0
        self._global_scale = mad(self._global, self._global_center) if self._global else 0.0
        self._cache = {}
        for key, vals in self._groups.items():
            self._cache[key] = self._compute_center(vals)
        self._frozen = True

    def _compute_center(self, vals: list[float]) -> tuple[float, float, int]:
        n = len(vals)
        g_center = self._global_center
        g_scale = self._global_scale
        if not vals:
            return g_center, g_scale, 0
        c = median(vals)
        s = mad(vals, c)
        w = n / (n + self.shrinkage_k)
        return (w * c + (1 - w) * g_center, w * s + (1 - w) * g_scale, n)

    def values(self, key: tuple) -> list[float]:
        return self._groups.get(key, [])

    def has(self, key: tuple) -> bool:
        return len(self._groups.get(key, [])) >= self.min_samples

    def center(self, key: tuple) -> tuple[float, float, int]:
        """返回 (收缩后中心, 收缩后尺度, 样本数)。"""
        if not self._frozen:
            self.freeze()
        if key in self._cache:
            return self._cache[key]
        return self._global_center, self._global_scale, 0

    def score(self, key: tuple, value: float, mad_floor: float = 1e-6) -> float:
        """收缩后的稳健 z。"""
        center, scale, n = self.center(key)
        if scale < mad_floor:
            if abs(value - center) < mad_floor:
                return 0.0
            return 20.0 if value > center else 0.0
        z = 0.6745 * (value - center) / scale
        return z


def build_baselines(
    samples: Iterable[tuple[tuple, float]],
    min_samples: int = 20,
) -> BaselineBook:
    book = BaselineBook(min_samples=min_samples)
    for key, val in samples:
        book.add(key, val)
    book.freeze()
    return book


@dataclass
class PeerStats:
    """同城同档（可选同片）对照组的汇总统计。"""

    city: str
    film_id: str | None
    n_cinemas: int
    median_occupancy: float
    median_avg_price: float | None
    median_share: float | None
    median_attendance: float | None


class PeerIndex:
    """同城同档影院对照索引，用于偷换票房（D 类）检测。

    偷换票房的本质是"横向不一致"：同城同档同片，别人场均 15 人，
    你场均 80 人；或热门片别人占比 30%，你只有 5%。因此 D 类检测以
    同城同档为对照，而不是全国。
    """

    def __init__(self) -> None:
        # (city, film_id) -> {cinema_id: 指标}
        self.film_cinema: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(dict)
        # city -> 影院级指标
        self.city_cinema: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)

    def add_film_cinema(self, city: str, film_id: str, cinema_id: str, metrics: dict[str, float]) -> None:
        self.film_cinema[(city, film_id)][cinema_id] = metrics

    def add_city_cinema(self, city: str, cinema_id: str, metrics: dict[str, float]) -> None:
        self.city_cinema[city][cinema_id] = metrics

    def film_peers(self, city: str, film_id: str, exclude: str | None = None) -> PeerStats:
        d = self.film_cinema.get((city, film_id), {})
        occ = [m["occupancy"] for cid, m in d.items() if cid != exclude]
        prices = [m["avg_price"] for cid, m in d.items() if cid != exclude and m.get("avg_price")]
        shares = [m["share"] for cid, m in d.items() if cid != exclude and "share" in m]
        att = [m["attendance"] for cid, m in d.items() if cid != exclude and "attendance" in m]
        return PeerStats(
            city=city, film_id=film_id, n_cinemas=len(occ),
            median_occupancy=median(occ),
            median_avg_price=median(prices) if prices else None,
            median_share=median(shares) if shares else None,
            median_attendance=median(att) if att else None,
        )
