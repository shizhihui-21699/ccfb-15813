"""检测结果的数据结构：证据、发现、评分聚合。

可解释性设计：
- 每条规则命中都产出一条 Evidence，带"人话"描述 + 原始数字 + 阈值；
- 一个对象（场次/影院-影片-日/影院-日）聚合多条 Evidence 得到可疑度；
- 输出必须回答"为什么可疑"，所以证据文本是产物的一部分，不是日志。
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class Level(str, Enum):
    HIGH = "high"        # 高可疑，建议现场核查
    MEDIUM = "medium"    # 中可疑，建议调阅记录
    LOW = "low"          # 低可疑，纳入监测


class FindingType(str, Enum):
    FAKE = "fake"                # 虚假票房 / 刷票房
    DIVERSION = "diversion"      # 偷换票房 / 截留


@dataclass
class Evidence:
    rule: str                    # F01 / D03 ...
    label: str
    weight: float
    value: float                 # 实际观测值
    threshold: float | None      # 判定阈值（None 表示相对比较）
    message: str                 # 人话解释，含数字
    action: str = ""             # 核查建议
    extra: dict[str, Any] = field(default_factory=dict)
    primary: bool = True         # 是否主证据（支撑证据不能单独定罪）

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "label": self.label,
            "weight": round(self.weight, 4),
            "primary": self.primary,
            "value": _round(self.value),
            "threshold": _round(self.threshold) if self.threshold is not None else None,
            "message": self.message,
            "action": self.action,
            "extra": {k: _round(v) if isinstance(v, (int, float)) else v for k, v in self.extra.items()},
        }


def _round(v: float, nd: int = 4) -> float:
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    try:
        if math.isinf(v) or math.isnan(v):
            return v
        return round(float(v), nd)
    except (TypeError, ValueError):
        return v


@dataclass
class Finding:
    """一个可疑对象及其全部证据。"""

    finding_type: FindingType
    scope: str                   # showtime | cinema_film_day | cinema_day
    object_id: str               # 场次ID / 影院ID+影片ID+日期 / 影院ID+日期
    date: dt.date
    cinema_id: str
    film_id: str | None = None
    showtime_id: str | None = None
    score: float = 0.0
    level: Level = Level.LOW
    evidence: list[Evidence] = field(default_factory=list)
    mitigators: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def add(self, ev: Evidence) -> None:
        self.evidence.append(ev)

    def add_mitigator(self, code: str) -> None:
        if code not in self.mitigators:
            self.mitigators.append(code)

    @property
    def primary_evidence(self) -> list[Evidence]:
        return [e for e in self.evidence if e.primary]

    @property
    def support_evidence(self) -> list[Evidence]:
        return [e for e in self.evidence if not e.primary]

    def finalize(self, cfg) -> None:
        """评分：主证据门槛 + 加权求和 + 误报抑制打折。

        关键是"主证据门槛"：只有支撑类证据（满座、单设备占比高、连座）
        而没有一条主证据时，可疑度直接判定为 0 —— 这类组合在正常经营中
        很常见（包场、热门场次），不应进入可疑清单。

        评分用加权求和而非指数饱和，是为了可审计：
        核对人员能一眼算出"这条发现的分是怎么来的"。
        """
        if not self.primary_evidence:
            self.score = 0.0
            self.level = Level.LOW
            return
        raw = sum(e.weight for e in self.evidence)
        # 证据条数封顶，避免同一维度重复堆叠
        capped = min(raw, 1.6)
        # 误报抑制：每个降权因素打 8 折（最多 3 个）
        discount = 0.8 ** min(len(self.mitigators), 3)
        self.score = round(min(1.0, capped * discount), 4)
        if self.score >= cfg.score_threshold_high:
            self.level = Level.HIGH
        elif self.score >= cfg.score_threshold_medium:
            self.level = Level.MEDIUM
        elif self.score >= cfg.score_threshold_low:
            self.level = Level.LOW
        else:
            self.level = Level.LOW

    @property
    def passed(self) -> bool:
        return bool(self.primary_evidence) and self.score > 0

    def reason_text(self) -> str:
        """生成"为什么可疑"的一句话摘要（给清单主列用）。"""
        if not self.evidence:
            return ""
        top = sorted(self.evidence, key=lambda e: e.weight, reverse=True)
        return "；".join(e.message for e in top[:3])

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_type": self.finding_type.value,
            "scope": self.scope,
            "object_id": self.object_id,
            "date": self.date.isoformat(),
            "cinema_id": self.cinema_id,
            "film_id": self.film_id,
            "showtime_id": self.showtime_id,
            "score": self.score,
            "level": self.level.value,
            "mitigators": self.mitigators,
            "metrics": self.metrics,
            "evidence": [e.to_dict() for e in self.evidence],
            "reason": self.reason_text(),
        }


@dataclass
class DetectorResult:
    findings: list[Finding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def extend(self, other: "DetectorResult") -> None:
        self.findings.extend(other.findings)
        self.notes.extend(other.notes)
