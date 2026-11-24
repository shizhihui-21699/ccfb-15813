"""影院-日 风险聚合：把场次/影片级发现汇总为影院清单。

监管方每天上午看两张清单：
1) 可疑影院 TOP N（按风险分排序，附主要疑点）；
2) 可疑场次明细（可下钻）。

影院风险不是简单相加：单场次刷票房 1 场 vs 20 场，严重程度不同，
因此用"最高单条 + 数量饱和项 + 深夜集中度"组合，避免刷场次数量直接
线性放大导致高频正常影院误报。
"""

from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from dataclasses import dataclass, field

from .config import DetectorConfig
from .evidence import Finding, FindingType, Level


@dataclass
class CinemaRisk:
    cinema_id: str
    date: dt.date
    score: float = 0.0
    level: Level = Level.LOW
    fake_count: int = 0
    diversion_count: int = 0
    top_findings: list[Finding] = field(default_factory=list)
    highlight: list[str] = field(default_factory=list)
    late_night_fake: int = 0
    total_showtimes: int = 0


def aggregate_cinema_risk(
    findings: list[Finding], cfg: DetectorConfig
) -> list[CinemaRisk]:
    by: dict[tuple[str, dt.date], list[Finding]] = defaultdict(list)
    for f in findings:
        by[(f.cinema_id, f.date)].append(f)

    out: list[CinemaRisk] = []
    for (cid, d), fs in by.items():
        fs_sorted = sorted(fs, key=lambda x: x.score, reverse=True)
        top = fs_sorted[0].score
        n = len(fs)
        # 数量饱和项：n=1 ->0.0; n=5 ->~0.5; 越多越接近 1 但边际递减
        sat = 1.0 - math.exp(-n / 4.0)
        score = min(1.0, 0.65 * top + 0.35 * sat)

        fake = [f for f in fs if f.finding_type == FindingType.FAKE]
        div = [f for f in fs if f.finding_type == FindingType.DIVERSION]
        late_night_fake = sum(
            1 for f in fake if f.metrics.get("late_night") is True
        )

        risk = CinemaRisk(cid, d, score=round(score, 4), fake_count=len(fake),
                          diversion_count=len(div), top_findings=fs_sorted[:5],
                          late_night_fake=late_night_fake)
        if score >= cfg.score_threshold_high:
            risk.level = Level.HIGH
        elif score >= cfg.score_threshold_medium:
            risk.level = Level.MEDIUM
        else:
            risk.level = Level.LOW
        risk.highlight = _highlights(fake, div, late_night_fake)
        out.append(risk)

    out.sort(key=lambda r: r.score, reverse=True)
    return out


def _highlights(fake: list[Finding], div: list[Finding], late_night: int) -> list[str]:
    hl: list[str] = []
    if late_night:
        hl.append(f"{late_night} 个深夜/凌晨场次上座异常")
    high_fake = [f for f in fake if f.level == Level.HIGH]
    if high_fake:
        conv = sum(1 for f in high_fake if any(e.rule in ("F03", "F04") for e in f.evidence))
        if conv:
            hl.append(f"{conv} 个高上座场次呈现设备/时间集中购票")
    refund = [f for f in high_fake if any(e.rule == "F07" for e in f.evidence)]
    if refund:
        hl.append(f"{len(refund)} 个场次开场前集中购票后退票")
    mismatch = [f for f in div if any(e.rule == "D01" for e in f.evidence)]
    if mismatch:
        hl.append(f"{len(mismatch)} 个场次上报影片与排片不符")
    if not hl and (fake or div):
        hl.append("存在需核查的异常，详见明细")
    return hl
