"""报告输出：Markdown 日报、CSV 清单、JSON 结构化结果。

清单面向两类读者：
- 监管人员：Markdown/CSV，先看影院 TOP、再看场次明细，每条带人话理由；
- 下游系统：JSON，含规则码、证据、指标，便于入库和二次分析。
"""

from __future__ import annotations

import csv
import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .aggregate import CinemaRisk
from .config import DetectorConfig
from .evidence import Finding, FindingType, Level
from .store import Dataset

_LEVEL_CN = {Level.HIGH: "高", Level.MEDIUM: "中", Level.LOW: "低"}
_TYPE_CN = {FindingType.FAKE: "虚假票房", FindingType.DIVERSION: "偷换票房"}


@dataclass
class ReportBundle:
    run_date: dt.date
    findings: list[Finding]
    cinema_risks: list[CinemaRisk]
    ds: Dataset
    cfg: DetectorConfig
    load_summary: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def showtime_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.scope == "showtime"]

    @property
    def high(self) -> list[Finding]:
        return [f for f in self.findings if f.level == Level.HIGH]

    def counts(self) -> dict[str, int]:
        return {
            "total": len(self.findings),
            "fake": sum(1 for f in self.findings if f.finding_type == FindingType.FAKE),
            "diversion": sum(1 for f in self.findings if f.finding_type == FindingType.DIVERSION),
            "high": sum(1 for f in self.findings if f.level == Level.HIGH),
            "medium": sum(1 for f in self.findings if f.level == Level.MEDIUM),
            "low": sum(1 for f in self.findings if f.level == Level.LOW),
            "cinemas": len({f.cinema_id for f in self.findings}),
        }

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------
    def to_markdown(self, top_cinemas: int = 50, top_showtimes: int = 100) -> str:
        c = self.counts()
        L: list[str] = []
        L.append(f"# 票房异常监管日报（{self.run_date:%Y-%m-%d}）")
        L.append("")
        L.append(f"> 数据装载：{self.load_summary}")
        L.append("")
        L.append("## 一、总体概况")
        L.append("")
        L.append(f"- 命中异常 **{c['total']}** 条，涉及影院 **{c['cinemas']}** 家")
        L.append(f"- 虚假票房 {c['fake']} 条，偷换票房 {c['diversion']} 条")
        L.append(f"- 风险等级：高 {c['high']} / 中 {c['medium']} / 低 {c['low']}")
        if self.notes:
            L.append("")
            L.append("**处理说明：**")
            for n in self.notes:
                L.append(f"- {n}")
        L.append("")

        L.append(f"## 二、可疑影院清单（前 {top_cinemas}）")
        L.append("")
        L.append("| 排名 | 影院 | 城市 | 日期 | 风险分 | 等级 | 虚假 | 偷换 | 主要疑点 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(self.cinema_risks[:top_cinemas], start=1):
            cine = self.ds.cinemas.get(r.cinema_id)
            name = cine.name if cine else r.cinema_id
            city = cine.city if cine else ""
            hl = "；".join(r.highlight) if r.highlight else "—"
            L.append(
                f"| {i} | {name}<br>`{r.cinema_id}` | {city} | {r.date.isoformat()} | "
                f"{r.score:.2f} | {_LEVEL_CN[r.level]} | {r.fake_count} | {r.diversion_count} | {hl} |"
            )
        L.append("")

        L.append(f"## 三、可疑场次明细（前 {top_showtimes}）")
        L.append("")
        L.append("| 日期 | 影院 | 影片 | 场次时间 | 类型 | 分 | 等级 | 为什么可疑 |")
        L.append("|---|---|---|---|---|---|---|---|")
        shown = 0
        for f in sorted(self.showtime_findings, key=lambda x: x.score, reverse=True):
            if shown >= top_showtimes:
                break
            shown += 1
            cine = self.ds.cinemas.get(f.cinema_id)
            name = cine.name if cine else f.cinema_id
            film = self.ds.films.get(f.film_id) if f.film_id else None
            title = film.title if film else (f.film_id or "")
            t = f.metrics.get("start_time", "")
            try:
                tm = dt.datetime.fromisoformat(t).strftime("%m-%d %H:%M")
            except (ValueError, TypeError):
                tm = str(t)
            typ = _TYPE_CN[f.finding_type]
            if f.finding_type == FindingType.DIVERSION and f.scope != "showtime":
                typ = "偷换(影片级)"
            why = _why_short(f)
            L.append(
                f"| {f.date.isoformat()} | {name}<br>`{f.cinema_id}` | {title} | {tm} | "
                f"{typ} | {f.score:.2f} | {_LEVEL_CN[f.level]} | {why} |"
            )
        L.append("")

        # 高可疑影院逐条说明
        L.append("## 四、高可疑条目逐条说明")
        L.append("")
        for f in sorted(self.high, key=lambda x: x.score, reverse=True)[:30]:
            L.append(_finding_detail(f, self.ds))
            L.append("")

        L.append("---")
        L.append("")
        L.append("### 附：规则与误报抑制说明")
        L.append("")
        from .config import FAKE_RULES, DIVERSION_RULES, MITIGATORS
        L.append("| 规则 | 名称 | 权重 | 说明 |")
        L.append("|---|---|---|---|")
        for r in (*FAKE_RULES, *DIVERSION_RULES):
            L.append(f"| {r.code} | {r.label} | {r.weight:.2f} | {r.description} |")
        L.append("")
        L.append("**误报抑制项（命中后对可疑度打折）：**")
        L.append("")
        for r in MITIGATORS:
            L.append(f"- **{r.code} {r.label}**：{r.description}")
        L.append("")
        return "\n".join(L)

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------
    def write_csv(self, out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []

        p1 = out_dir / f"cinema_risk_{self.run_date:%Y%m%d}.csv"
        with open(p1, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["影院编码", "影院名称", "城市", "日期", "风险分", "等级",
                        "虚假票房条数", "偷换票房条数", "深夜异常场次", "主要疑点"])
            for r in self.cinema_risks:
                cine = self.ds.cinemas.get(r.cinema_id)
                w.writerow([
                    r.cinema_id, cine.name if cine else "", cine.city if cine else "",
                    r.date.isoformat(), f"{r.score:.4f}", _LEVEL_CN[r.level],
                    r.fake_count, r.diversion_count, r.late_night_fake,
                    "；".join(r.highlight),
                ])
        paths.append(p1)

        p2 = out_dir / f"findings_{self.run_date:%Y%m%d}.csv"
        with open(p2, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["日期", "影院编码", "影院名称", "影片编码", "影片名称", "场次编码",
                        "场次时间", "类型", "粒度", "可疑度", "等级", "命中规则",
                        "为什么可疑", "核查建议", "误报抑制"])
            for f in sorted(self.findings, key=lambda x: x.score, reverse=True):
                cine = self.ds.cinemas.get(f.cinema_id)
                film = self.ds.films.get(f.film_id) if f.film_id else None
                rules = ";".join(e.rule for e in f.evidence)
                actions = "；".join(dict.fromkeys(e.action for e in f.evidence if e.action))
                w.writerow([
                    f.date.isoformat(), f.cinema_id, cine.name if cine else "",
                    f.film_id or "", film.title if film else "",
                    f.showtime_id or "", f.metrics.get("start_time", ""),
                    _TYPE_CN[f.finding_type], f.scope, f"{f.score:.4f}",
                    _LEVEL_CN[f.level], rules, _why_short(f), actions,
                    ",".join(f.mitigators),
                ])
        paths.append(p2)
        return paths

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------
    def to_json_obj(self) -> dict:
        return {
            "run_date": self.run_date.isoformat(),
            "load_summary": self.load_summary,
            "counts": self.counts(),
            "cinema_risks": [
                {
                    "cinema_id": r.cinema_id,
                    "cinema_name": (self.ds.cinemas[r.cinema_id].name
                                    if r.cinema_id in self.ds.cinemas else r.cinema_id),
                    "city": (self.ds.cinemas[r.cinema_id].city
                             if r.cinema_id in self.ds.cinemas else ""),
                    "date": r.date.isoformat(),
                    "score": r.score,
                    "level": r.level.value,
                    "fake_count": r.fake_count,
                    "diversion_count": r.diversion_count,
                    "late_night_fake": r.late_night_fake,
                    "highlight": r.highlight,
                }
                for r in self.cinema_risks
            ],
            "findings": [f.to_dict() for f in self.findings],
            "notes": self.notes,
        }

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_json_obj(), fh, ensure_ascii=False, indent=2, default=str)
        return path


def _why_short(f: Finding, max_ev: int = 3) -> str:
    evs = sorted(f.evidence, key=lambda e: e.weight, reverse=True)[:max_ev]
    return "；".join(e.message for e in evs)


def _finding_detail(f: Finding, ds: Dataset) -> str:
    cine = ds.cinemas.get(f.cinema_id)
    name = cine.name if cine else f.cinema_id
    film = ds.films.get(f.film_id) if f.film_id else None
    title = film.title if film else ""
    L = [
        f"### [{_TYPE_CN[f.finding_type]}·{_LEVEL_CN[f.level]}] {name} · {title} "
        f"（{f.date.isoformat()}，可疑度 {f.score:.2f}）",
        "",
    ]
    if f.showtime_id:
        L.append(f"- 场次：`{f.showtime_id}`，时间 {f.metrics.get('start_time','')}")
    for e in sorted(f.evidence, key=lambda e: e.weight, reverse=True):
        L.append(f"- **{e.rule} {e.label}**：{e.message}")
        if e.action:
            L.append(f"  - 核查建议：{e.action}")
    if f.mitigators:
        L.append(f"- 已计入的降权因素：{', '.join(f.mitigators)}")
        L.append("  - 说明：这些因素可解释部分异常，但仍建议抽样核实")
    return "\n".join(L)
