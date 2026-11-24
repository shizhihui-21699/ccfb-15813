"""评估：对照 ground truth 计算召回/精确率，并单列误报类型。

回答监管方最关心的问题：
1) 注入的作弊抓到了多少（召回）；
2) 抓出来的有多少是真作弊（精确率）；
3) 节假日高峰、首映零点场、合法包场/团购被误报了多少（误报率）。

匹配粒度：
- 场次级作弊（幽灵/退票/低价/偷换）按 showtime_id 匹配；
- 偷换票房若只在场次上被识别（D01），也按场次匹配。
"""

from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

from .config import DEFAULT_CONFIG
from .evidence import FindingType, Level
from .pipeline import PipelineResult, run_pipeline

FAKE_KINDS = {"fake_ghost", "fake_refund", "fake_lowprice"}
DIV_KINDS = {"div_mismatch", "div_online", "div_price"}
LEGIT_KINDS = {"legit_group"}


@dataclass
class EvalReport:
    truth_total: int = 0
    truth_fake: int = 0
    truth_div: int = 0
    truth_legit: int = 0
    caught_fake: int = 0
    caught_div: int = 0
    caught_legit: int = 0        # 合法样本被误报
    findings_total: int = 0
    findings_high: int = 0
    by_kind: dict[str, dict[str, int]] = field(default_factory=dict)
    false_positive_findings: int = 0
    details: list[str] = field(default_factory=list)

    @property
    def recall_fake(self) -> float:
        return self.caught_fake / self.truth_fake if self.truth_fake else 0.0

    @property
    def recall_div(self) -> float:
        return self.caught_div / self.truth_div if self.truth_div else 0.0

    @property
    def legit_fpr(self) -> float:
        return self.caught_legit / self.truth_legit if self.truth_legit else 0.0

    @property
    def precision(self) -> float:
        denom = self.findings_total
        if denom == 0:
            return 0.0
        return (self.caught_fake + self.caught_div) / denom

    @property
    def passed(self) -> bool:
        # 判定标准：假票房召回>=0.5，偷换召回>=0.5，合法样本误报<=0.2
        return (
            self.recall_fake >= 0.5
            and self.recall_div >= 0.5
            and self.legit_fpr <= 0.20
        )

    def summary_text(self) -> str:
        lines = [
            "=" * 62,
            "自检评估结果",
            "=" * 62,
            f"Ground truth：假票房 {self.truth_fake} / 偷换 {self.truth_div} / "
            f"合法对照 {self.truth_legit}",
            "",
            f"假票房召回率      : {self.recall_fake:.1%} "
            f"({self.caught_fake}/{self.truth_fake})",
            f"偷换票房召回率    : {self.recall_div:.1%} "
            f"({self.caught_div}/{self.truth_div})",
            f"合法样本误报率    : {self.legit_fpr:.1%} "
            f"({self.caught_legit}/{self.truth_legit})",
            f"输出发现总数      : {self.findings_total}（高 {self.findings_high}）",
            f"清单精确率(近似)  : {self.precision:.1%}",
            "",
            "分类型命中：",
        ]
        for kind, st in sorted(self.by_kind.items()):
            lines.append(
                f"  {kind:14s} 命中 {st['caught']}/{st['total']}"
            )
        if self.details:
            lines.append("")
            lines.append("备注：")
            lines.extend(f"  - {d}" for d in self.details[:20])
        lines.append("=" * 62)
        lines.append(f"总体判定：{'通过' if self.passed else '未通过'}")
        return "\n".join(lines)


def load_truth(data_dir: str | Path) -> list[dict]:
    p = Path(data_dir) / "truth.csv"
    if not p.exists():
        return []
    with open(p, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def evaluate(
    data_dir: str | Path,
    cfg=DEFAULT_CONFIG,
) -> EvalReport:
    truth = load_truth(data_dir)
    result: PipelineResult = run_pipeline(
        data_dir, cfg=cfg, restrict_to_date=False
    )
    findings = result.bundle.findings

    rep = EvalReport()
    rep.truth_total = len(truth)
    rep.truth_fake = sum(1 for t in truth if t["kind"] in FAKE_KINDS)
    rep.truth_div = sum(1 for t in truth if t["kind"] in DIV_KINDS)
    rep.truth_legit = sum(1 for t in truth if t["kind"] in LEGIT_KINDS)
    rep.findings_total = len(findings)
    rep.findings_high = sum(1 for f in findings if f.level == Level.HIGH)

    for t in truth:
        rep.by_kind.setdefault(t["kind"], {"total": 0, "caught": 0})
        rep.by_kind[t["kind"]]["total"] += 1

    # 索引发现：场次级 + 影院-影片-日级
    st_findings: dict[str, list] = {}
    cfd_findings: dict[tuple[str, str, str], list] = {}
    for f in findings:
        if f.showtime_id:
            st_findings.setdefault(f.showtime_id, []).append(f)
        if f.scope == "cinema_film_day":
            cfd_findings.setdefault(
                (f.cinema_id, f.film_id or "", f.date.isoformat()), []
            ).append(f)

    matched_findings: set[int] = set()
    for t in truth:
        sid = t.get("showtime_id") or ""
        kind = t["kind"]
        caught = False
        if sid and sid in st_findings:
            caught = True
            for f in st_findings[sid]:
                matched_findings.add(id(f))
        # 影院-影片-日级匹配（D 类）
        if not caught:
            key = (t["cinema_id"], t["film_id"], t["date"])
            if key in cfd_findings:
                caught = True
                for f in cfd_findings[key]:
                    matched_findings.add(id(f))
        if caught:
            rep.by_kind[kind]["caught"] += 1
            if kind in FAKE_KINDS:
                rep.caught_fake += 1
            elif kind in DIV_KINDS:
                rep.caught_div += 1
            elif kind in LEGIT_KINDS:
                rep.caught_legit += 1
        elif kind in LEGIT_KINDS and sid in st_findings:
            rep.caught_legit += 1

    rep.false_positive_findings = sum(
        1 for f in findings if id(f) not in matched_findings
    )
    if rep.false_positive_findings:
        rep.details.append(
            f"未被 truth 匹配的发现 {rep.false_positive_findings} 条"
            f"（含真实数据中未标注的异常、以及误报）"
        )
    return rep
