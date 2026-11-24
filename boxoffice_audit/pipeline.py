"""检测流水线编排：装载 -> 特征 -> 基线 -> 检测 -> 聚合 -> 报告。

单日批处理入口。为了"第二天上午出结果"，这里的设计是：
- 装载与特征为 O(行数)，检测为 O(场次数)，聚合为 O(发现数)；
- 基线只用历史窗口，不在批处理中做全量两两比较；
- 所有阶段可单独调用，便于回溯与复算。
"""

from __future__ import annotations

import datetime as dt
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .aggregate import CinemaRisk, aggregate_cinema_risk
from .config import DEFAULT_CONFIG, DetectorConfig
from .detectors.diversion import DiversionDetector
from .detectors.showtime_fake import FakeBoxOfficeDetector
from .evidence import Finding
from .features import build_all_features
from .report import ReportBundle
from .store import Dataset


@dataclass
class PipelineStats:
    load_seconds: float = 0.0
    feature_seconds: float = 0.0
    fake_seconds: float = 0.0
    diversion_seconds: float = 0.0
    aggregate_seconds: float = 0.0
    total_seconds: float = 0.0
    showtimes: int = 0
    orders: int = 0

    def as_dict(self) -> dict:
        return {
            "load_seconds": round(self.load_seconds, 3),
            "feature_seconds": round(self.feature_seconds, 3),
            "fake_seconds": round(self.fake_seconds, 3),
            "diversion_seconds": round(self.diversion_seconds, 3),
            "aggregate_seconds": round(self.aggregate_seconds, 3),
            "total_seconds": round(self.total_seconds, 3),
            "showtimes": self.showtimes,
            "orders": self.orders,
        }


@dataclass
class PipelineResult:
    bundle: ReportBundle
    stats: PipelineStats


def run_pipeline(
    data_dir: str | Path,
    *,
    run_date: dt.date | None = None,
    cfg: DetectorConfig | None = None,
    restrict_to_date: bool = True,
) -> PipelineResult:
    """执行一次完整检测。

    restrict_to_date=True 时只输出 run_date 当天的发现（默认取数据中最后一天），
    但基线与特征仍使用全部历史数据（因为历史越充分基线越稳）。
    """
    cfg = cfg or DEFAULT_CONFIG
    t0 = time.perf_counter()
    stats = PipelineStats()

    ds = Dataset.load(data_dir)
    stats.load_seconds = time.perf_counter() - t0
    stats.showtimes = len(ds.showtimes)
    stats.orders = sum(len(v) for v in ds.orders_by_showtime.values())

    t = time.perf_counter()
    features = build_all_features(ds, cfg)
    stats.feature_seconds = time.perf_counter() - t

    t = time.perf_counter()
    fake = FakeBoxOfficeDetector(ds, features, cfg).run()
    stats.fake_seconds = time.perf_counter() - t

    t = time.perf_counter()
    diversion = DiversionDetector(ds, features, cfg).run()
    stats.diversion_seconds = time.perf_counter() - t

    findings: list[Finding] = [*fake.findings, *diversion.findings]

    dates = ds.all_dates()
    if run_date is None:
        run_date = dates[-1] if dates else dt.date.today()
    if restrict_to_date:
        findings = [f for f in findings if f.date == run_date]

    t = time.perf_counter()
    risks = aggregate_cinema_risk(findings, cfg)
    stats.aggregate_seconds = time.perf_counter() - t

    notes = [*fake.notes, *diversion.notes]
    bundle = ReportBundle(
        run_date=run_date,
        findings=findings,
        cinema_risks=risks,
        ds=ds,
        cfg=cfg,
        load_summary=ds.load_report.summary(),
        notes=notes,
    )
    stats.total_seconds = time.perf_counter() - t0
    return PipelineResult(bundle, stats)


def write_reports(
    result: PipelineResult, out_dir: str | Path
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle = result.bundle
    paths: dict[str, Path] = {}

    md = out / f"daily_report_{bundle.run_date:%Y%m%d}.md"
    md.write_text(bundle.to_markdown(), encoding="utf-8")
    paths["markdown"] = md

    for p in bundle.write_csv(out):
        paths[p.stem] = p

    js = out / f"findings_{bundle.run_date:%Y%m%d}.json"
    bundle.write_json(js)
    paths["json"] = js

    run = out / "run_stats.json"
    run.write_text(json.dumps(result.stats.as_dict(), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    paths["stats"] = run
    return paths
