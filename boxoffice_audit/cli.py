"""命令行入口。

用法：
  python -m boxoffice_audit.cli run --data ./data [--date 2026-10-05] [--out ./out]
  python -m boxoffice_audit.cli synth --out ./data --days 60 --cinemas 40
  python -m boxoffice_audit.cli selftest
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from .config import DEFAULT_CONFIG
from .pipeline import run_pipeline, write_reports
from .report import ReportBundle


def _parse_date(s: str | None) -> dt.date | None:
    if not s:
        return None
    return dt.date.fromisoformat(s)


def cmd_run(args: argparse.Namespace) -> int:
    result = run_pipeline(
        args.data,
        run_date=_parse_date(args.date),
        cfg=DEFAULT_CONFIG,
        restrict_to_date=not args.all_dates,
    )
    b = result.bundle
    paths = write_reports(result, args.out)
    c = b.counts()
    print(f"检测完成（{b.run_date}）：命中 {c['total']} 条，涉及 {c['cinemas']} 家影院")
    print(f"  虚假票房 {c['fake']} / 偷换票房 {c['diversion']}；"
          f"高 {c['high']} 中 {c['medium']} 低 {c['low']}")
    print(f"  耗时 {result.stats.total_seconds:.2f}s，"
          f"场次 {result.stats.showtimes}，订单 {result.stats.orders}")
    for k, p in paths.items():
        print(f"  -> {k}: {p}")
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    from .synth.generate import generate_dataset
    generate_dataset(
        out_dir=args.out,
        days=args.days,
        cinemas=args.cinemas,
        seed=args.seed,
        inject=not args.clean,
    )
    print(f"合成数据已生成: {args.out}（{args.days} 天 × {args.cinemas} 影院）")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    from .synth.generate import build_selftest_dataset
    from .evaluation import evaluate

    tmp = Path(args.out or "./_selftest")
    report = evaluate(build_selftest_dataset(tmp))
    print(report.summary_text())
    return 0 if report.passed else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="boxoffice_audit", description="票房异常监管识别")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="运行单日检测并输出报告")
    r.add_argument("--data", required=True)
    r.add_argument("--out", default="./out")
    r.add_argument("--date", default=None, help="YYYY-MM-DD，默认取数据最后一天")
    r.add_argument("--all-dates", action="store_true", help="输出所有日期的发现")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("synth", help="生成含已知注入的合成数据")
    s.add_argument("--out", default="./data")
    s.add_argument("--days", type=int, default=60)
    s.add_argument("--cinemas", type=int, default=40)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--clean", action="store_true", help="不注入异常（对照组）")
    s.set_defaults(func=cmd_synth)

    t = sub.add_parser("selftest", help="端到端自检（合成数据 + 召回/误报评估）")
    t.add_argument("--out", default=None)
    t.set_defaults(func=cmd_selftest)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
