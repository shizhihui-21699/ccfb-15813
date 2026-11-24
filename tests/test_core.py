"""单元测试：数据解析、基线、特征、规则、误报抑制。

运行：python3 -m unittest discover -s tests -v
（也可以在项目根目录 python3 -m pytest tests -q）
"""

from __future__ import annotations

import datetime as dt
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from boxoffice_audit.baseline import BaselineBook, median, robust_z
from boxoffice_audit.calendar_ctx import (
    Calendar, CalendarEntry, DayType, FilmStage, film_stage,
    is_premiere_midnight, build_context,
)
from boxoffice_audit.config import DEFAULT_CONFIG
from boxoffice_audit.evidence import Evidence, Finding, FindingType
from boxoffice_audit.features import compute_showtime_features
from boxoffice_audit.schema import Column, FieldError, parse_value, read_table
from boxoffice_audit.store import Order, Showtime


def _st(sid="S1", hour=1, price=45.0, seats=100):
    return Showtime(
        sid, "C1", "F1", "H1",
        dt.datetime(2026, 9, 15, hour, 0), seats, price,
    )


def _order(oid, seats=2, channel="online", price=45.0, purchase=None,
           device="D1", member=True, status="valid", refund_time=None,
           seat_from=1):
    return Order(
        oid, "S1", channel, seats, price, price * seats,
        purchase or dt.datetime(2026, 9, 15, 0, 0), device,
        "U1", member, status, refund_time, "1", seat_from, seat_from + seats - 1,
    )


class TestSchemaParsing(unittest.TestCase):
    def test_parse_types(self):
        self.assertEqual(parse_value("12", Column("x", kind="int")), 12)
        self.assertEqual(parse_value("45.5元", Column("x", kind="str")), "45.5元")
        self.assertAlmostEqual(
            parse_value("¥1,234.5", Column("x", kind="float")), 1234.5
        )
        self.assertEqual(
            parse_value("2026/09/15", Column("x", kind="date")), dt.date(2026, 9, 15)
        )
        self.assertTrue(parse_value("是", Column("x", kind="bool")))

    def test_parse_errors(self):
        with self.assertRaises(FieldError):
            parse_value("", Column("x", kind="int"))
        with self.assertRaises(FieldError):
            parse_value("abc", Column("x", kind="int"))

    def test_nullable(self):
        self.assertIsNone(parse_value("", Column("x", kind="int", nullable=True)))

    def test_missing_required_column_raises(self):
        import tempfile, csv
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "cinemas.csv"
            with open(p, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["影院编码"])           # 缺少 name/city
                w.writerow(["C1"])
            with self.assertRaises(ValueError):
                read_table(p, "cinemas")

    def test_bad_rows_are_rejected_not_fatal(self):
        import tempfile, csv
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "cinemas.csv"
            with open(p, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["影院编码", "影院名称", "城市"])
                w.writerow(["C1", "好影院", "北京"])
                w.writerow(["", "坏影院", "北京"])
            r = read_table(p, "cinemas")
            self.assertEqual(r.ok, 1)
            self.assertEqual(len(r.rejected), 1)

    def test_chinese_header_aliases(self):
        import tempfile, csv
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "films.csv"
            with open(p, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["影片编码", "影片名称", "上映日期"])
                w.writerow(["F1", "某片", "2026-09-01"])
            r = read_table(p, "films")
            self.assertEqual(r.rows[0]["film_id"], "F1")
            self.assertEqual(r.rows[0]["release_date"], dt.date(2026, 9, 1))


class TestBaseline(unittest.TestCase):
    def test_robust_z_resists_outlier(self):
        normal = [0.1, 0.12, 0.09, 0.11, 0.1, 0.13, 0.08, 0.1]
        z_normal = robust_z(0.11, normal)
        z_outlier = robust_z(0.9, normal)
        self.assertLess(abs(z_normal), 3)
        self.assertGreater(z_outlier, 5)

    def test_freeze_gives_near_zero_for_typical(self):
        book = BaselineBook(min_samples=5)
        for v in [0.1, 0.2, 0.3, 0.15, 0.25] * 4:
            book.add(("normal", "00", "standard"), v)
        book.freeze()
        self.assertLess(abs(book.score(("normal", "00", "standard"), 0.2)), 1.0)
        self.assertGreater(book.score(("normal", "00", "standard"), 0.95), 5)

    def test_shrinkage_for_small_samples(self):
        book = BaselineBook(min_samples=20, shrinkage_k=8)
        for v in [0.5, 0.9]:
            book.add(("rare", "02", "standard"), v)
        for v in [0.1] * 50:
            book.add(("common", "02", "standard"), v)
        book.freeze()
        # 小样本键的中心应向全局收缩，不会直接取 0.7
        c, _s, n = book.center(("rare", "02", "standard"))
        self.assertEqual(n, 2)
        self.assertLess(c, 0.5)


class TestCalendarContext(unittest.TestCase):
    def test_day_types(self):
        cal = Calendar([
            CalendarEntry(dt.date(2026, 10, 2), DayType.HOLIDAY, "国庆档"),
        ])
        self.assertEqual(cal.day_type(dt.date(2026, 10, 2)), DayType.HOLIDAY)
        self.assertEqual(cal.day_type(dt.date(2026, 9, 26)), DayType.WEEKEND)  # 周六
        self.assertEqual(cal.day_type(dt.date(2026, 9, 22)), DayType.NORMAL)   # 周二

    def test_film_stage(self):
        release = dt.date(2026, 9, 15)
        self.assertEqual(film_stage(release, release), FilmStage.PREMIERE)
        self.assertEqual(film_stage(release, release + dt.timedelta(days=3)),
                         FilmStage.OPENING)
        self.assertEqual(film_stage(release, release + dt.timedelta(days=20)),
                         FilmStage.RUN)
        self.assertEqual(film_stage(None, release), FilmStage.UNKNOWN)

    def test_premiere_midnight(self):
        release = dt.date(2026, 9, 15)
        self.assertTrue(is_premiere_midnight(
            release, dt.datetime(2026, 9, 15, 0, 30)))
        self.assertFalse(is_premiere_midnight(
            release, dt.datetime(2026, 9, 16, 0, 30)))   # 非首映日
        self.assertFalse(is_premiere_midnight(
            release, dt.datetime(2026, 9, 15, 20, 0)))   # 非凌晨

    def test_baseline_key_separates_holiday(self):
        cal = Calendar([CalendarEntry(dt.date(2026, 10, 2), DayType.HOLIDAY)])
        c1 = build_context(dt.datetime(2026, 10, 2, 14, 0), cal,
                           dt.date(2026, 9, 1), "standard")
        c2 = build_context(dt.datetime(2026, 9, 22, 14, 0), cal,
                           dt.date(2026, 9, 1), "standard")
        self.assertNotEqual(c1.baseline_key, c2.baseline_key)
        self.assertEqual(c1.baseline_key[0], "holiday")


class TestShowtimeFeatures(unittest.TestCase):
    def test_occupancy_and_gross(self):
        orders = [_order(f"O{i}", seats=10, price=40.0) for i in range(5)]
        f = compute_showtime_features(_st(seats=100), orders, DEFAULT_CONFIG)
        self.assertEqual(f.sold_tickets, 50)
        self.assertAlmostEqual(f.occupancy, 0.5)
        self.assertAlmostEqual(f.gross, 2000.0)
        self.assertAlmostEqual(f.avg_price, 40.0)

    def test_refund_excluded_from_net_but_counted_in_peak(self):
        orders = [
            _order("O1", seats=10, status="refund",
                   refund_time=dt.datetime(2026, 9, 15, 0, 30)),
            _order("O2", seats=5),
        ]
        f = compute_showtime_features(_st(seats=100), orders, DEFAULT_CONFIG)
        self.assertEqual(f.sold_tickets, 5)
        self.assertEqual(f.refunded_tickets, 10)
        self.assertEqual(f.peak_tickets, 15)
        self.assertAlmostEqual(f.refund_rate, 10 / 15)

    def test_group_excluded_from_device_concentration(self):
        # 3 笔团体票同设备 + 2 笔散客同设备：散客集中度应按散客算
        orders = [_order(f"G{i}", seats=20, channel="group") for i in range(3)]
        orders += [_order(f"R{i}", seats=2, channel="online", device="Dx")
                   for i in range(2)]
        orders += [_order(f"S{i}", seats=2, channel="online", device=f"Dy{i}")
                   for i in range(3)]
        f = compute_showtime_features(_st(seats=200), orders, DEFAULT_CONFIG)
        self.assertEqual(f.group_tickets, 60)
        self.assertEqual(f.retail_tickets, 10)   # 4 + 6
        self.assertLess(f.retail_top_device_share, 0.6)

    def test_burst_detection(self):
        start = dt.datetime(2026, 9, 15, 19, 0)
        orders = [_order(f"O{i}", seats=4,
                         purchase=start - dt.timedelta(minutes=3))
                  for i in range(5)]
        f = compute_showtime_features(_st(hour=19), orders, DEFAULT_CONFIG)
        self.assertGreater(f.burst_share, 0.9)


class TestFindingScoring(unittest.TestCase):
    def _ev(self, primary, weight):
        return Evidence("F01", "t", weight, 1.0, None, "msg", primary=primary)

    def test_support_only_scores_zero(self):
        f = Finding(FindingType.FAKE, "showtime", "S1", dt.date(2026, 9, 15), "C1")
        f.add(self._ev(primary=False, weight=0.2))
        f.add(self._ev(primary=False, weight=0.2))
        f.finalize(DEFAULT_CONFIG)
        self.assertEqual(f.score, 0.0)
        self.assertFalse(f.passed)

    def test_primary_evidence_scores(self):
        f = Finding(FindingType.FAKE, "showtime", "S1", dt.date(2026, 9, 15), "C1")
        f.add(self._ev(primary=True, weight=0.35))
        f.finalize(DEFAULT_CONFIG)
        self.assertAlmostEqual(f.score, 0.35)

    def test_mitigators_discount(self):
        f1 = Finding(FindingType.FAKE, "showtime", "S1", dt.date(2026, 9, 15), "C1")
        f1.add(self._ev(True, 0.6))
        f1.finalize(DEFAULT_CONFIG)
        f2 = Finding(FindingType.FAKE, "showtime", "S1", dt.date(2026, 9, 15), "C1")
        f2.add(self._ev(True, 0.6))
        f2.add_mitigator("M01")
        f2.finalize(DEFAULT_CONFIG)
        self.assertLess(f2.score, f1.score)
        self.assertAlmostEqual(f2.score, 0.6 * 0.8, places=4)

    def test_score_capped(self):
        f = Finding(FindingType.FAKE, "showtime", "S1", dt.date(2026, 9, 15), "C1")
        for _ in range(10):
            f.add(self._ev(True, 0.5))
        f.finalize(DEFAULT_CONFIG)
        self.assertLessEqual(f.score, 1.0)


if __name__ == "__main__":
    unittest.main()
