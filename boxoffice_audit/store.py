"""数据仓库：装载各表、建立索引、按场次/日聚合。

设计目标：
- 全国日批可扩展：订单按场次聚合为 O(#场次) 特征，避免全量 Cartesian 比较；
- 容错：坏行进 rejected，不阻断批处理；
- 可解释：所有中间统计都能回溯到原始行。

记录类用 __slots__，数百万行时内存可控。
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from .calendar_ctx import Calendar, CalendarEntry, DayType
from .schema import RejectedRow, read_table


# 退款/取消状态词
_REFUND_STATUS = {
    "refund", "refunded", "cancel", "canceled", "cancelled", "退票", "已退",
    "已取消", "退款", "void",
}
_VALID_STATUS = {"valid", "ok", "paid", "used", "已售", "正常", "已使用", "已完成", ""}


def order_is_refunded(order: "Order") -> bool:
    if order.refund_time is not None:
        return True
    if order.status:
        return order.status.strip().lower() in _REFUND_STATUS
    return False


@dataclass(slots=True)
class Cinema:
    cinema_id: str
    name: str
    province: str | None
    city: str
    tier: int | None
    screens: int | None


@dataclass(slots=True)
class Film:
    film_id: str
    title: str
    release_date: dt.date | None
    is_import: bool | None


@dataclass(slots=True)
class Hall:
    hall_id: str
    cinema_id: str
    hall_format: str | None
    seats: int | None


@dataclass(slots=True)
class Showtime:
    showtime_id: str
    cinema_id: str
    film_id: str
    hall_id: str | None
    start_time: dt.datetime
    planned_seats: int | None
    settlement_price: float | None


@dataclass(slots=True)
class Order:
    order_id: str
    showtime_id: str
    channel: str
    seats: int
    unit_price: float
    amount: float
    purchase_time: dt.datetime
    device_id: str | None
    account_id: str | None
    is_member: bool | None
    status: str | None
    refund_time: dt.datetime | None
    seat_row: str | None
    seat_from: int | None
    seat_to: int | None


@dataclass(slots=True)
class Settlement:
    settle_id: str | None
    cinema_id: str
    showtime_id: str | None
    filed_film_id: str | None
    reported_tickets: int | None
    reported_gross: float | None
    reported_price: float | None
    settle_date: dt.date | None


@dataclass(slots=True)
class OnlineSale:
    cinema_id: str
    film_id: str
    sale_date: dt.date
    online_tickets: int
    online_gross: float | None


@dataclass
class LoadReport:
    tables: dict[str, int] = field(default_factory=dict)
    rejected: dict[str, list[RejectedRow]] = field(default_factory=dict)
    missing_files: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{k}: {v} 行" for k, v in sorted(self.tables.items())]
        rej = sum(len(v) for v in self.rejected.values())
        if rej:
            parts.append(f"拒收 {rej} 行")
        if self.missing_files:
            parts.append(f"缺失文件 {len(self.missing_files)} 个")
        return "; ".join(parts)


# 表名 -> 数据目录下的候选文件名
_TABLE_FILES = {
    "cinemas": ("cinemas.csv", "影院.csv"),
    "films": ("films.csv", "影片.csv"),
    "halls": ("halls.csv", "影厅.csv"),
    "showtimes": ("showtimes.csv", "场次.csv", "排片.csv"),
    "orders": ("orders.csv", "订单.csv", "售票记录.csv"),
    "settlements": ("settlements.csv", "结算.csv", "上报.csv"),
    "online_sales": ("online_sales.csv", "线上销售.csv"),
    "calendar": ("calendar.csv", "日历.csv"),
}


class Dataset:
    """装载后的全量数据与聚合索引。"""

    def __init__(self) -> None:
        self.cinemas: dict[str, Cinema] = {}
        self.films: dict[str, Film] = {}
        self.halls: dict[str, Hall] = {}
        self.showtimes: dict[str, Showtime] = {}
        self.orders_by_showtime: dict[str, list[Order]] = defaultdict(list)
        self.settlements_by_showtime: dict[str, list[Settlement]] = defaultdict(list)
        self.settlements_by_cinema: dict[str, list[Settlement]] = defaultdict(list)
        self.online_sales: list[OnlineSale] = []
        self.calendar = Calendar()
        self.load_report = LoadReport()
        # 城市/影院索引
        self.cinemas_by_city: dict[str, list[Cinema]] = defaultdict(list)
        self.showtimes_by_cinema_date: dict[tuple[str, dt.date], list[Showtime]] = defaultdict(list)

    # ------------------------------------------------------------------
    # 装载
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, data_dir: str | Path) -> "Dataset":
        ds = cls()
        d = Path(data_dir)
        if not d.exists():
            raise FileNotFoundError(f"数据目录不存在: {d}")

        def _load(table: str):
            for fname in _TABLE_FILES[table]:
                p = d / fname
                if p.exists():
                    return read_table(p, table)
            ds.load_report.missing_files.append(_TABLE_FILES[table][0])
            return None

        # 基础维表
        r = _load("cinemas")
        if r:
            ds.load_report.tables["cinemas"] = r.ok
            ds.load_report.rejected["cinemas"] = r.rejected
            for row in r.rows:
                c = Cinema(
                    cinema_id=str(row["cinema_id"]), name=str(row["name"]),
                    province=row.get("province"), city=str(row["city"]),
                    tier=row.get("tier"), screens=row.get("screens"),
                )
                ds.cinemas[c.cinema_id] = c
                ds.cinemas_by_city[c.city].append(c)

        r = _load("films")
        if r:
            ds.load_report.tables["films"] = r.ok
            ds.load_report.rejected["films"] = r.rejected
            for row in r.rows:
                f = Film(str(row["film_id"]), str(row["title"]),
                         row.get("release_date"), row.get("is_import"))
                ds.films[f.film_id] = f

        r = _load("halls")
        if r:
            ds.load_report.tables["halls"] = r.ok
            ds.load_report.rejected["halls"] = r.rejected
            for row in r.rows:
                h = Hall(str(row["hall_id"]), str(row["cinema_id"]),
                         row.get("hall_format"), row.get("seats"))
                ds.halls[h.hall_id] = h

        r = _load("showtimes")
        if r:
            ds.load_report.tables["showtimes"] = r.ok
            ds.load_report.rejected["showtimes"] = r.rejected
            for row in r.rows:
                s = Showtime(
                    str(row["showtime_id"]), str(row["cinema_id"]), str(row["film_id"]),
                    row.get("hall_id"), row["start_time"],
                    row.get("planned_seats"), row.get("settlement_price"),
                )
                ds.showtimes[s.showtime_id] = s
                ds.showtimes_by_cinema_date[(s.cinema_id, s.start_time.date())].append(s)

        r = _load("orders")
        if r:
            ds.load_report.tables["orders"] = r.ok
            ds.load_report.rejected["orders"] = r.rejected
            for row in r.rows:
                o = Order(
                    str(row["order_id"]), str(row["showtime_id"]), str(row["channel"]),
                    int(row["seats"]), float(row["unit_price"]), float(row["amount"]),
                    row["purchase_time"], row.get("device_id"), row.get("account_id"),
                    row.get("is_member"), row.get("status"), row.get("refund_time"),
                    row.get("seat_row"), row.get("seat_from"), row.get("seat_to"),
                )
                ds.orders_by_showtime[o.showtime_id].append(o)

        r = _load("settlements")
        if r:
            ds.load_report.tables["settlements"] = r.ok
            ds.load_report.rejected["settlements"] = r.rejected
            for row in r.rows:
                st = Settlement(
                    row.get("settle_id"), str(row["cinema_id"]), row.get("showtime_id"),
                    row.get("filed_film_id"), row.get("reported_tickets"),
                    row.get("reported_gross"), row.get("reported_price"), row.get("settle_date"),
                )
                if st.showtime_id:
                    ds.settlements_by_showtime[st.showtime_id].append(st)
                ds.settlements_by_cinema[st.cinema_id].append(st)

        r = _load("online_sales")
        if r:
            ds.load_report.tables["online_sales"] = r.ok
            ds.load_report.rejected["online_sales"] = r.rejected
            for row in r.rows:
                ds.online_sales.append(OnlineSale(
                    str(row["cinema_id"]), str(row["film_id"]), row["sale_date"],
                    int(row["online_tickets"]), row.get("online_gross"),
                ))

        r = _load("calendar")
        if r:
            ds.load_report.tables["calendar"] = r.ok
            ds.load_report.rejected["calendar"] = r.rejected
            entries = [CalendarEntry(row["date"], _day_type(row["day_type"]),
                                     row.get("holiday_name")) for row in r.rows]
            ds.calendar = Calendar(entries)

        return ds

    # ------------------------------------------------------------------
    # 便捷查询
    # ------------------------------------------------------------------
    def film(self, film_id: str) -> Film | None:
        return self.films.get(film_id)

    def hall(self, hall_id: str | None) -> Hall | None:
        return self.halls.get(hall_id) if hall_id else None

    def cinema(self, cinema_id: str) -> Cinema | None:
        return self.cinemas.get(cinema_id)

    def all_dates(self) -> list[dt.date]:
        dates = {s.start_time.date() for s in self.showtimes.values()}
        return sorted(dates)

    def showtimes_on(self, d: dt.date) -> list[Showtime]:
        return [s for s in self.showtimes.values() if s.start_time.date() == d]

    def orders_for(self, showtime_id: str) -> list[Order]:
        return self.orders_by_showtime.get(showtime_id, [])


def _day_type(raw: str) -> DayType:
    s = str(raw).strip().lower()
    mapping = {
        "holiday": DayType.HOLIDAY, "节假日": DayType.HOLIDAY, "法定节假日": DayType.HOLIDAY,
        "档期": DayType.HOLIDAY, "festival": DayType.HOLIDAY,
        "weekend": DayType.WEEKEND, "周末": DayType.WEEKEND,
        "workday": DayType.WORKDAY, "调休": DayType.WORKDAY, "工作日": DayType.WORKDAY,
        "normal": DayType.NORMAL, "平日": DayType.NORMAL,
    }
    return mapping.get(s, DayType.NORMAL)
