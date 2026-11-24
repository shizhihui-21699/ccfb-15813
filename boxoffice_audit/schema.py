"""数据模型：表结构定义、列名别名映射、类型解析与校验。

监管数据来自不同院线/TMS/票务平台，列名不统一，且常有缺列、脏值。
本模块用"规范列 + 别名"的方式做归一化，并把问题行写入 rejects，
而不是让整批数据失败——日批处理必须可容错、可追溯。
"""

from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence


class FieldError(ValueError):
    """单行字段解析失败。"""


@dataclass(frozen=True)
class Column:
    name: str                       # 规范列名
    aliases: tuple[str, ...] = ()   # 可接受的源列名（含中文）
    kind: str = "str"               # str | int | float | datetime | date | bool
    required: bool = True
    nullable: bool = False

    def all_names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


@dataclass(frozen=True)
class TableSpec:
    name: str
    columns: tuple[Column, ...]
    primary_key: tuple[str, ...] = ()

    def column(self, name: str) -> Column:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)


_TRUE = {"1", "true", "yes", "y", "是", "真"}
_FALSE = {"0", "false", "no", "n", "否", "假"}


def parse_value(raw: str | None, col: Column) -> object:
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        if col.nullable:
            return None
        raise FieldError(f"{col.name} 为空")
    s = raw.strip()
    try:
        if col.kind == "str":
            return s
        if col.kind == "int":
            return int(float(s)) if s.replace(".", "", 1).replace("-", "", 1).isdigit() else int(s)
        if col.kind == "float":
            return float(s.replace(",", "").replace("¥", "").replace("￥", ""))
        if col.kind == "date":
            return _parse_date(s)
        if col.kind == "datetime":
            return _parse_datetime(s)
        if col.kind == "bool":
            low = s.lower()
            if low in _TRUE:
                return True
            if low in _FALSE:
                return False
            raise FieldError(f"{col.name}={raw!r} 不是布尔值")
    except FieldError:
        raise
    except (TypeError, ValueError) as exc:
        raise FieldError(f"{col.name}={raw!r} 无法解析为 {col.kind}: {exc}") from exc
    raise FieldError(f"未知类型 {col.kind}")


def _parse_date(s: str) -> dt.date:
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y.%m.%d"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    # 允许带时间
    return _parse_datetime(s).date()


def _parse_datetime(s: str) -> dt.datetime:
    s = s.strip().replace("/", "-")
    try:
        return dt.datetime.fromisoformat(s)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise FieldError(f"无法解析时间 {s!r}")


# ---------------------------------------------------------------------------
# 表定义
# ---------------------------------------------------------------------------

CINEMA_COLUMNS = (
    Column("cinema_id", ("影院编码", "影院ID", "cinema_code")),
    Column("name", ("影院名称", "影院", "cinema_name")),
    Column("province", ("省", "省份", "province_name"), nullable=True),
    Column("city", ("城市", "市", "city_name")),
    Column("tier", ("城市等级", "城市级别", "city_tier"), kind="int", nullable=True),
    Column("screens", ("银幕数", "screen_count"), kind="int", nullable=True),
)

FILM_COLUMNS = (
    Column("film_id", ("影片编码", "影片ID", "film_code")),
    Column("title", ("影片名称", "片名", "film_name")),
    Column("release_date", ("上映日期", "首映日期", "release"), kind="date", nullable=True),
    Column("is_import", ("是否进口", "import"), kind="bool", nullable=True),
)

HALL_COLUMNS = (
    Column("hall_id", ("影厅编码", "影厅ID", "hall_code")),
    Column("cinema_id", ("影院编码", "影院ID", "cinema_code")),
    Column("hall_format", ("影厅格式", "格式", "format"), nullable=True),
    Column("seats", ("座位数", "seat_count"), kind="int", nullable=True),
)

SHOWTIME_COLUMNS = (
    Column("showtime_id", ("场次编码", "场次ID", "showtime_code", "session_id")),
    Column("cinema_id", ("影院编码", "影院ID", "cinema_code")),
    Column("film_id", ("影片编码", "影片ID", "film_code")),
    Column("hall_id", ("影厅编码", "影厅ID", "hall_code"), nullable=True),
    Column("start_time", ("开场时间", "放映时间", "start"), kind="datetime"),
    Column("planned_seats", ("可售座位数", "座位数", "seat_count"), kind="int", nullable=True),
    # 该场次在票务/结算系统中登记的结算票价（用于低价票与结构票价识别）
    Column("settlement_price", ("结算票价", "最低票价", "base_price"), kind="float", nullable=True),
)

ORDER_COLUMNS = (
    Column("order_id", ("订单号", "订单编码", "order_code")),
    Column("showtime_id", ("场次编码", "场次ID", "showtime_code", "session_id")),
    Column("channel", ("渠道", "售票渠道", "sale_channel")),
    Column("seats", ("购票数", "票数", "数量", "ticket_count"), kind="int"),
    Column("unit_price", ("单价", "票价", "price"), kind="float"),
    Column("amount", ("金额", "总价", "实付金额", "total"), kind="float", nullable=True),
    Column("purchase_time", ("下单时间", "购票时间", "sale_time"), kind="datetime"),
    Column("device_id", ("设备号", "设备ID", "device"), nullable=True),
    Column("account_id", ("账号", "用户ID", "account", "user_id"), nullable=True),
    Column("is_member", ("是否会员", "会员", "member"), kind="bool", nullable=True),
    Column("status", ("状态", "订单状态", "order_status"), nullable=True),
    Column("refund_time", ("退票时间", "refund_time"), kind="datetime", nullable=True),
    Column("seat_row", ("排", "座位排", "row"), nullable=True),
    Column("seat_from", ("起始座位", "seat_start"), kind="int", nullable=True),
    Column("seat_to", ("结束座位", "seat_end"), kind="int", nullable=True),
)

SETTLEMENT_COLUMNS = (
    Column("settle_id", ("结算ID", "上报ID"), nullable=True),
    Column("cinema_id", ("影院编码", "影院ID", "cinema_code")),
    Column("showtime_id", ("场次编码", "场次ID", "showtime_code"), nullable=True),
    # 影院上报时把钱记到哪部影片名下（偷换票房关键：可能 != 排片影片）
    Column("filed_film_id", ("上报影片编码", "记账影片", "filed_film"), nullable=True),
    Column("reported_tickets", ("上报人次", "上报票数", "tickets"), kind="int", nullable=True),
    Column("reported_gross", ("上报票房", "上报金额", "gross"), kind="float", nullable=True),
    Column("reported_price", ("上报均价", "average_price"), kind="float", nullable=True),
    Column("settle_date", ("结算日期", "日期", "date"), kind="date", nullable=True),
)

ONLINE_SALES_COLUMNS = (
    Column("cinema_id", ("影院编码", "影院ID", "cinema_code")),
    Column("film_id", ("影片编码", "影片ID", "film_code")),
    Column("sale_date", ("日期", "date"), kind="date"),
    Column("online_tickets", ("线上人次", "线上票数", "tickets"), kind="int"),
    Column("online_gross", ("线上票房", "线上金额", "gross"), kind="float", nullable=True),
)

CALENDAR_COLUMNS = (
    Column("date", ("日期",), kind="date"),
    # holiday=法定节假日 | weekend=普通周末 | workday=调休上班 | normal=平日
    Column("day_type", ("日期类型", "类型", "type")),
    Column("holiday_name", ("节假日名称", "名称"), nullable=True),
)

TABLES: dict[str, TableSpec] = {
    "cinemas": TableSpec("cinemas", CINEMA_COLUMNS, ("cinema_id",)),
    "films": TableSpec("films", FILM_COLUMNS, ("film_id",)),
    "halls": TableSpec("halls", HALL_COLUMNS, ("hall_id",)),
    "showtimes": TableSpec("showtimes", SHOWTIME_COLUMNS, ("showtime_id",)),
    "orders": TableSpec("orders", ORDER_COLUMNS, ("order_id",)),
    "settlements": TableSpec("settlements", SETTLEMENT_COLUMNS),
    "online_sales": TableSpec("online_sales", ONLINE_SALES_COLUMNS),
    "calendar": TableSpec("calendar", CALENDAR_COLUMNS, ("date",)),
}


def resolve_header(spec: TableSpec, header: Sequence[str]) -> dict[str, str | None]:
    """把源表头映射到规范列名；返回 {规范列: 源列名或 None}。"""
    normalized = {str(h).strip(): h for h in header}
    lowered = {str(h).strip().lower(): h for h in header}
    mapping: dict[str, str | None] = {}
    for col in spec.columns:
        found = None
        for alias in col.all_names():
            if alias in normalized:
                found = normalized[alias]
                break
            if alias.lower() in lowered:
                found = lowered[alias.lower()]
                break
        mapping[col.name] = found
    return mapping


def missing_required(spec: TableSpec, mapping: dict[str, str | None]) -> list[str]:
    """真正缺失的必填列。

    可空列（nullable=True）在源表中不存在是允许的：当作"该字段无数据"，
    逐行填 None。这与"必填列缺失导致文件不可用"是两种不同情况。
    """
    return [
        c.name for c in spec.columns
        if c.required and not c.nullable and mapping.get(c.name) is None
    ]


@dataclass
class RejectedRow:
    line_no: int
    row: dict[str, str]
    reason: str


@dataclass
class LoadResult:
    table: str
    rows: list[dict[str, object]] = field(default_factory=list)
    rejected: list[RejectedRow] = field(default_factory=list)
    source_rows: int = 0

    @property
    def ok(self) -> int:
        return len(self.rows)


def read_table(
    path: str | Path,
    table: str,
    *,
    max_rejects: int = 1000,
    encoding: str | None = None,
) -> LoadResult:
    """读取一张 CSV，返回规范化后的行。文件级错误直接抛出。"""
    spec = TABLES[table]
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"缺少数据表文件: {p}")

    encodings = [encoding] if encoding else ["utf-8-sig", "utf-8", "gb18030"]
    last_err: Exception | None = None
    fh = None
    for enc in encodings:
        try:
            fh = open(p, "r", encoding=enc, newline="")
            break
        except (UnicodeDecodeError, LookupError) as exc:
            last_err = exc
            continue
    if fh is None:
        raise ValueError(f"无法识别 {p} 的编码: {last_err}")

    result = LoadResult(table=table)
    with fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        mapping = resolve_header(spec, header)
        miss = missing_required(spec, mapping)
        if miss:
            raise ValueError(f"{p} 缺少必需列: {miss}；实际表头: {header}")

        for i, raw in enumerate(reader, start=2):
            result.source_rows += 1
            rec: dict[str, object] = {}
            try:
                for col in spec.columns:
                    src = mapping.get(col.name)
                    value = None if src is None else raw.get(src)
                    # 缺失列且非必填 -> None
                    if src is None and not col.required:
                        rec[col.name] = None
                        continue
                    rec[col.name] = parse_value(value, col)
            except FieldError as exc:
                if len(result.rejected) < max_rejects:
                    result.rejected.append(RejectedRow(i, dict(raw), str(exc)))
                continue
            # 派生：金额缺省时由单价*票数计算
            if table == "orders" and rec.get("amount") is None:
                rec["amount"] = round(float(rec["unit_price"]) * int(rec["seats"]), 2)
            result.rows.append(rec)
    return result
