"""上下文层：日期类型、影片生命周期、时段分桶、影厅分层。

误报的根源是把"正常的异常"当成"异常的异常"。本模块把这些正常因素
显式建模成条件变量，后续基线只在同一条件下比较：

- 日期类型（节假日/周末/调休工作日/平日）：节假日观影高峰是正常的
- 影片生命周期（首映日/首周/长线）：首映零点场是正常高峰
- 时段分桶：不同时段的正常上座率本来就不同
- 影厅格式：特效厅上座率天然偏高

这些变量既用于基线分层，也用于生成"误报抑制"标记（mitigator）。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

from .config import is_late_night, PREMIUM_FORMATS


class DayType(str, Enum):
    HOLIDAY = "holiday"      # 法定节假日、档期
    WEEKEND = "weekend"      # 普通周末
    WORKDAY = "workday"      # 调休上班的周末
    NORMAL = "normal"        # 平日


class FilmStage(str, Enum):
    PREMIERE = "premiere"    # 上映首日
    OPENING = "opening"      # 首周（2-7 天）
    RUN = "run"              # 常规放映
    UNKNOWN = "unknown"


# 首映窗口：首映日及其后 N 天观影热度显著偏高
PREMIERE_WINDOW_DAYS = 1
OPENING_WINDOW_DAYS = 7


def time_bucket(hour: int) -> str:
    """2 小时分桶，用于按时段做基线对照。"""
    return f"{hour // 2 * 2:02d}"


@dataclass(frozen=True)
class CalendarEntry:
    date: dt.date
    day_type: DayType
    holiday_name: str | None = None


class Calendar:
    """日期类型解析。优先使用监管方提供的放假安排，缺失则退化为周末/平日。"""

    def __init__(self, entries: Iterable[CalendarEntry] | None = None):
        self._map: dict[dt.date, CalendarEntry] = {e.date: e for e in (entries or [])}
        self.used_external = bool(self._map)

    def day_type(self, d: dt.date) -> DayType:
        e = self._map.get(d)
        if e:
            return e.day_type
        if d.weekday() >= 5:
            return DayType.WEEKEND
        return DayType.NORMAL

    def holiday_name(self, d: dt.date) -> str | None:
        e = self._map.get(d)
        return e.holiday_name if e else None

    def is_holiday(self, d: dt.date) -> bool:
        return self.day_type(d) == DayType.HOLIDAY


def film_stage(release_date: dt.date | None, show_date: dt.date) -> FilmStage:
    if release_date is None:
        return FilmStage.UNKNOWN
    delta = (show_date - release_date).days
    if delta < 0:
        return FilmStage.UNKNOWN          # 点映/超前点映，另行判断
    if delta < PREMIERE_WINDOW_DAYS:
        return FilmStage.PREMIERE
    if delta < OPENING_WINDOW_DAYS:
        return FilmStage.OPENING
    return FilmStage.RUN


def is_premiere_midnight(release_date: dt.date | None, start: dt.datetime) -> bool:
    """首映零点场：上映当天凌晨的场次，属于正常观影高峰。"""
    if release_date is None:
        return False
    return start.date() == release_date and is_late_night(start.hour)


def hall_tier(hall_format: str | None) -> str:
    """影厅分层：特效厅单独一层，避免其高上座被误判。"""
    if hall_format and str(hall_format).upper() in PREMIUM_FORMATS:
        return "premium"
    return "standard"


@dataclass(frozen=True)
class ShowContext:
    """单场次的上下文特征，供检测器与基线分层使用。"""

    date: dt.date
    hour: int
    bucket: str
    late_night: bool
    day_type: DayType
    film_stage: FilmStage
    premiere_midnight: bool
    hall_tier: str

    @property
    def baseline_key(self) -> tuple[str, str, str]:
        """(日期类型, 时段桶, 影厅层级) —— 基线分层键。"""
        return (self.day_type.value, self.bucket, self.hall_tier)


def build_context(
    start: dt.datetime,
    calendar: Calendar,
    release_date: dt.date | None,
    hall_format: str | None,
) -> ShowContext:
    d = start.date()
    return ShowContext(
        date=d,
        hour=start.hour,
        bucket=time_bucket(start.hour),
        late_night=is_late_night(start.hour),
        day_type=calendar.day_type(d),
        film_stage=film_stage(release_date, d),
        premiere_midnight=is_premiere_midnight(release_date, start),
        hall_tier=hall_tier(hall_format),
    )
