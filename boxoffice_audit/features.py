"""特征工程：把场次 + 订单聚合为可解释的统计特征。

所有特征都保留"分子/分母"，报告里可以直接向影院出示：
"该场售出 96 张，其中 84 张来自 2 个设备（87.5%）"。
"""

from __future__ import annotations

import datetime as dt
import statistics
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .config import (
    CHANNEL_GROUP, CHANNEL_ONLINE, EXCLUDED_FROM_DEVICE_CONCENTRATION, DetectorConfig,
)
from .store import Dataset, Order, Showtime, order_is_refunded


@dataclass
class ShowtimeFeatures:
    showtime_id: str
    cinema_id: str
    film_id: str
    hall_id: str | None
    start_time: dt.datetime

    planned_seats: int
    sold_tickets: int               # 有效售出（扣退票）
    gross: float                    # 有效收入
    refunded_tickets: int
    refunded_amount: float
    peak_tickets: int               # 退款前峰值（含后续退掉的）

    occupancy: float                # sold / planned
    peak_occupancy: float
    refund_rate: float              # refunded / peak

    avg_price: float
    settlement_price: float | None
    price_ratio: float | None       # avg_price / settlement_price

    channel_counts: dict[str, int]
    channel_gross: dict[str, float]
    online_tickets: int
    group_tickets: int
    counter_tickets: int
    online_ratio: float
    group_ratio: float

    device_counts: dict[str, int]
    account_counts: dict[str, int]
    top_device_share: float
    top_account_share: float
    device_count: int

    nonmember_tickets: int
    nonmember_ratio: float

    first_purchase: dt.datetime | None
    last_purchase: dt.datetime | None
    burst_tickets: int              # 开场前 X 分钟内售出
    burst_share: float
    presale_hours: float | None     # 最早购票距开场的时长
    single_order_share: float       # 最大单笔订单占比
    contiguous_ratio: float         # 连座占比（整排/大段连座）

    channel_settlement_price_ok: bool = True

    # 排除团体/包场渠道后的散客设备集中度（包场天然一机多票，不能计入）
    retail_tickets: int = 0
    retail_device_counts: dict[str, int] = field(default_factory=dict)
    retail_top_device_share: float = 0.0
    retail_device_count: int = 0


def _q(d: dict[str, float], k: str) -> float:
    return d.get(k, 0.0)


def compute_showtime_features(
    st: Showtime,
    orders: Sequence[Order],
    cfg: DetectorConfig,
    hall_format: str | None = None,
) -> ShowtimeFeatures:
    planned = st.planned_seats or 0
    # 可售座位缺失时用售出数兜底，避免除零
    if planned <= 0:
        planned = sum(o.seats for o in orders) or 1

    sold = 0
    gross = 0.0
    refunded_t = 0
    refunded_amt = 0.0
    peak_t = 0

    channel_counts: dict[str, int] = {}
    channel_gross: dict[str, float] = {}
    device_counts: dict[str, int] = {}
    account_counts: dict[str, int] = {}
    retail_device_counts: dict[str, int] = {}
    retail_tickets = 0
    nonmember = 0
    total_valid = 0

    purchase_times: list[dt.datetime] = []
    burst_window = dt.timedelta(minutes=10)

    order_ticket_sizes: list[int] = []
    contiguous_ok = 0
    contiguous_total = 0

    for o in orders:
        is_refund = order_is_refunded(o)
        if is_refund:
            refunded_t += o.seats
            refunded_amt += o.amount
            continue
        sold += o.seats
        gross += o.amount
        total_valid += o.seats
        channel_counts[o.channel] = channel_counts.get(o.channel, 0) + o.seats
        channel_gross[o.channel] = channel_gross.get(o.channel, 0.0) + o.amount
        if o.device_id:
            device_counts[o.device_id] = device_counts.get(o.device_id, 0) + o.seats
            if o.channel not in EXCLUDED_FROM_DEVICE_CONCENTRATION:
                retail_device_counts[o.device_id] = (
                    retail_device_counts.get(o.device_id, 0) + o.seats
                )
                retail_tickets += o.seats
        if o.account_id:
            account_counts[o.account_id] = account_counts.get(o.account_id, 0) + o.seats
        if o.is_member is False:
            nonmember += o.seats
        purchase_times.append(o.purchase_time)
        order_ticket_sizes.append(o.seats)
        # 连座：单笔 >=4 张且起止座位覆盖
        if o.seat_from is not None and o.seat_to is not None:
            contiguous_total += 1
            if o.seat_to - o.seat_from + 1 >= min(o.seats, 4) and o.seats >= 4:
                contiguous_ok += 1

    peak_t = sold + refunded_t

    avg_price = gross / sold if sold else 0.0
    settlement_price = st.settlement_price
    price_ratio = (avg_price / settlement_price) if settlement_price and settlement_price > 0 else None

    online_t = channel_counts.get(CHANNEL_ONLINE, 0)
    group_t = channel_counts.get(CHANNEL_GROUP, 0)
    counter_t = channel_counts.get("counter", 0)

    top_dev = max(device_counts.values()) / sold if sold and device_counts else 0.0
    top_acc = max(account_counts.values()) / sold if sold and account_counts else 0.0
    retail_top_dev = (
        max(retail_device_counts.values()) / retail_tickets
        if retail_tickets and retail_device_counts else 0.0
    )

    first_p = min(purchase_times) if purchase_times else None
    last_p = max(purchase_times) if purchase_times else None
    presale_hours = None
    burst = 0
    if first_p is not None:
        presale_hours = (st.start_time - first_p).total_seconds() / 3600.0
        # 开场后购票（现场购票）不计入 burst
        for o in orders:
            if order_is_refunded(o):
                continue
            if st.start_time - dt.timedelta(minutes=10) <= o.purchase_time <= st.start_time:
                burst += o.seats

    single_order_share = (max(order_ticket_sizes) / sold) if sold and order_ticket_sizes else 0.0

    return ShowtimeFeatures(
        showtime_id=st.showtime_id,
        cinema_id=st.cinema_id,
        film_id=st.film_id,
        hall_id=st.hall_id,
        start_time=st.start_time,
        planned_seats=planned,
        sold_tickets=sold,
        gross=round(gross, 2),
        refunded_tickets=refunded_t,
        refunded_amount=round(refunded_amt, 2),
        peak_tickets=peak_t,
        occupancy=sold / planned if planned else 0.0,
        peak_occupancy=peak_t / planned if planned else 0.0,
        refund_rate=refunded_t / peak_t if peak_t else 0.0,
        avg_price=round(avg_price, 2),
        settlement_price=settlement_price,
        price_ratio=price_ratio,
        channel_counts=channel_counts,
        channel_gross=channel_gross,
        online_tickets=online_t,
        group_tickets=group_t,
        counter_tickets=counter_t,
        online_ratio=online_t / sold if sold else 0.0,
        group_ratio=group_t / sold if sold else 0.0,
        device_counts=device_counts,
        account_counts=account_counts,
        top_device_share=top_dev,
        top_account_share=top_acc,
        device_count=len(device_counts),
        retail_tickets=retail_tickets,
        retail_device_counts=retail_device_counts,
        retail_top_device_share=retail_top_dev,
        retail_device_count=len(retail_device_counts),
        nonmember_tickets=nonmember,
        nonmember_ratio=nonmember / total_valid if total_valid else 0.0,
        first_purchase=first_p,
        last_purchase=last_p,
        burst_tickets=burst,
        burst_share=burst / sold if sold else 0.0,
        presale_hours=presale_hours,
        single_order_share=single_order_share,
        contiguous_ratio=(contiguous_ok / contiguous_total) if contiguous_total else 0.0,
    )


def build_all_features(ds: Dataset, cfg: DetectorConfig) -> dict[str, ShowtimeFeatures]:
    out: dict[str, ShowtimeFeatures] = {}
    for sid, st in ds.showtimes.items():
        orders = ds.orders_by_showtime.get(sid, [])
        hall = ds.halls.get(st.hall_id) if st.hall_id else None
        out[sid] = compute_showtime_features(
            st, orders, cfg, hall.hall_format if hall else None
        )
    return out
