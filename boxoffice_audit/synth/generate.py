"""合成数据生成器：用于端到端验证与调参。

生成内容：
- 正常影院日常经营（含工作日/周末/节假日明显不同的上座率）；
- 正常但"看起来异常"的情形（关键对照，用于误报测试）：
  * 节假日正常观影高峰；
  * 首映零点场（真实高上座）；
  * 合法包场/团购（团体渠道、单笔大额、提前预订、有折扣）；
  * 特效厅正常高上座。
- 注入的作弊（带 ground truth 标签，供评估）：
  * 幽灵场：深夜满座、设备集中、极短时间售出；
  * 锁场退票：开场前集中购票、临近开场大量退票；
  * 低价刷量：1 元票大量售出；
  * 偷换票房：排片 A 上报 B；线上销量 > 上报；上报票价 < 实付。

ground truth 写入 truth.csv，字段：kind, cinema_id, film_id, date, showtime_id
kind: fake_ghost / fake_refund / fake_lowprice / div_mismatch / div_online / div_price
"""

from __future__ import annotations

import csv
import datetime as dt
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

TIERED_CITIES = [
    ("一线", 1, "北京"), ("一线", 1, "上海"),
    ("二线", 2, "成都"), ("二线", 2, "杭州"), ("二线", 2, "武汉"),
    ("三线", 3, "洛阳"), ("三线", 3, "绵阳"), ("四线", 4, "曲靖"),
]

FILM_POOL = [
    # (film_id, title, is_hot, 排片权重)
    ("F001", "星河远征", True, 5),
    ("F002", "长安夜", True, 4),
    ("F003", "深海回声", True, 3),
    ("F004", "夏日来信", False, 2),
    ("F005", "小城故事", False, 1.5),
    ("F006", "山那边的风", False, 1),
    ("F007", "旧巷灯火", False, 0.8),
    ("F008", "机器之心", True, 3.5),
]

HALL_FORMATS = ["standard", "standard", "standard", "standard", "IMAX", "DOLBY"]
CHANNELS_NORMAL = (
    ["online"] * 68 + ["counter"] * 12 + ["self"] * 18 + ["group"] * 2
)


@dataclass
class SyntheticConfig:
    days: int = 60
    cinemas: int = 40
    seed: int = 7
    inject: bool = True
    start_date: dt.date = dt.date(2026, 9, 1)
    # 注入密度：工作日数占比
    ghost_rate: float = 0.18
    refund_rate: float = 0.12
    lowprice_rate: float = 0.12
    mismatch_rate: float = 0.10
    online_rate: float = 0.10
    price_rate: float = 0.10


# ---------------------------------------------------------------------------
# 日历：构造国庆档 + 中秋 + 周末 + 调休
# ---------------------------------------------------------------------------
def build_calendar(start: dt.date, days: int) -> list[tuple[dt.date, str, str]]:
    """返回 (date, day_type, holiday_name)。"""
    out = []
    for i in range(days):
        d = start + dt.timedelta(days=i)
        # 2026 国庆前后：10-01..10-07 为节假日，9-27/10-10 调休上班
        if dt.date(2026, 10, 1) <= d <= dt.date(2026, 10, 7):
            out.append((d, "holiday", "国庆档"))
        elif d in (dt.date(2026, 9, 27), dt.date(2026, 10, 10)):
            out.append((d, "workday", "调休"))
        elif d.weekday() >= 5:
            out.append((d, "weekend", ""))
        else:
            out.append((d, "normal", ""))
    return out


def _day_multiplier(day_type: str) -> float:
    return {"holiday": 1.9, "weekend": 1.45, "workday": 0.95, "normal": 1.0}[day_type]


def _hour_profile(day_type: str) -> list[tuple[int, float]]:
    """各时段排片权重与基准上座率乘子。"""
    # (hour, weight)
    base = [
        (9, 0.4), (11, 0.6), (13, 0.9), (15, 1.0), (17, 0.9),
        (19, 1.3), (21, 1.2), (23, 0.35), (1, 0.12), (3, 0.05),
    ]
    if day_type == "holiday":
        return [(h, w * (1.25 if h in (9, 11, 13, 15) else 1.0)) for h, w in base]
    if day_type == "weekend":
        return [(h, w * (1.1 if h in (13, 15) else 1.0)) for h, w in base]
    return base


def _base_occupancy(hour: int, day_type: str, hall_format: str, is_hot: bool) -> float:
    prof = {
        9: 0.10, 11: 0.14, 13: 0.22, 15: 0.24, 17: 0.22,
        19: 0.42, 21: 0.36, 23: 0.08, 1: 0.03, 3: 0.02,
    }
    occ = prof.get(hour, 0.1)
    occ *= _day_multiplier(day_type)
    if is_hot:
        occ *= 1.35
    else:
        occ *= 0.7
    if hall_format in ("IMAX", "DOLBY"):
        occ *= 1.25
    return min(occ, 0.92)


# ---------------------------------------------------------------------------
@dataclass
class _Ctx:
    rng: random.Random
    rows: dict[str, list] = field(default_factory=lambda: {
        "cinemas": [], "films": [], "halls": [], "showtimes": [],
        "orders": [], "settlements": [], "online_sales": [], "calendar": [],
    })
    truth: list[dict] = field(default_factory=list)
    order_seq: int = 0
    st_seq: int = 0

    def next_order(self) -> str:
        self.order_seq += 1
        return f"O{self.order_seq:09d}"

    def next_showtime(self) -> str:
        self.st_seq += 1
        return f"S{self.st_seq:08d}"


def _make_seats(rng: random.Random, hall_format: str) -> int:
    if hall_format == "IMAX":
        return rng.choice([280, 320, 360])
    if hall_format == "DOLBY":
        return rng.choice([180, 220])
    return rng.choice([80, 100, 120, 140, 160])


def generate_dataset(
    out_dir: str | Path,
    days: int = 60,
    cinemas: int = 40,
    seed: int = 7,
    inject: bool = True,
) -> dict:
    """生成合成数据集并写入 CSV。返回统计信息。"""
    rng = random.Random(seed)
    cfg = SyntheticConfig(days=days, cinemas=cinemas, seed=seed, inject=inject)
    ctx = _Ctx(rng)

    start = cfg.start_date
    calendar = build_calendar(start, days)
    cal_map = {d: (t, name) for d, t, name in calendar}
    for d, t, name in calendar:
        ctx.rows["calendar"].append({
            "date": d.isoformat(), "day_type": t, "holiday_name": name,
        })

    # 影片
    film_meta = {}
    for fid, title, is_hot, _w in FILM_POOL:
        # 让影片错峰上映：热门片在上映初期
        release = start + dt.timedelta(days=rng.randint(-20, 25))
        film_meta[fid] = {"title": title, "is_hot": is_hot, "release": release}
        ctx.rows["films"].append({
            "film_id": fid, "title": title, "release_date": release.isoformat(),
            "is_import": "1" if rng.random() < 0.3 else "0",
        })

    # 影院与影厅
    cinema_objs = []
    for i in range(cinemas):
        tier_name, tier, city = TIERED_CITIES[i % len(TIERED_CITIES)]
        cid = f"C{i+1:04d}"
        city_suffix = rng.choice(["万达", "中影", "星轶", "横店", "CGV"])
        name = f"{city}{city_suffix}影城{i+1}号店"
        n_hall = rng.choice([6, 8, 10, 12])
        halls = []
        for h in range(n_hall):
            hid = f"{cid}H{h+1:02d}"
            fmt = rng.choice(HALL_FORMATS)
            seats = _make_seats(rng, fmt)
            halls.append((hid, fmt, seats))
            ctx.rows["halls"].append({
                "hall_id": hid, "cinema_id": cid, "hall_format": fmt, "seats": seats,
            })
        cinema_objs.append({
            "cinema_id": cid, "name": name, "province": city, "city": city,
            "tier": tier, "screens": n_hall, "halls": halls,
        })
        ctx.rows["cinemas"].append({
            "cinema_id": cid, "name": name, "province": city, "city": city,
            "tier": tier, "screens": n_hall,
        })

    # 逐日、逐影院生成排片与售票
    for i in range(days):
        d = start + dt.timedelta(days=i)
        day_type, holiday_name = cal_map[d]
        for cine in cinema_objs:
            cid = cine["cinema_id"]
            tier = cine["tier"]
            _generate_cinema_day(ctx, cine, d, day_type, film_meta, inject=inject)

    _write_csvs(Path(out_dir), ctx.rows)
    _write_truth(Path(out_dir), ctx.truth)
    return {
        "cinemas": len(ctx.rows["cinemas"]),
        "films": len(ctx.rows["films"]),
        "showtimes": len(ctx.rows["showtimes"]),
        "orders": len(ctx.rows["orders"]),
        "truth": len(ctx.truth),
    }


def _generate_cinema_day(
    ctx: _Ctx, cine: dict, d: dt.date, day_type: str,
    film_meta: dict, inject: bool,
) -> None:
    rng = ctx.rng
    cid = cine["cinema_id"]
    tier = cine["tier"]
    halls = cine["halls"]

    # 今日上映影片（按热度加权）
    active = []
    for fid, meta in film_meta.items():
        if meta["release"] <= d:
            active.append(fid)
    if not active:
        active = ["F001"]
    weights = [film_meta[f]["is_hot"] and 3.0 or 1.0 for f in active]

    # 每家影院当天排片：每个影厅 2-4 场
    for hid, hfmt, seats in halls:
        n_show = rng.choice([2, 3, 3, 4])
        used_hours: set[int] = set()
        for _ in range(n_show):
            hour = rng.choice([9, 11, 13, 15, 17, 19, 21, 23, 1, 3])
            if hour in used_hours:
                continue
            used_hours.add(hour)
            fid = rng.choices(active, weights=weights[:len(active)])[0]
            meta = film_meta[fid]
            _emit_showtime(ctx, cid, hid, hfmt, seats, fid, meta, d, hour, day_type)

    # 注入异常（仅工作日/平日，避免节假日节假日已高上座导致识别无意义）
    if not inject:
        return
    # 只有 tier<=3 的城市部分影院作弊，制造"这种影院"的集中性
    if tier >= 4 and rng.random() < 0.7:
        return
    n_inject = 0
    if day_type in ("normal", "workday"):
        r = rng.random()
        if r < 0.10:  # 约 10% 影院-日 有作弊
            _inject_fake(ctx, cine, d, film_meta)
        elif r < 0.20:  # 另有约 10% 影院-日 偷换票房
            _inject_diversion(ctx, cine, d, film_meta)
    elif day_type in ("weekend", "holiday"):
        # 节假日也可能作弊，但用"包场伪装"为主，测试是否被降权
        if rng.random() < 0.04:
            _inject_disguised(ctx, cine, d, film_meta)


def _emit_showtime(
    ctx: _Ctx, cid: str, hid: str, hfmt: str, seats: int,
    fid: str, meta: dict, d: dt.date, hour: int, day_type: str,
    *, force_occ: float | None = None, n_orders: int | None = None,
    channel_profile: Sequence[str] | None = None,
    price_override: float | None = None,
    device_pool: int | None = None,
    purchase_window_min: float | None = None,
    filed_film: str | None = None,
    online_extra: int = 0,
    report_price_ratio: float = 1.0,
    report_tickets_ratio: float = 1.0,
) -> str:
    """生成一个场次及其订单/结算/线上记录。"""
    rng = ctx.rng
    sid = ctx.next_showtime()
    # 上映日期决定是否首映零点场
    release = meta["release"]
    start_dt = dt.datetime.combine(d, dt.time(hour, rng.choice([0, 15, 30, 45])))
    if hour < 6:
        start_dt = dt.datetime.combine(d, dt.time(hour, rng.choice([0, 10, 20])))

    occ = force_occ if force_occ is not None else _base_occupancy(hour, day_type, hfmt, meta["is_hot"])
    occ = min(max(occ, 0.0), 1.0)
    sold_target = int(round(seats * occ))

    base_price = {"1": 55.0, "2": 45.0, "3": 38.0, "4": 32.0}.get("2", 45.0)
    price = price_override if price_override is not None else base_price + rng.uniform(-6, 8)
    price = max(9.9, round(price, 1))

    # 结算价（影院上报的最低结算价，一般略低于售价）
    settle_price = round(price * rng.uniform(0.85, 0.98), 1)

    ctx.rows["showtimes"].append({
        "showtime_id": sid, "cinema_id": cid, "film_id": fid, "hall_id": hid,
        "start_time": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "planned_seats": seats, "settlement_price": settle_price,
    })

    # 生成订单
    remaining = sold_target
    channels = list(channel_profile) if channel_profile else CHANNELS_NORMAL
    devices = [f"DEV{ctx.rng.randint(1, device_pool)}" for _ in range(1)] if device_pool else None
    order_count = n_orders if n_orders is not None else max(1, sold_target // rng.choice([2, 3, 4]))
    if sold_target == 0:
        order_count = 0

    acc = 0
    online_seats = 0
    for j in range(order_count):
        if remaining <= 0:
            break
        if j == order_count - 1:
            seats_n = remaining
        else:
            seats_n = max(1, min(remaining, rng.choice([1, 1, 2, 2, 3, 4])))
        remaining -= seats_n
        ch = rng.choice(channels)
        if ch == "online":
            online_seats += seats_n
        # 购票时间
        if purchase_window_min is not None:
            # 全部在下单窗口内（刷票特征）
            minutes_before = rng.uniform(0, purchase_window_min)
        elif ch == "group":
            minutes_before = rng.uniform(60 * 18, 60 * 72)   # 团购提前
        else:
            minutes_before = rng.uniform(30, 60 * 48)
        # 现场柜台购票可能接近开场
        if ch in ("counter", "self") and rng.random() < 0.4:
            minutes_before = rng.uniform(0, 60)
        purchase = start_dt - dt.timedelta(minutes=minutes_before)
        if purchase > start_dt:
            purchase = start_dt - dt.timedelta(minutes=5)

        dev = devices[0] if (devices and ch != "group") else f"DEV{rng.randint(1, 5000)}"
        account = f"U{rng.randint(1, 100000)}"
        is_member = "0" if rng.random() < 0.55 else "1"
        if ch == "group":
            is_member = "1"
        row_seat = str(rng.randint(1, 12))
        ctx.rows["orders"].append({
            "order_id": ctx.next_order(), "showtime_id": sid, "channel": ch,
            "seats": seats_n, "unit_price": round(price, 2),
            "amount": round(price * seats_n, 2),
            "purchase_time": purchase.strftime("%Y-%m-%d %H:%M:%S"),
            "device_id": dev, "account_id": account, "is_member": is_member,
            "status": "valid", "refund_time": "",
            "seat_row": row_seat, "seat_from": 1, "seat_to": min(seats_n, 20),
        })

    # 结算上报
    filed = filed_film or fid
    rep_price = round(price * report_price_ratio, 2)
    rep_tickets = int(round(sold_target * report_tickets_ratio))
    ctx.rows["settlements"].append({
        "settle_id": f"ST{ctx.next_order()}", "cinema_id": cid, "showtime_id": sid,
        "filed_film_id": filed, "reported_tickets": rep_tickets,
        "reported_gross": round(rep_price * rep_tickets, 2),
        "reported_price": rep_price, "settle_date": d.isoformat(),
    })

    # 线上销售记录：只对线上渠道占比高的场次生成，且正常情况下线上票量
    # 不应超过上报票量（线上平台订单会进入影院系统）。偷换场景由
    # online_extra 模拟"线上卖了但没入系统"。
    if online_extra:
        ctx.rows["online_sales"].append({
            "cinema_id": cid, "film_id": filed, "sale_date": d.isoformat(),
            "online_tickets": sold_target + online_extra,
            "online_gross": round(price * (sold_target + online_extra), 2),
        })
    elif sold_target > 0 and not filed_film:
        # 正常场次：线上票量 = 线上渠道售票数（与上报一致，不产生缺口）
        if online_seats > 0:
            ctx.rows["online_sales"].append({
                "cinema_id": cid, "film_id": fid, "sale_date": d.isoformat(),
                "online_tickets": online_seats, "online_gross": round(price * online_seats, 2),
            })
    return sid


def _trim_reported_for_film(
    ctx: _Ctx, cid: str, fid: str, d: dt.date, ratio: float
) -> None:
    """把该影院当日该片已生成的结算记录的上报票量按比例压缩。"""
    day_prefix = d.isoformat()
    # 该影院当日该片的场次
    sids = {
        s["showtime_id"] for s in ctx.rows["showtimes"]
        if s["cinema_id"] == cid and s["film_id"] == fid
        and s["start_time"].startswith(day_prefix)
    }
    for s in ctx.rows["settlements"]:
        if s["showtime_id"] in sids and s["filed_film_id"] == fid:
            s["reported_tickets"] = int(s["reported_tickets"] * ratio)
            s["reported_gross"] = round(s["reported_gross"] * ratio, 2)


def _inject_fake(ctx: _Ctx, cine: dict, d: dt.date, film_meta: dict) -> None:
    """注入虚假票房：幽灵场 / 锁场退票 / 低价刷量。"""
    rng = ctx.rng
    cid = cine["cinema_id"]
    halls = cine["halls"]
    fid = rng.choice([f for f in film_meta if film_meta[f]["release"] <= d] or ["F001"])
    meta = film_meta[fid]
    hid, hfmt, seats = rng.choice(halls)
    kind = rng.choices(
        ["ghost", "refund", "lowprice"], weights=[0.45, 0.3, 0.25]
    )[0]

    if kind == "ghost":
        # 深夜满座，设备集中，极短窗口售出
        hour = rng.choice([23, 1, 3])
        sid = _emit_showtime(
            ctx, cid, hid, hfmt, seats, fid, meta, d, hour, "normal",
            force_occ=rng.uniform(0.92, 1.0), n_orders=rng.randint(1, 3),
            channel_profile=["self", "counter", "online"],
            device_pool=rng.randint(1, 2),
            purchase_window_min=rng.uniform(5, 20),
        )
        ctx.truth.append({"kind": "fake_ghost", "cinema_id": cid, "film_id": fid,
                          "date": d.isoformat(), "showtime_id": sid})
    elif kind == "refund":
        hour = rng.choice([19, 21, 23])
        sid = _emit_showtime(
            ctx, cid, hid, hfmt, seats, fid, meta, d, hour, "normal",
            force_occ=rng.uniform(0.8, 0.98), n_orders=rng.randint(2, 4),
            device_pool=rng.randint(1, 3),
        )
        # 把约 80% 的"票数"标记退票（按票数累计，而不是按订单笔数——
        # 否则最后一笔大订单会留下大部分票，退票率看起来很低的假象）
        orders = [o for o in ctx.rows["orders"] if o["showtime_id"] == sid]
        st = dt.datetime.fromisoformat(
            next(s["start_time"] for s in ctx.rows["showtimes"] if s["showtime_id"] == sid)
        )
        total = sum(o["seats"] for o in orders)
        target = total * 0.8
        acc = 0
        for o in orders:
            if acc >= target:
                break
            o["status"] = "refund"
            o["refund_time"] = (
                st - dt.timedelta(minutes=rng.uniform(1, 30))
            ).strftime("%Y-%m-%d %H:%M:%S")
            acc += o["seats"]
        ctx.truth.append({"kind": "fake_refund", "cinema_id": cid, "film_id": fid,
                          "date": d.isoformat(), "showtime_id": sid})
    else:  # lowprice
        hour = rng.choice([11, 13, 15])
        sid = _emit_showtime(
            ctx, cid, hid, hfmt, seats, fid, meta, d, hour, "normal",
            force_occ=rng.uniform(0.6, 0.85), n_orders=rng.randint(2, 4),
            price_override=1.0, channel_profile=["self", "counter"],
            device_pool=rng.randint(1, 3),
        )
        ctx.truth.append({"kind": "fake_lowprice", "cinema_id": cid, "film_id": fid,
                          "date": d.isoformat(), "showtime_id": sid})


def _inject_disguised(ctx: _Ctx, cine: dict, d: dt.date, film_meta: dict) -> None:
    """节假日/周末注入"像刷票但合法"的包场，用于误报测试。

    合法包场特征：团体渠道、单笔大额、提前预订、有折扣（不低价）、无退票。
    """
    rng = ctx.rng
    cid = cine["cinema_id"]
    hid, hfmt, seats = rng.choice(cine["halls"])
    fid = rng.choice(list(film_meta))
    meta = film_meta[fid]
    hour = rng.choice([13, 15, 19])
    sid = _emit_showtime(
        ctx, cid, hid, hfmt, seats, fid, meta, d, hour, "holiday",
        force_occ=rng.uniform(0.75, 0.95), n_orders=1,
        channel_profile=["group"],
    )
    # 折扣但不低价
    orders = [o for o in ctx.rows["orders"] if o["showtime_id"] == sid]
    for o in orders:
        o["unit_price"] = round(o["unit_price"] * 0.72, 2)
        o["amount"] = round(o["unit_price"] * o["seats"], 2)
    ctx.truth.append({"kind": "legit_group", "cinema_id": cid, "film_id": fid,
                      "date": d.isoformat(), "showtime_id": sid})


def _inject_diversion(ctx: _Ctx, cine: dict, d: dt.date, film_meta: dict) -> None:
    """注入偷换票房：排片记账不符 / 线上销量未入系统 / 结构票价。"""
    rng = ctx.rng
    cid = cine["cinema_id"]
    hid, hfmt, seats = rng.choice(cine["halls"])
    hot = [f for f in film_meta if film_meta[f]["is_hot"] and film_meta[f]["release"] <= d]
    cold = [f for f in film_meta if not film_meta[f]["is_hot"] and film_meta[f]["release"] <= d]
    if not hot or not cold:
        return
    hot_f = rng.choice(hot)
    cold_f = rng.choice(cold)
    hour = rng.choice([19, 21])
    kind = rng.choices(
        ["mismatch", "online", "price"], weights=[0.4, 0.3, 0.3]
    )[0]

    if kind == "mismatch":
        # 排片热门片，但记账到冷门片
        sid = _emit_showtime(
            ctx, cid, hid, hfmt, seats, hot_f, film_meta[hot_f], d, hour, "normal",
            force_occ=rng.uniform(0.5, 0.8), filed_film=cold_f,
        )
        ctx.truth.append({"kind": "div_mismatch", "cinema_id": cid, "film_id": hot_f,
                          "date": d.isoformat(), "showtime_id": sid})
    elif kind == "online":
        # 线上卖了票，但影院整体少报该片当日票量（截留）。
        # 真实偷换不会只压一个场次——那会被同片其他场次的上报掩盖，
        # 所以这里把该影院当日同片的正常场次一并压缩上报。
        _trim_reported_for_film(ctx, cid, hot_f, d, ratio=0.45)
        sid = _emit_showtime(
            ctx, cid, hid, hfmt, seats, hot_f, film_meta[hot_f], d, hour, "normal",
            force_occ=rng.uniform(0.4, 0.7),
            online_extra=rng.randint(30, 90), report_tickets_ratio=0.45,
        )
        ctx.truth.append({"kind": "div_online", "cinema_id": cid, "film_id": hot_f,
                          "date": d.isoformat(), "showtime_id": sid})
    else:  # price
        # 观众实付正常，但上报结算价压低（结构票价）
        sid = _emit_showtime(
            ctx, cid, hid, hfmt, seats, hot_f, film_meta[hot_f], d, hour, "normal",
            force_occ=rng.uniform(0.45, 0.75), report_price_ratio=0.6,
        )
        ctx.truth.append({"kind": "div_price", "cinema_id": cid, "film_id": hot_f,
                          "date": d.isoformat(), "showtime_id": sid})


def _write_csvs(out: Path, rows: dict[str, list]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, data in rows.items():
        if not data:
            continue
        p = out / f"{name}.csv"
        with open(p, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)


def _write_truth(out: Path, truth: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    p = out / "truth.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["kind", "cinema_id", "film_id", "date", "showtime_id"])
        w.writeheader()
        w.writerows(truth)


def build_selftest_dataset(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    generate_dataset(out, days=60, cinemas=40, seed=7, inject=True)
    return out
