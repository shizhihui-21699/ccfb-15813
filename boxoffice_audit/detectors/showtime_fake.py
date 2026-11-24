"""虚假票房检测（F 类）：刷票房、幽灵场、锁场、低价票。

判定逻辑（全部为"场次级"，输出可疑场次清单）：
- 只有在"异常上座/异常占比"前提下，其他证据才有意义；
- 深夜场必须以同日同时段同层影院为基线，而非固定阈值；
- 包场/团购特征（团体渠道、单笔大额、提前预订、合理折扣）触发降权；
- 首映零点场、节假日使用专用上下文降权/换基线。
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from typing import Sequence

from ..baseline import BaselineBook, robust_z
from ..calendar_ctx import ShowContext, build_context, DayType, FilmStage
from ..config import (
    CHANNEL_GROUP, CHANNEL_ONLINE, DetectorConfig, FAKE_RULES, MITIGATORS,
)
from ..evidence import DetectorResult, Evidence, Finding, FindingType
from ..features import ShowtimeFeatures
from ..store import Dataset

_RULE = {r.code: r for r in FAKE_RULES}
_MIT = {r.code: r for r in MITIGATORS}


def _ev(code: str, value: float, threshold: float | None, message: str, **extra) -> Evidence:
    r = _RULE[code]
    return Evidence(r.code, r.label, r.weight, value, threshold, message, r.action,
                    extra, primary=r.primary)


def _mitigator_flags(
    f: ShowtimeFeatures, ctx: ShowContext, cfg: DetectorConfig
) -> list[str]:
    """识别"看起来像刷票其实是正常"的情形，用于降权。"""
    out: list[str] = []
    if ctx.day_type == DayType.HOLIDAY:
        out.append("M01")
    if ctx.premiere_midnight:
        out.append("M02")
    if ctx.hall_tier == "premium":
        out.append("M04")
    # 疑似正常包场/团购
    group_like = (
        f.group_ratio >= cfg.group_ratio
        or (f.single_order_share >= cfg.group_single_order_share and f.sold_tickets >= 20)
    )
    presale = f.presale_hours is not None and f.presale_hours >= cfg.group_presale_hours
    discounted = f.price_ratio is not None and f.price_ratio <= (1 - cfg.group_price_discount)
    if group_like and presale and not discounted:
        out.append("M03")
    return out


class FakeBoxOfficeDetector:
    def __init__(self, ds: Dataset, features: dict[str, ShowtimeFeatures], cfg: DetectorConfig):
        self.ds = ds
        self.f = features
        self.cfg = cfg
        self._contexts: dict[str, ShowContext] = {}
        self._occ_baseline: BaselineBook | None = None
        self._price_by_city: dict[str, list[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    def prepare(self) -> None:
        cfg = self.cfg
        # 上下文 + 上座率分层基线
        self._occ_baseline = BaselineBook(min_samples=cfg.baseline_min_samples)
        for sid, f in self.f.items():
            st = self.ds.showtimes[sid]
            film = self.ds.films.get(f.film_id)
            hall = self.ds.halls.get(st.hall_id) if st.hall_id else None
            ctx = build_context(
                st.start_time, self.ds.calendar,
                film.release_date if film else None,
                hall.hall_format if hall else None,
            )
            self._contexts[sid] = ctx
            # 只用"有效场次"建基线，避免极端刷票场污染基线
            if f.sold_tickets >= 3 and f.occupancy <= 0.98:
                self._occ_baseline.add(ctx.baseline_key, f.occupancy)
            city = self._city(f.cinema_id)
            if city and f.avg_price > 0:
                self._price_by_city[city].append(f.avg_price)
        self._occ_baseline.freeze()

    def _city(self, cinema_id: str) -> str | None:
        c = self.ds.cinemas.get(cinema_id)
        return c.city if c else None

    # ------------------------------------------------------------------
    def run(self) -> DetectorResult:
        self.prepare()
        cfg = self.cfg
        res = DetectorResult()
        # 城市低价参考
        city_low = {c: _pct(v, 0.15) for c, v in self._price_by_city.items()}

        for sid, f in self.f.items():
            st = self.ds.showtimes[sid]
            ctx = self._contexts[sid]
            if f.planned_seats < cfg.min_tickets_showtime and f.sold_tickets < cfg.min_tickets_showtime:
                continue
            evs = self._evaluate(f, ctx, city_low)
            if not evs:
                continue
            finding = Finding(
                finding_type=FindingType.FAKE,
                scope="showtime",
                object_id=sid,
                date=st.start_time.date(),
                cinema_id=f.cinema_id,
                film_id=f.film_id,
                showtime_id=sid,
            )
            for e in evs:
                finding.add(e)
            for m in _mitigator_flags(f, ctx, cfg):
                finding.add_mitigator(m)
            finding.metrics = _metrics(f, ctx)
            finding.finalize(cfg)
            # 只有支撑证据时不进入清单
            if not finding.primary_evidence:
                continue
            res.findings.append(finding)

        res.findings.sort(key=lambda x: x.score, reverse=True)
        if len(res.findings) > cfg.max_findings_showtime:
            res.notes.append(
                f"可疑场次超过上限 {cfg.max_findings_showtime}，仅输出前 N 条"
            )
            res.findings = res.findings[: cfg.max_findings_showtime]
        return res

    # ------------------------------------------------------------------
    def _evaluate(
        self, f: ShowtimeFeatures, ctx: ShowContext, city_low: dict[str, float]
    ) -> list[Evidence]:
        cfg = self.cfg
        evs: list[Evidence] = []

        # 是否"异常上座"：夜间场次用分层基线；白天场次用绝对高阈值
        occupancy_anomalous = False
        if ctx.late_night and self._occ_baseline is not None:
            z = self._occ_baseline.score(ctx.baseline_key, f.occupancy, cfg.mad_floor)
            if z >= cfg.late_night_occupancy_z and f.sold_tickets >= cfg.min_tickets_showtime:
                occupancy_anomalous = True
                base_center, _base_scale, base_n = self._occ_baseline.center(ctx.baseline_key)
                evs.append(_ev(
                    "F01", f.occupancy, round(base_center, 4),
                    f"{f.start_time:%H:%M} 深夜场上座率 {f.occupancy:.0%}"
                    f"（{f.sold_tickets}/{f.planned_seats} 座），"
                    f"同城同档同时段正常水平为 {base_center:.0%}"
                    f"（基于 {base_n} 个历史场次）",
                    z=z, baseline_occupancy=round(base_center, 4),
                    day_type=ctx.day_type.value, bucket=ctx.bucket,
                ))
        # 高上座只作支撑证据：满座在正常经营中很常见（包场/热门场次）。
        if f.occupancy >= cfg.high_occupancy and f.sold_tickets >= cfg.min_tickets_showtime:
            evs.append(_ev(
                "F02", f.occupancy, cfg.high_occupancy,
                f"上座率 {f.occupancy:.0%}（{f.sold_tickets}/{f.planned_seats}），接近满座",
            ))

        # 非深夜场：高上座 + 批量购票行为本身可构成主证据（"黄金场异常满座+机器下单"）
        if not ctx.late_night and f.occupancy >= cfg.high_occupancy:
            if _batch_purchase_pattern(f, cfg):
                occupancy_anomalous = True

        if not occupancy_anomalous and f.occupancy < 0.5:
            # 上座很低时，除锁场退票外其余刷票规则不适用
            pass
        else:
            # 设备/账号集中（排除团体渠道：包场天然是一台设备）
            dev_share, dev_n, dev_tickets = _device_concentration(f)
            if (dev_share >= cfg.device_concentration
                    and dev_tickets >= cfg.device_min_distinct):
                evs.append(_ev(
                    "F03", dev_share, cfg.device_concentration,
                    f"{dev_tickets} 张散客票中 {dev_share:.0%} 来自同一设备"
                    f"（共 {dev_n} 个设备下单）",
                    devices=dev_n,
                ))
            elif (f.top_account_share >= cfg.device_concentration
                  and f.sold_tickets >= cfg.device_min_distinct):
                evs.append(_ev(
                    "F03", f.top_account_share, cfg.device_concentration,
                    f"{f.sold_tickets} 张票中 {f.top_account_share:.0%} 来自同一账号",
                ))

            # 售票时间高度集中
            if (f.burst_share >= cfg.sale_burst_share
                    and f.burst_tickets >= cfg.sale_burst_min_tickets):
                evs.append(_ev(
                    "F04", f.burst_share, cfg.sale_burst_share,
                    f"{f.burst_tickets}/{f.sold_tickets} 张票在开场前 10 分钟内集中售出"
                    f"（{f.burst_share:.0%}）",
                ))

        # 低价票
        low = city_low.get(self._city(f.cinema_id) or "", 0.0)
        if (low > 0 and f.avg_price > 0
                and f.avg_price <= low * cfg.low_price_ratio
                and f.sold_tickets >= cfg.low_price_min_tickets):
            evs.append(_ev(
                "F05", f.avg_price, round(low * cfg.low_price_ratio, 2),
                f"均价 {f.avg_price:.1f} 元，仅为同城均价低分位（{low:.1f} 元）的 "
                f"{f.avg_price / low:.0%}",
                city_ref_price=round(low, 2),
            ))

        # 非会员匿名
        if (f.nonmember_ratio >= cfg.nonmember_ratio
                and f.sold_tickets >= cfg.nonmember_min_tickets
                and f.occupancy >= 0.5):
            evs.append(_ev(
                "F06", f.nonmember_ratio, cfg.nonmember_ratio,
                f"{f.sold_tickets} 张票中 {f.nonmember_ratio:.0%} 为非会员购票",
            ))

        # 锁场后退票：退票率高，或"峰值售出"与"净票房"严重背离
        net_vs_peak_drop = (f.peak_tickets - f.sold_tickets)
        refund_signal = (
            f.refund_rate >= cfg.refund_rate_high
            or (f.peak_tickets >= cfg.refund_min_tickets and net_vs_peak_drop >= 0.5 * f.peak_tickets)
        )
        if refund_signal and f.peak_tickets >= cfg.refund_min_tickets:
            evs.append(_ev(
                "F07", f.refund_rate, cfg.refund_rate_high,
                f"峰值售出 {f.peak_tickets} 张，开场前退票 {f.refunded_tickets} 张"
                f"（退票率 {f.refund_rate:.0%}），净票房仅剩 {f.gross:.0f} 元",
            ))

        # 整排连座
        if (f.contiguous_ratio >= cfg.contiguous_ratio
                and f.sold_tickets >= cfg.device_min_tickets):
            evs.append(_ev(
                "F08", f.contiguous_ratio, cfg.contiguous_ratio,
                f"连座大额订单占比 {f.contiguous_ratio:.0%}，呈现批量占座",
            ))

        return evs


def _device_concentration(f: ShowtimeFeatures) -> tuple[float, int, int]:
    """散客（排除团体/包场）设备集中度：(最大占比, 设备数, 散客票数)。"""
    if f.retail_device_counts and f.retail_tickets:
        return f.retail_top_device_share, f.retail_device_count, f.retail_tickets
    return f.top_device_share, f.device_count, f.sold_tickets


def _batch_purchase_pattern(f: ShowtimeFeatures, cfg: DetectorConfig) -> bool:
    """非深夜高上座场次里，是否存在"机器批量下单"的强特征组合。

    只有设备集中 + 时间集中同时出现，才把高上座从"支撑证据"升级为主证据；
    单看其中一项，包场和热门场次都可能满足。
    """
    dev_share, _n, dev_tickets = _device_concentration(f)
    dev = (dev_share >= cfg.device_concentration and dev_tickets >= cfg.device_min_distinct)
    burst = (f.burst_share >= cfg.sale_burst_share
             and f.burst_tickets >= cfg.sale_burst_min_tickets)
    return dev and burst


def _pct(xs: Sequence[float], q: float) -> float:
    from ..baseline import _percentile
    return _percentile(xs, q)


def _metrics(f: ShowtimeFeatures, ctx: ShowContext) -> dict:
    return {
        "start_time": f.start_time.isoformat(),
        "occupancy": round(f.occupancy, 4),
        "peak_occupancy": round(f.peak_occupancy, 4),
        "sold_tickets": f.sold_tickets,
        "planned_seats": f.planned_seats,
        "gross": f.gross,
        "avg_price": f.avg_price,
        "refund_rate": round(f.refund_rate, 4),
        "device_count": f.device_count,
        "top_device_share": round(f.top_device_share, 4),
        "online_ratio": round(f.online_ratio, 4),
        "group_ratio": round(f.group_ratio, 4),
        "day_type": ctx.day_type.value,
        "late_night": ctx.late_night,
        "film_stage": ctx.film_stage.value,
        "premiere_midnight": ctx.premiere_midnight,
        "hall_tier": ctx.hall_tier,
    }
