"""偷换票房检测（D 类）：卖 A 记 B、截留票款、结构票价。

三类信号：
1) 一致性问题：同一场次排片影片 vs 结算上报影片不一致（D01）；
2) 横向异常：冷门片场均奇高（D02）、热门片份额被压低（D03）—— 以
   "同城同档 + 同片"为对照，而非全国，避免一线/三四线差异误报；
3) 渠道/票价缺口：线上销量 > 上报票量（D04）、上报票价 < 实付（D05）、
   柜台占比异常（D06）。

输出粒度：
- D01/D04/D05 为场次级 -> 可疑场次清单；
- D02/D03/D06 为"影院-影片-日"或"影院-日"级 -> 可疑影院清单。
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..baseline import BaselineBook, median
from ..config import (
    CHANNEL_COUNTER, CHANNEL_ONLINE, DetectorConfig, DIVERSION_RULES, MITIGATORS,
)
from ..evidence import DetectorResult, Evidence, Finding, FindingType
from ..features import ShowtimeFeatures
from ..store import Dataset, Settlement, order_is_refunded

_RULE = {r.code: r for r in DIVERSION_RULES}
_MIT = {r.code: r for r in MITIGATORS}


def _ev(code: str, value: float, threshold: float | None, message: str, **extra) -> Evidence:
    r = _RULE[code]
    return Evidence(r.code, r.label, r.weight, value, threshold, message, r.action,
                    extra, primary=r.primary)


@dataclass
class FilmCinemaDay:
    cinema_id: str
    film_id: str
    date: dt.date
    showtimes: int = 0
    tickets: int = 0
    gross: float = 0.0
    planned: int = 0
    reported_tickets: int = 0
    reported_gross: float = 0.0
    reported_price: float | None = None
    _price_weight: float = 0.0
    filed_film_mismatch: int = 0
    # 由外部填充
    city: str = ""
    heat_pct: float | None = None      # 全国热度分位（0-1）
    city_share: float | None = None    # 该片在该影院的票房份额
    peer_city_share: float | None = None
    attendance: float = 0.0
    peer_attendance: float | None = None

    @property
    def occupancy(self) -> float:
        return self.tickets / self.planned if self.planned else 0.0

    @property
    def avg_price(self) -> float:
        return self.gross / self.tickets if self.tickets else 0.0

    @property
    def avg_showtime_tickets(self) -> float:
        return self.tickets / self.showtimes if self.showtimes else 0.0


@dataclass
class CinemaDay:
    cinema_id: str
    date: dt.date
    tickets: int = 0
    gross: float = 0.0
    online_tickets: int = 0
    counter_tickets: int = 0
    group_tickets: int = 0
    showtimes: int = 0
    reported_tickets: int = 0
    reported_gross: float = 0.0
    city: str = ""

    @property
    def counter_ratio(self) -> float:
        return self.counter_tickets / self.tickets if self.tickets else 0.0

    @property
    def online_ratio(self) -> float:
        return self.online_tickets / self.tickets if self.tickets else 0.0


class DiversionDetector:
    def __init__(self, ds: Dataset, features: dict[str, ShowtimeFeatures], cfg: DetectorConfig):
        self.ds = ds
        self.f = features
        self.cfg = cfg
        self.film_days: dict[tuple[str, str, dt.date], FilmCinemaDay] = {}
        self.cinema_days: dict[tuple[str, dt.date], CinemaDay] = {}
        self._film_city_peers: dict[tuple[str, str], list[FilmCinemaDay]] = defaultdict(list)
        self._city_share_median: dict[tuple[str, str], float] = {}
        self._film_heat: dict[str, float] = {}

    # ------------------------------------------------------------------
    def prepare(self) -> None:
        self._aggregate()
        self._compute_heat()
        self._compute_peers()

    def _aggregate(self) -> None:
        # 场次 -> 影片/影院/日
        for sid, f in self.f.items():
            d = f.start_time.date()
            key = (f.cinema_id, f.film_id, d)
            fd = self.film_days.get(key)
            if fd is None:
                city = self.ds.cinemas[f.cinema_id].city if f.cinema_id in self.ds.cinemas else ""
                fd = FilmCinemaDay(f.cinema_id, f.film_id, d, city=city)
                self.film_days[key] = fd
            fd.showtimes += 1
            fd.tickets += f.sold_tickets
            fd.gross += f.gross
            fd.planned += f.planned_seats

            ck = (f.cinema_id, d)
            cd = self.cinema_days.get(ck)
            if cd is None:
                city = self.ds.cinemas[f.cinema_id].city if f.cinema_id in self.ds.cinemas else ""
                cd = CinemaDay(f.cinema_id, d, city=city)
                self.cinema_days[ck] = cd
            cd.tickets += f.sold_tickets
            cd.gross += f.gross
            cd.online_tickets += f.online_tickets
            cd.counter_tickets += f.counter_tickets
            cd.group_tickets += f.group_tickets
            cd.showtimes += 1

        # 结算上报
        for sid, settles in self.ds.settlements_by_showtime.items():
            st = self.ds.showtimes.get(sid)
            if not st:
                continue
            d = st.start_time.date()
            for s in settles:
                filed = s.filed_film_id or st.film_id
                key = (st.cinema_id, filed, d)
                fd = self.film_days.get(key)
                if fd is None:
                    city = self.ds.cinemas[st.cinema_id].city if st.cinema_id in self.ds.cinemas else ""
                    fd = FilmCinemaDay(st.cinema_id, filed, d, city=city)
                    self.film_days[key] = fd
                fd.reported_tickets += s.reported_tickets or 0
                fd.reported_gross += s.reported_gross or 0.0
                # 上报均价按票数加权，避免被单条零星记录带偏
                if s.reported_price and (s.reported_tickets or 0) > 0:
                    w = float(s.reported_tickets)
                    prev = fd.reported_price or 0.0
                    fd.reported_price = (prev * fd._price_weight + s.reported_price * w) / (
                        fd._price_weight + w
                    )
                    fd._price_weight += w
                if filed != st.film_id:
                    fd.filed_film_mismatch += 1
                ck = (st.cinema_id, d)
                cd = self.cinema_days.get(ck)
                if cd:
                    cd.reported_tickets += s.reported_tickets or 0
                    cd.reported_gross += s.reported_gross or 0.0

    def _compute_heat(self) -> None:
        """全国热度：按影片全国总票房分位，用于界定冷门/热门。"""
        national: dict[str, float] = defaultdict(float)
        for (cid, fid, d), fd in self.film_days.items():
            national[fid] += fd.gross
        if not national:
            return
        ordered = sorted(national.items(), key=lambda kv: kv[1])
        n = len(ordered)
        for i, (fid, _) in enumerate(ordered):
            self._film_heat[fid] = (i + 1) / n if n else 0.5

    def _compute_peers(self) -> None:
        city_film_share: dict[tuple[str, str], list[float]] = defaultdict(list)
        city_film_att: dict[tuple[str, str], list[float]] = defaultdict(list)
        city_film_occ: dict[tuple[str, str], list[float]] = defaultdict(list)
        # 影院日总票房
        cinema_day_gross: dict[tuple[str, dt.date], float] = {
            k: cd.gross for k, cd in self.cinema_days.items()
        }
        for (cid, fid, d), fd in self.film_days.items():
            fd.heat_pct = self._film_heat.get(fid)
            total = cinema_day_gross.get((cid, d), 0.0)
            fd.city_share = fd.gross / total if total > 0 else 0.0
            fd.attendance = fd.tickets / fd.showtimes if fd.showtimes else 0.0
            key = (fd.city, fid)
            city_film_share[key].append(fd.city_share)
            if fd.showtimes > 0:
                city_film_att[key].append(fd.attendance)
                city_film_occ[key].append(fd.occupancy)
            self._film_city_peers[key].append(fd)
        self._share_median = {k: median(v) for k, v in city_film_share.items()}
        self._att_median = {k: median(v) for k, v in city_film_att.items()}
        self._occ_median = {k: median(v) for k, v in city_film_occ.items()}

    # ------------------------------------------------------------------
    def run(self) -> DetectorResult:
        self.prepare()
        res = DetectorResult()
        res.extend(self._detect_mismatch())      # D01 场次级
        res.extend(self._detect_film_anomaly())  # D02/D03 影院-影片-日
        res.extend(self._detect_online_gap())    # D04 场次级
        res.extend(self._detect_price_gap())     # D05 场次级
        res.extend(self._detect_counter())       # D06 影院-日
        res.findings.sort(key=lambda x: x.score, reverse=True)
        return res

    # -- D01 排片与上报影片不一致 --------------------------------------
    def _detect_mismatch(self) -> DetectorResult:
        res = DetectorResult()
        for sid, settles in self.ds.settlements_by_showtime.items():
            st = self.ds.showtimes.get(sid)
            if not st:
                continue
            bad = [s for s in settles if s.filed_film_id and s.filed_film_id != st.film_id]
            if not bad:
                continue
            f = self.f.get(sid)
            if not f:
                continue
            filed = bad[0].filed_film_id
            filed_title = self.ds.films[filed].title if filed in self.ds.films else filed
            shown_title = self.ds.films[st.film_id].title if st.film_id in self.ds.films else st.film_id
            finding = Finding(
                FindingType.DIVERSION, "showtime", sid, st.start_time.date(),
                st.cinema_id, st.film_id, sid,
            )
            finding.add(_ev(
                "D01", len(bad), 1,
                f"场次 {sid} 排片放映《{shown_title}》，结算却记账到《{filed_title}》"
                f"（{len(bad)} 条上报记录影片不符）",
                shown_film=st.film_id, filed_film=filed,
                mismatched_settlements=len(bad),
            ))
            finding.metrics = {
                "start_time": st.start_time.isoformat(),
                "shown_film": shown_title,
                "filed_film": filed_title,
                "sold_tickets": f.sold_tickets,
                "gross": f.gross,
            }
            finding.finalize(self.cfg)
            res.findings.append(finding)
        return res

    # -- D02 冷门片场均异常 --------------------------------------------
    def _detect_film_anomaly(self) -> DetectorResult:
        res = DetectorResult()
        cfg = self.cfg
        # 同城同片当日"异常影院数"：若全城多家一起异常，更可能是影片/排片本身
        # 的问题（或全国性数据口径问题），而非单家影院偷换。此时降权。
        city_film_anom_cinemas: dict[tuple[str, str, dt.date], int] = defaultdict(int)
        for (cid2, fid2, d2), fd2 in self.film_days.items():
            peer = self._att_median.get((fd2.city, fid2))
            if (peer and peer > 0 and fd2.attendance >= peer * cfg.attendance_ratio_high
                    and fd2.tickets >= 5):
                city_film_anom_cinemas[(fd2.city, fid2, d2)] += 1
        city_total_cinemas: dict[str, int] = defaultdict(int)
        for c in self.ds.cinemas.values():
            city_total_cinemas[c.city] += 1

        for (cid, fid, d), fd in self.film_days.items():
            if fd.showtimes < cfg.min_showtimes_film_day or fd.tickets < 5:
                continue
            # 只在结算上报存在时评判（否则不是偷换问题）
            if fd.reported_tickets == 0 and fd.reported_gross == 0:
                continue
            heat = fd.heat_pct if fd.heat_pct is not None else 0.5
            peer_att = self._att_median.get((fd.city, fid))
            evs: list[Evidence] = []

            # 同城同片同日异常影院数（含自己）占该城影院数比例过高 -> 影片级问题
            n_anom = city_film_anom_cinemas.get((fd.city, fid, d), 0)
            n_city = max(1, city_total_cinemas.get(fd.city, 1))
            film_level_problem = n_anom >= max(3, n_city * 0.5)

            # D02 冷门片场均异常：用上座率（座位归一）而非绝对人次
            peer_occ = self._occ_median.get((fd.city, fid))
            d02_hit = (
                heat <= cfg.cold_film_heat_pct
                and peer_occ is not None and peer_occ > 0
                and fd.occupancy >= max(peer_occ * cfg.occupancy_ratio_high,
                                        cfg.cold_film_occupancy_floor)
                and fd.tickets >= 10
            )
            if d02_hit:
                evs.append(_ev(
                    "D02", fd.occupancy, round(peer_occ * cfg.occupancy_ratio_high, 4),
                    f"冷门片《{self._title(fid)}》（全国热度后 {heat:.0%}）"
                    f"在该影院上座率 {fd.occupancy:.0%}，同城同片中位数仅 {peer_occ:.0%}"
                    f"（{fd.occupancy / peer_occ:.1f} 倍），票房规模与影片热度明显不符",
                    heat_pct=round(heat, 3), peer_occupancy=round(peer_occ, 4),
                    attendance=round(fd.attendance, 2),
                    peer_attendance=round(peer_att, 2) if peer_att else None,
                ))

            # D03 热门片份额被压低
            peer_share = self._share_median.get((fd.city, fid))
            if (heat >= cfg.hot_film_heat_pct and peer_share is not None
                    and peer_share >= 0.05
                    and fd.city_share is not None
                    and fd.city_share <= peer_share - cfg.share_gap):
                evs.append(_ev(
                    "D03", fd.city_share, round(peer_share - cfg.share_gap, 4),
                    f"热门片《{self._title(fid)}》在该影院票房份额仅 {fd.city_share:.0%}，"
                    f"同城同片中位数为 {peer_share:.0%}（差 {peer_share - fd.city_share:.0%}）",
                    peer_share=round(peer_share, 4), heat_pct=round(heat, 3),
                ))

            if not evs:
                continue
            finding = Finding(
                FindingType.DIVERSION, "cinema_film_day",
                f"{cid}|{fid}|{d.isoformat()}", d, cid, fid,
            )
            for e in evs:
                finding.add(e)
            # 节假日为该片带来正常热度 -> D02 降权
            if self.ds.calendar.is_holiday(d):
                finding.add_mitigator("M01")
            # 全城多家影院同片一起异常：更可能是影片/口径问题，不是本影院偷换
            if any(e.rule == "D02" for e in evs) and film_level_problem:
                finding.add_mitigator("M05")
            finding.metrics = {
                "showtimes": fd.showtimes,
                "tickets": fd.tickets,
                "gross": round(fd.gross, 2),
                "attendance": round(fd.attendance, 2),
                "reported_tickets": fd.reported_tickets,
                "reported_gross": round(fd.reported_gross, 2),
                "city_share": round(fd.city_share or 0, 4),
                "heat_pct": round(heat, 3),
                "film_level_problem": film_level_problem,
                "city_anomalous_cinemas": n_anom,
            }
            finding.finalize(cfg)
            if not finding.primary_evidence:
                continue
            res.findings.append(finding)
        return res

    # -- D04 线上销量 vs 上报票房缺口 ----------------------------------
    def _detect_online_gap(self) -> DetectorResult:
        """影院-影片-日 级：线上平台售票量 vs 影院上报票量。

        偷换票房的一种常见形态：观众在线上买了 A 片，影院却把钱记到 B 片
        或干脆少报。按影片对账能直接锁定被"吞掉"的影片；再以影院-日
        总账做兜底，防止把票记到别的影片名下导致按片对账漏检。
        """
        res = DetectorResult()
        cfg = self.cfg
        online_by_day: dict[tuple[str, dt.date], int] = defaultdict(int)
        online_by_film: dict[tuple[str, str, dt.date], int] = defaultdict(int)
        for o in self.ds.online_sales:
            online_by_day[(o.cinema_id, o.sale_date)] += o.online_tickets
            online_by_film[(o.cinema_id, o.film_id, o.sale_date)] += o.online_tickets

        def _emit(cid, d, film_id, on_tickets, rep_tickets, scope, note=""):
            gap = on_tickets - rep_tickets
            ratio = gap / max(on_tickets, 1)
            if on_tickets < cfg.online_gap_min_tickets or ratio < cfg.online_gap_ratio:
                return
            title = self._title(film_id) if film_id else ""
            subject = f"《{title}》" if title else "全影院"
            finding = Finding(
                FindingType.DIVERSION, scope,
                f"{cid}|{film_id or ''}|{d.isoformat()}", d, cid, film_id,
            )
            finding.add(_ev(
                "D04", ratio, cfg.online_gap_ratio,
                f"{subject}当日线上售出 {on_tickets} 张，影院上报仅 {rep_tickets} 张，"
                f"缺口 {gap} 张（{ratio:.0%}）{note}",
                online_tickets=on_tickets, reported_tickets=rep_tickets, gap=gap,
            ))
            finding.metrics = {
                "online_tickets": on_tickets,
                "reported_tickets": rep_tickets,
                "gap": gap,
            }
            if self.ds.calendar.is_holiday(d):
                finding.add_mitigator("M01")
            finding.finalize(cfg)
            res.findings.append(finding)

        # 1) 按影片对账：直接锁定被少报的影片
        for (cid, fid, d), on_tickets in online_by_film.items():
            fd = self.film_days.get((cid, fid, d))
            rep = fd.reported_tickets if fd else 0
            _emit(cid, d, fid, on_tickets, rep, "cinema_film_day")

        # 2) 影院-日总账兜底（票被记到别的影片名下时，按片对账会漏）
        for (cid, d), on_tickets in online_by_day.items():
            cd = self.cinema_days.get((cid, d))
            if cd is None or cd.reported_tickets <= 0:
                continue
            _emit(cid, d, None, on_tickets, cd.reported_tickets, "cinema_day")
        return res

    # -- D05 票价结构异常 ----------------------------------------------
    def _detect_price_gap(self) -> DetectorResult:
        """场次级：该场上报结算价 vs 观众实付均价。

        用场次级而非影片日汇总，是因为结构票价通常只发生在部分场次，
        日汇总会把它稀释掉。结算表本就按场次上报，天然支持场次级判定。
        """
        res = DetectorResult()
        cfg = self.cfg
        for sid, settles in self.ds.settlements_by_showtime.items():
            st = self.ds.showtimes.get(sid)
            f = self.f.get(sid)
            if not st or not f or f.sold_tickets < cfg.price_gap_min_tickets:
                continue
            if f.avg_price <= 0:
                continue
            # 该场上报均价（按票数加权）
            rep_t = 0
            rep_amt = 0.0
            for s in settles:
                if s.reported_price and (s.reported_tickets or 0) > 0:
                    rep_t += s.reported_tickets
                    rep_amt += s.reported_price * s.reported_tickets
            if rep_t <= 0:
                continue
            rep_price = rep_amt / rep_t
            ratio = rep_price / f.avg_price
            if ratio > 0.85:
                continue
            finding = Finding(
                FindingType.DIVERSION, "showtime", sid, st.start_time.date(),
                st.cinema_id, st.film_id, sid,
            )
            finding.add(_ev(
                "D05", ratio, 0.85,
                f"观众实付均价 {f.avg_price:.1f} 元，该场上报结算均价仅 {rep_price:.1f} 元"
                f"（{ratio:.0%}），存在结构票价/阴阳票价",
                reported_price=round(rep_price, 2),
                paid_price=round(f.avg_price, 2),
            ))
            finding.metrics = {
                "start_time": st.start_time.isoformat(),
                "paid_avg_price": round(f.avg_price, 2),
                "reported_price": round(rep_price, 2),
                "sold_tickets": f.sold_tickets,
            }
            finding.finalize(cfg)
            res.findings.append(finding)
        return res

    # -- D06 柜台渠道占比异常 ------------------------------------------
    def _detect_counter(self) -> DetectorResult:
        res = DetectorResult()
        cfg = self.cfg
        city_ratio: dict[str, list[float]] = defaultdict(list)
        for cd in self.cinema_days.values():
            if cd.tickets > 0:
                city_ratio[cd.city].append(cd.counter_ratio)
        city_med = {c: median(v) for c, v in city_ratio.items()}
        for (cid, d), cd in self.cinema_days.items():
            if cd.tickets < 50:
                continue
            med = city_med.get(cd.city, 0.0)
            if (cd.counter_ratio >= cfg.counter_ratio_high
                    and cd.counter_ratio >= med + cfg.share_gap):
                finding = Finding(
                    FindingType.DIVERSION, "cinema_day",
                    f"{cid}|{d.isoformat()}", d, cid,
                )
                finding.add(_ev(
                    "D06", cd.counter_ratio, cfg.counter_ratio_high,
                    f"影院柜台/自有系统售票占比 {cd.counter_ratio:.0%}"
                    f"（{cd.counter_tickets}/{cd.tickets} 张），同城中位数为 {med:.0%}",
                    city_median=round(med, 4),
                ))
                finding.metrics = {
                    "tickets": cd.tickets,
                    "gross": round(cd.gross, 2),
                    "counter_ratio": round(cd.counter_ratio, 4),
                    "online_ratio": round(cd.online_ratio, 4),
                    "showtimes": cd.showtimes,
                }
                if self.ds.calendar.is_holiday(d):
                    finding.add_mitigator("M01")
                finding.finalize(cfg)
                res.findings.append(finding)
        return res

    # ------------------------------------------------------------------
    def _title(self, fid: str) -> str:
        return self.ds.films[fid].title if fid in self.ds.films else fid
