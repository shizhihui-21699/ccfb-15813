"""全局配置：时段定义、阈值、规则权重。

监管方调参集中在此文件。所有阈值都有注释说明业务含义，
并区分"绝对阈值"和"相对基线阈值"，避免节假日/首映误报。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet


# ---------------------------------------------------------------------------
# 时段定义
# ---------------------------------------------------------------------------
# 深夜/凌晨场：23:00 - 06:00。这是幽灵场、刷票房的高发时段，
# 但首映零点场也落在此区间，需要单独区分（见 calendar_ctx 的 premiere 逻辑）。
LATE_NIGHT_START_HOUR = 23
LATE_NIGHT_END_HOUR = 6


def is_late_night(hour: int) -> bool:
    return hour >= LATE_NIGHT_START_HOUR or hour < LATE_NIGHT_END_HOUR


# ---------------------------------------------------------------------------
# 渠道
# ---------------------------------------------------------------------------
CHANNEL_ONLINE = "online"      # 猫眼/淘票票等第三方平台（相对可信的对照源）
CHANNEL_COUNTER = "counter"    # 影院柜台（偷换票房高发）
CHANNEL_SELF = "self"          # 影院自助机
CHANNEL_GROUP = "group"        # 团体/包场

TRUSTED_CHANNELS: FrozenSet[str] = frozenset({CHANNEL_ONLINE})
# 团体/包场渠道不计入"设备集中度"——包场天然是一个买家一台设备。
EXCLUDED_FROM_DEVICE_CONCENTRATION: FrozenSet[str] = frozenset({CHANNEL_GROUP})


# ---------------------------------------------------------------------------
# 规则规格
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuleSpec:
    code: str
    label: str
    weight: float          # 命中后对可疑度的贡献
    description: str
    action: str = ""       # 命中该规则时给核查人员的建议
    # 主证据 vs 支撑证据。
    # 主证据可以单独支撑一条发现；支撑证据只能加分，不能单独定罪。
    # 否则"满座""单台设备下单多"这类正常现象会把清单淹没。
    primary: bool = True


# 虚假票房（刷票房/幽灵场/锁场）
FAKE_RULES: tuple[RuleSpec, ...] = (
    RuleSpec(
        "F01", "深夜场次异常上座", 0.35,
        "深夜/凌晨场次上座率显著高于同层影院历史基线",
        "调取该场次监控与入场记录，核对实际观影人数",
        primary=True,
    ),
    RuleSpec(
        "F02", "满座异常", 0.15,
        "场次上座率接近 100%（需其他证据配合）",
        "核对座位销售明细与现场监控",
        primary=False,
    ),
    RuleSpec(
        "F03", "设备/账号高度集中", 0.20,
        "少数设备或账号购买了该场次大部分票（已排除团体渠道）",
        "调取购票设备/IP/支付账号，核查是否为片方或影院自购",
        primary=False,
    ),
    RuleSpec(
        "F04", "售票时间高度集中", 0.25,
        "绝大多数票在极短时间窗口内售出（机器批量下单特征）",
        "核对下单时间序列与订单来源",
        primary=True,
    ),
    RuleSpec(
        "F05", "异常低价票", 0.35,
        "平均票价远低于同城最低结算价，存在 1 元票/低价票刷量",
        "核对票面价格与结算价，追查低价票来源",
        primary=True,
    ),
    RuleSpec(
        "F06", "非会员匿名购票", 0.10,
        "高上座场次几乎全部为非会员、无实名信息购票",
        "核查购票人身份与联系方式真实性",
        primary=False,
    ),
    RuleSpec(
        "F07", "锁场后退票", 0.40,
        "开场前集中购票占座、临近开场大量退票，净票房与峰值票房背离",
        "对比峰值售票与退票记录，核查退票资金流向",
        primary=True,
    ),
    RuleSpec(
        "F08", "整排连座购买", 0.08,
        "单笔订单购买整排/大量连座，呈现批量占座特征",
        "核对订单明细与购票人关系",
        primary=False,
    ),
)

# 偷换票房（卖 A 记 B / 截留票款）
DIVERSION_RULES: tuple[RuleSpec, ...] = (
    RuleSpec(
        "D01", "排片与售票影片不一致", 0.60,
        "同一影厅同一场次，排片系统记录影片与售票/结算系统上报影片不一致",
        "调取该场次排片单、票根与结算记录，核实实际放映影片",
        primary=True,
    ),
    RuleSpec(
        "D02", "冷门片场均异常", 0.40,
        "全国场均极低的冷门片在某影院场均人次远超同城同片",
        "核查该片实际排片、上座与观众票根",
        primary=True,
    ),
    RuleSpec(
        "D03", "热门片票房份额被压低", 0.25,
        "热门片在该影院的票房份额显著低于同城同档影院",
        "对比同城同档影院该片份额，核查是否存在转移",
        primary=False,
    ),
    RuleSpec(
        "D04", "线上销量与上报票房缺口", 0.45,
        "第三方线上平台售出票量显著高于影院上报的对应场次票量",
        "比对平台订单与影院上报流水，追查未入系统票款",
        primary=True,
    ),
    RuleSpec(
        "D05", "票价结构异常", 0.35,
        "上报结算票价显著低于观众实际支付价（结构票价/阴阳票价）",
        "核对观众支付凭证与上报票价",
        primary=True,
    ),
    RuleSpec(
        "D06", "柜台渠道占比异常", 0.12,
        "影院柜台/自有系统售票占比远高于同城同档影院，偷换高发渠道",
        "重点核查柜台现金与手工票",
        primary=False,
    ),
)

# 误报抑制（负权重 / 降权标记）
MITIGATORS: tuple[RuleSpec, ...] = (
    RuleSpec(
        "M01", "节假日基线调整", 0.0,
        "当日为法定节假日/档期，已使用节假日专用基线对照",
    ),
    RuleSpec(
        "M02", "首映零点场", 0.0,
        "首映日零点场属于正常观影高峰，已用首映专用基线对照",
    ),
    RuleSpec(
        "M03", "疑似正常包场/团购", 0.0,
        "整场团体购买、提前预订、价格折扣，符合包场/团购特征，降权处理",
    ),
    RuleSpec(
        "M04", "热门特效厅", 0.0,
        "IMAX/杜比等特效厅上座率天然偏高，已按影厅类型分层",
    ),
    RuleSpec(
        "M05", "影片级异常（非影院个例）", 0.0,
        "同城多家影院同一影片同时异常，更可能是影片热度或数据口径问题，降权处理",
    ),
)


@dataclass
class DetectorConfig:
    """检测器可调参数。"""

    # --- 样本量下限（低于此不判定，避免小样本噪声） ---
    min_tickets_showtime: int = 5
    min_showtimes_cinema_day: int = 12
    min_gross_cinema_day: float = 2000.0

    # --- 场次级 ---
    late_night_occupancy_z: float = 2.5      # 深夜场上座率稳健 z 阈值
    high_occupancy: float = 0.85             # "满座"绝对阈值
    device_concentration: float = 0.50       # 单设备购票占比
    device_min_tickets: int = 8
    device_min_distinct: int = 15            # 参与设备集中度判定的最少票数
    sale_burst_share: float = 0.75           # 前 10 分钟售票占比
    sale_burst_min_tickets: int = 12
    low_price_ratio: float = 0.60            # 均价 / 同城低价参考
    low_price_min_tickets: int = 10
    nonmember_ratio: float = 0.90
    nonmember_min_tickets: int = 20
    refund_rate_high: float = 0.30
    refund_min_tickets: int = 10
    contiguous_ratio: float = 0.80

    # --- 影片-影院-日 ---
    share_gap: float = 0.15                  # 份额背离阈值
    attendance_ratio_high: float = 3.0       # 影院该片场均 / 同城同片场均
    # 座位归一化后的上座率比较（避免"大厅人多"被误判）。冷门片在同城
    # 同片上座率通常很低，若某影院高出数倍，才说明票房被转移过来。
    occupancy_ratio_high: float = 3.0
    cold_film_occupancy_floor: float = 0.45
    cold_film_heat_pct: float = 0.20         # 全国热度后 20% 视为冷门
    hot_film_heat_pct: float = 0.80
    min_showtimes_film_day: int = 2

    # --- 偷换 ---
    min_mismatch_showtimes: int = 2
    online_gap_ratio: float = 0.20
    online_gap_min_tickets: int = 30
    counter_ratio_high: float = 0.60
    price_gap_min_tickets: int = 50          # D05 样本下限，避免小样本票价噪声

    # --- 包场/团购识别（用于降权） ---
    group_ratio: float = 0.60                # 团体渠道占比
    group_single_order_share: float = 0.70   # 单笔订单占场次票数
    group_presale_hours: float = 12.0        # 提前预订小时数
    group_price_discount: float = 0.25       # 相对同片均价折扣

    # --- 基线 ---
    baseline_window_days: int = 28
    baseline_min_samples: int = 20
    mad_floor: float = 1e-6

    # --- 评分 ---
    score_threshold_high: float = 0.70
    score_threshold_medium: float = 0.45
    score_threshold_low: float = 0.30
    max_findings_showtime: int = 2000

    # --- 权重缩放 ---
    fake_weight_scale: float = 1.0
    diversion_weight_scale: float = 1.0


DEFAULT_CONFIG = DetectorConfig()


# 影厅格式（特效厅分层用）
PREMIUM_FORMATS: FrozenSet[str] = frozenset({"IMAX", "DOLBY", "CINITY", "4DX"})
