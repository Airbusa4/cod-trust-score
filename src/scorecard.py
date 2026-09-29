"""Rule-based COD Trust Scorecard (Phase 1).

Deterministic and explainable: every point added or removed is recorded
as a reason so the decision can be shown to judges, ops teams and users.
"""
from dataclasses import dataclass, field

# Tier policy: minimum score -> COD limit (THB) and allowed refusals per 90 days
TIERS = [
    # (tier, min_score, cod_limit_thb, refusals_allowed_90d, label)
    ("A", 90, None, 3, "น่าเชื่อถือสูง"),
    ("B", 70, 3000, 2, "น่าเชื่อถือปานกลาง"),
    ("C", 50, 1000, 1, "ต้องระวัง"),
    ("D", -999, 0, 0, "เสี่ยงสูง งด COD"),
]
TIER_ORDER = [t[0] for t in TIERS]
BORDERLINE_MARGIN = 5  # points from a tier boundary that count as "borderline"


@dataclass
class ScoreResult:
    score: int
    tier: str
    reasons: list = field(default_factory=list)  # (points, text)
    borderline: bool = False


def tier_for_score(score: float) -> str:
    for tier, min_score, *_ in TIERS:
        if score >= min_score:
            return tier
    return "D"


def tier_policy(tier: str) -> dict:
    for t, min_score, limit, refusals, label in TIERS:
        if t == tier:
            return {
                "tier": t,
                "cod_limit_thb": limit,
                "refusals_allowed_90d": refusals,
                "label": label,
            }
    raise ValueError(f"Unknown tier {tier}")


def bayesian_refusal_rate(refused: int, cod_orders: int, a: float = 1.0, b: float = 19.0) -> float:
    """Smoothed refusal rate. Prior a/(a+b) = 5% platform average.

    A buyer with 1 refusal out of 1 order gets ~9.5%, not 100%.
    """
    return (refused + a) / (cod_orders + a + b)


def score_user(u: dict) -> ScoreResult:
    score = 100
    reasons = []

    def add(points: int, text: str) -> None:
        nonlocal score
        if points:
            score += points
            reasons.append((points, text))

    rate = bayesian_refusal_rate(u["refused_count"], u["cod_orders"])
    add(-round(max(rate - 0.05, 0) * 200), f"อัตราตีกลับ (ปรับแบบ Bayesian) {rate:.0%}")
    add(-10 * int(u.get("refused_90d", 0)), f"ตีกลับใน 90 วันล่าสุด {u.get('refused_90d', 0)} ครั้ง")
    add(min(int(u["success_count"]) * 2, 30), f"รับของสำเร็จ {u['success_count']} ครั้ง")

    if u["account_age_days"] < 30:
        add(-15, f"บัญชีใหม่ อายุ {u['account_age_days']} วัน")
    elif u["account_age_days"] > 365:
        add(5, "บัญชีเก่ากว่า 1 ปี")

    if _truthy(u["kyc_verified"]):
        add(10, "ยืนยันตัวตน (KYC) แล้ว")
    if _truthy(u.get("shopeepay_linked", False)):
        add(5, "ผูก ShopeePay แล้ว")

    shared = int(u["shared_device_accounts"])
    if shared > 2:
        add(-30, f"อุปกรณ์เดียวกันถูกใช้กับ {shared} บัญชี")
    elif shared > 0:
        add(-5, f"อุปกรณ์ถูกใช้ร่วมกับ {shared} บัญชี")

    if u.get("address_changes_90d", 0) >= 3:
        add(-10, f"เปลี่ยนที่อยู่ {u['address_changes_90d']} ครั้งใน 90 วัน")
    if u.get("high_value_ratio", 1) >= 2.5:
        add(-10, f"มูลค่าออเดอร์ล่าสุดสูงกว่าปกติ {u['high_value_ratio']} เท่า")

    score = max(0, min(130, score))
    tier = tier_for_score(score)
    boundaries = [t[1] for t in TIERS if t[1] > 0]
    borderline = any(abs(score - b) <= BORDERLINE_MARGIN for b in boundaries)
    return ScoreResult(score=score, tier=tier, reasons=reasons, borderline=borderline)


def needs_llm_review(u: dict, result: ScoreResult) -> bool:
    """Send only the cases where judgement helps: borderline, text reasons, or about to be blocked."""
    return (
        result.borderline
        or result.tier == "D"
        # A text reason only matters if it could change a non-A decision
        or (result.tier != "A"
            and bool(str(u.get("last_refusal_reason") or "").strip())
            and u["refused_count"] > 0)
    )


def _truthy(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in {"true", "1", "yes"}
    return bool(v)
