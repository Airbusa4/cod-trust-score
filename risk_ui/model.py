"""COD Risk Score maths for the Streamlit app, in plain numpy / pandas.

SYNTHETIC DATA - for illustration only. This is NOT real Shopee data.

Everything here mirrors cod_risk_demo/ exactly (features, model, tiers, money);
cod_risk_demo/src/export_app.py checks that on all 76,232 orders.
"""
import calendar

import numpy as np
import pandas as pd

TIERS = (1, 2, 3, 4)
# Lowest risk that puts an order in each tier (Tier 1 is everything below 3%).
TIER_CUTOFFS = {2: 0.03, 3: 0.10, 4: 0.77}
TIER_ACTIONS = {1: "Normal COD + COD Reminder", 2: "COD Confirmation + Order Hold",
                3: "Refundable Deposit 10%", 4: "COD → Prepaid Requirement"}
TIER_NAMES = {1: "Tier 1 · Normal COD", 2: "Tier 2 · Confirm & hold", 3: "Tier 3 · Deposit", 4: "Tier 4 · Prepaid only"}
# Tiers are ordered, so one hue (orange) in monotone steps: higher risk = darker
# (checked: monotone, distinct steps, the darkest step still clears 2:1 on the black background).
TIER_COLORS = {1: "#ffb069", 2: "#f47b20", 3: "#c25e17", 4: "#8c4614"}
TIER_TEXT_ON = {1: "#0b0b0c", 2: "#0b0b0c", 3: "#ffffff", 4: "#ffffff"}  # readable text on each tier color
TIER_RANGES = {1: "below 3%", 2: "3% to 10%", 3: "10% to 77%", 4: "77% and above"}
DEPOSIT_SHARE = 0.10

# Policy simulator starting values. ASSUMPTIONS, not measured: adjust them in the app.
# catch = share of predicted failures the tier's action prevents;
# friction = share of good orders lost because of the extra step.
SIM_DEFAULTS = {"catch": {1: 0.05, 2: 0.25, 3: 0.50, 4: 0.90},
                "friction": {1: 0.0, 2: 0.005, 3: 0.03, 4: 0.15}}

NICE_NAMES = {
    "hist_cod_orders": "Past COD orders",
    "hist_refusals": "Past refusals",
    "refusal_rate_smoothed": "Refusal rate (smoothed)",
    "recent_refusal_rate_smoothed_90d": "Refusal rate, last 90 days",
    "days_since_last_refusal": "Days since last refusal",
    "has_ever_refused": "Has ever refused",
    "hist_buyer_caused_misses": "Past 'not home / no cash' misses",
    "hist_courier_caused_failures": "Past courier failures",
    "order_value": "Order value (THB)",
    "value_vs_aov": "Value vs buyer's usual",
    "freq_change_ratio": "Order frequency change (30d)",
    "account_age_days": "Account age (days)",
    "is_new_cod_buyer": "First COD order",
    "is_late_night": "Ordered 01:00-04:59",
    "same_item_other_shops_48h": "Same item at other shops (48h)",
    "is_campaign_day": "Campaign day",
    "expected_days_to_delivery": "Expected delivery days",
    "area_logistics_failure_rate": "Area courier failure rate (90d)",
    "address_is_condo_with_office": "Condo with front office",
    "order_month": "Order month",
}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------------
# Raw inputs -> the 20 model features (same maths as build_features.py)
# ---------------------------------------------------------------------------
def features_from_inputs(d, fs):
    """`d` is a DataFrame of raw inputs (one row per order). `fs` = feature settings.

    Needed columns: past_orders, past_refusals, orders_last_90d, refusals_last_90d,
    days_since_last_refusal, past_not_home, past_courier_failures, order_value,
    usual_order_value, orders_in_transit, orders_last_30d, days_since_first_cod,
    account_age_days, order_datetime, order_hour, same_item_other_shops, delivery_days,
    condo_with_office, area_logistics_failure_rate.
    """
    a = fs["refusal_prior_rate"] * fs["refusal_prior_strength"]
    b = (1 - fs["refusal_prior_rate"]) * fs["refusal_prior_strength"]
    n, r = d["past_orders"].astype(float), d["past_refusals"].astype(float)
    when = pd.to_datetime(d["order_datetime"])

    # Value vs usual: compares with ALL earlier placed orders (incl. ones still in transit).
    n_placed = n + d["orders_in_transit"]
    usual = d["usual_order_value"].astype(float)
    value_vs_aov = np.where((n_placed > 0) & (usual > 0), d["order_value"] / usual.where(usual > 0, 1), 1.0)

    # Orders in the last 30 days vs the buyer's smoothed usual monthly rate.
    avg_monthly = (n_placed + fs["freq_prior_monthly_rate"] * fs["freq_prior_months"]) / (
        d["days_since_first_cod"] / 30.44 + fs["freq_prior_months"])

    out = pd.DataFrame({
        "hist_cod_orders": n.astype(int),
        "hist_refusals": r.astype(int),
        "refusal_rate_smoothed": ((r + a) / (n + a + b)).round(6),
        "recent_refusal_rate_smoothed_90d": ((d["refusals_last_90d"] + a) / (d["orders_last_90d"] + a + b)).round(6),
        "days_since_last_refusal": np.where(r > 0, d["days_since_last_refusal"], -1).astype(int),
        "has_ever_refused": (r > 0).astype(int),
        "hist_buyer_caused_misses": d["past_not_home"].astype(int),
        "hist_courier_caused_failures": d["past_courier_failures"].astype(int),
        "order_value": d["order_value"],
        "value_vs_aov": np.round(value_vs_aov, 4),
        "freq_change_ratio": np.round(d["orders_last_30d"] / avg_monthly, 4),
        "account_age_days": d["account_age_days"].astype(int),
        "is_new_cod_buyer": (n == 0).astype(int),
        "is_late_night": d["order_hour"].between(1, 4).astype(int),
        "same_item_other_shops_48h": d["same_item_other_shops"].astype(int),
        "is_campaign_day": (when.dt.month == when.dt.day).astype(int).to_numpy(),
        "expected_days_to_delivery": d["delivery_days"].astype(int),
        "area_logistics_failure_rate": d["area_logistics_failure_rate"].astype(float).round(6),
        "address_is_condo_with_office": d["condo_with_office"].astype(int),
        "order_month": when.dt.month.to_numpy(),
    }, index=d.index)
    return out


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class Model:
    """Calibrated Logistic Regression, rebuilt from exported numbers."""

    def __init__(self, params):
        self.features = params["features"]
        self.mean = np.array(params["scaler_mean"])
        self.scale = np.array(params["scaler_scale"])
        self.coef = np.array(params["coef"])
        self.intercept = params["intercept"]
        self.a, self.b = params["platt_a"], params["platt_b"]
        self.bg = np.array(params["background_mean_scaled"])

    def _z(self, X):
        return (X[self.features].to_numpy(dtype=float) - self.mean) / self.scale

    def predict(self, X):
        raw = sigmoid(self.intercept + self._z(X) @ self.coef)
        return sigmoid(self.a * logit(raw) + self.b)

    def contributions(self, X):
        """SHAP values for a linear model (in calibrated log-odds): how much each
        feature pushes this order's risk up (+) or down (-) vs an average order."""
        return self.a * self.coef * (self._z(X) - self.bg)


def assign_tiers(p, cutoffs=TIER_CUTOFFS):
    p = np.asarray(p)
    return np.select([p >= cutoffs[4], p >= cutoffs[3], p >= cutoffs[2]], [4, 3, 2], 1)


# Money lost when a COD order fails (THB): shipping out + shipping back + the commission on the order.
SHIP_OUT_THB = 24
SHIP_RETURN_THB = 20
COMMISSION_RATE = 0.10
COST_FORMULA = (f"(฿{SHIP_OUT_THB} shipping out + ฿{SHIP_RETURN_THB} shipping back "
                f"+ {COMMISSION_RATE:.0%} of the order value in lost commission) × predicted risk")


def commission(order_value):
    return COMMISSION_RATE * np.asarray(order_value, dtype=float)


def cost_of_failure(order_value):
    """Money lost if this order fails (THB), per order."""
    return SHIP_OUT_THB + SHIP_RETURN_THB + commission(order_value)


# ---------------------------------------------------------------------------
# Plain-word reasons (same wording as cod_risk_demo/src/explain.py)
# ---------------------------------------------------------------------------
def plural(n, word):
    n = int(n)
    return f"{n} {word}" + ("" if n == 1 else "s")


def plain_reason(f, r):
    v = r[f]
    texts = {
        "hist_cod_orders": lambda: ("First COD order, no history yet" if v == 0 else
                                    f"Only {int(v)} past COD orders (little track record)" if v < 10 else
                                    f"{int(v)} past COD orders (long track record)"),
        "hist_refusals": lambda: "Never refused a COD parcel" if v == 0 else f"Refused {int(v)} of {int(r['hist_cod_orders'])} past COD parcels",
        "refusal_rate_smoothed": lambda: (f"Refused {int(r['hist_refusals'])} of {int(r['hist_cod_orders'])} past COD parcels"
                                          if r["hist_cod_orders"] > 0 else "No refusal history yet (average assumed)"),
        "recent_refusal_rate_smoothed_90d": lambda: f"Recent (90-day) refusal rate {v:.1%} after smoothing",
        "days_since_last_refusal": lambda: "Never refused before" if v < 0 else f"Last refusal {int(v)} days ago",
        "has_ever_refused": lambda: "Has refused a COD parcel before" if v else "Has never refused a COD parcel",
        "hist_buyer_caused_misses": lambda: f"{int(v)} past 'not home / no cash' misses",
        "hist_courier_caused_failures": lambda: f"{int(v)} past courier failures (not the buyer's fault)",
        "order_value": lambda: f"Order value ฿{v:,.0f}",
        "value_vs_aov": lambda: f"Order is {v:.1f}x the buyer's usual value",
        "freq_change_ratio": lambda: f"Ordering {v:.1f}x as often as usual in the last 30 days",
        "account_age_days": lambda: f"Account is {plural(v, 'day')} old",
        "is_new_cod_buyer": lambda: "First-ever COD order" if v else "Has COD history",
        "is_late_night": lambda: (f"Ordered late at night ({int(r['order_hour']):02d}:00-{int(r['order_hour']):02d}:59)"
                                  if v else "Ordered in normal hours"),
        "same_item_other_shops_48h": lambda: (f"Same item ordered with COD at {plural(v, 'other shop')} in 48h" if v
                                              else "No similar COD orders at other shops"),
        "is_campaign_day": lambda: "Campaign day order" if v else "Not a campaign day",
        "expected_days_to_delivery": lambda: f"Delivery expected in {plural(v, 'day')}",
        "area_logistics_failure_rate": lambda: f"Area courier failure rate {v:.2%} (last 90 days)",
        "address_is_condo_with_office": lambda: "Condo with a front office to receive parcels" if v else "House address (no front office)",
        "order_month": lambda: f"Ordered in {calendar.month_name[int(v)]}",
    }
    return texts[f]()


def top_reasons(model, feats_row, order_hour, k=3):
    """Top-k reasons for ONE order: list of dicts (feature, name, text, shap, direction)."""
    X = feats_row.to_frame().T if isinstance(feats_row, pd.Series) else feats_row
    phi = model.contributions(X)[0]
    r = {**X.iloc[0].to_dict(), "order_hour": order_hour}
    # Some features say the same thing in words (e.g. refusal count and refusal rate):
    # keep the first of each wording, so the top reasons are all different.
    reasons, seen = [], set()
    for j in np.argsort(-np.abs(phi)):
        text = plain_reason(model.features[j], r)
        if text in seen:
            continue
        seen.add(text)
        reasons.append({"feature": model.features[j], "name": NICE_NAMES[model.features[j]], "text": text,
                        "shap": float(phi[j]), "direction": "up" if phi[j] > 0 else "down"})
        if len(reasons) == k:
            break
    return reasons, phi


# ---------------------------------------------------------------------------
# Money: expected values from the PREDICTED risk only (no actual outcomes)
# ---------------------------------------------------------------------------
def simulate(tier, risk, order_value, dc, catch, friction):
    """Expected money effect of applying each order's tier action.

    tier: 0 = no action, 1-4 = that tier's action. risk: predicted failure chance.
    order_value: THB per order (commission is COMMISSION_RATE of it).
    catch / friction: {tier: rate}. dc needs caught_become_delivered, message_cost_thb.
    Returns totals in THB for these orders.
    """
    tier, risk, comm = np.asarray(tier), np.asarray(risk, dtype=float), commission(order_value)
    prevented = prevented_comm = lost_comm = messaged = 0.0
    for k in TIERS:
        in_tier = tier == k
        prevented += risk[in_tier].sum() * catch[k]
        prevented_comm += (risk[in_tier] * comm[in_tier]).sum() * catch[k]
        lost_comm += ((1 - risk[in_tier]) * comm[in_tier]).sum() * friction[k]
        messaged += in_tier.sum()
    shipping = prevented * (SHIP_OUT_THB + SHIP_RETURN_THB)
    back = prevented_comm * dc["caught_become_delivered"]
    msg = messaged * dc["message_cost_thb"]
    return {"failures_prevented": prevented, "shipping_saved": shipping, "commission_won_back": back,
            "commission_lost_friction": lost_comm, "message_cost": msg, "net": shipping + back - lost_comm - msg}
