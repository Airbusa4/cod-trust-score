"""Step 1: simulate buyers, areas and 18 months of COD orders.

SYNTHETIC DATA - for illustration only. This is NOT real Shopee data.

Flow:
  1. make_areas   -> 60 areas, each with a hidden courier failure rate
  2. make_buyers  -> 30,000 buyers with hidden traits (latent_wont, latent_cant, ...)
  3. make_orders  -> every buyer's COD orders over Feb 2025 - Jul 2026
  4. add_order_details -> value, hour, campaign day, delivery days, same item elsewhere
  5. score_orders -> hidden log-odds for won't / can't / logistics
  6. calibrate    -> one intercept per failure type so the model window hits the targets
  7. draw outcomes at random from those chances

Output: data/raw_events.csv (audit file, contains hidden values) and docs/hidden_rules.md
"""
import numpy as np
import pandas as pd

from src.common import (
    DATA_DIR, DOCS_DIR, SYNTHETIC_NOTE, day_number, days_to_datetime,
    load_config, order_context, write_csv, write_text,
)


def logit(p):
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def draw_mixture(rng, n, components, max_value):
    """Draw n values from a mixture of Beta distributions.
    Each component has a share of buyers, a mean and a concentration."""
    shares = np.array([c["share"] for c in components])
    which = rng.choice(len(components), size=n, p=shares / shares.sum())
    values = np.empty(n)
    for i, c in enumerate(components):
        mask = which == i
        a = c["mean"] * c["concentration"]
        b = (1 - c["mean"]) * c["concentration"]
        values[mask] = rng.beta(a, b, size=mask.sum())
    return np.clip(values, 1e-4, max_value), which


# ---------------------------------------------------------------------------
# 1. Areas
# ---------------------------------------------------------------------------
def make_areas(cfg, rng):
    ac = cfg["areas"]
    n = ac["n_areas"]
    # Assign areas to rate groups by exact counts (so "3 poor areas" really is 3).
    counts = [int(round(g["share"] * n)) for g in ac["logistics_rate_groups"]]
    counts[0] += n - sum(counts)
    rates = np.concatenate([
        rng.uniform(g["low"], g["high"], size=k)
        for g, k in zip(ac["logistics_rate_groups"], counts)
    ])
    rng.shuffle(rates)
    return pd.DataFrame({
        "area_id": np.arange(1, n + 1),
        "latent_area_logistics": rates,
        "base_delivery_days": rng.choice(ac["base_delivery_days"], size=n, p=ac["base_delivery_days_probs"]),
    })


# ---------------------------------------------------------------------------
# 2. Buyers
# ---------------------------------------------------------------------------
def make_buyers(cfg, rng, n_areas):
    bc, lc, d = cfg["buyers"], cfg["latent"], cfg["dates"]
    n = bc["n_buyers"]
    sim_start = d["sim_start"]
    model_day = day_number(d["model_start"], sim_start)
    end_day = day_number(d["sim_end"], sim_start) + 1  # exclusive end

    # Signup day (days since sim start; negative = before the simulation started).
    shares = bc["signup_period_shares"]
    period = rng.choice(3, size=n, p=[shares["before_sim"], shares["history_window"], shares["model_window"]])
    before_from = day_number(bc["signup_before_sim_from"], sim_start)
    bounds = [(before_from, 0), (0, model_day), (model_day, end_day)]
    signup_day = np.empty(n)
    for i, (lo, hi) in enumerate(bounds):
        mask = period == i
        signup_day[mask] = rng.uniform(lo, hi, size=mask.sum())

    # Ordering habit: the share of regular buyers depends on when they joined.
    rs = bc["regular_share_by_period"]
    regular_share = np.array([rs["before_sim"], rs["history_window"], rs["model_window"]])[period]
    is_regular = rng.random(n) < regular_share
    monthly_rate = np.where(
        is_regular,
        rng.uniform(*bc["regular_monthly_rate"], size=n),
        bc["occasional_monthly_rate_median"] * np.exp(rng.normal(0, bc["occasional_monthly_rate_sigma"], size=n)),
    )

    # Hidden traits. The high won't group is a bit larger among recent signups.
    latent_wont, wont_group = np.empty(n), np.empty(n, dtype=int)
    high_by_period = lc["wont_high_share_by_period"]
    for i, name in enumerate(["before_sim", "history_window", "model_window"]):
        comps = [dict(c) for c in lc["wont_mixture"]]
        comps[0]["share"] += comps[-1]["share"] - high_by_period[name]
        comps[-1]["share"] = high_by_period[name]
        mask = period == i
        latent_wont[mask], wont_group[mask] = draw_mixture(rng, mask.sum(), comps, lc["max_latent"])
    latent_cant, _ = draw_mixture(rng, n, lc["cant_mixture"], lc["max_latent"])

    # Drift: ~10% of buyers change their latent_wont in the last 90 days.
    is_drift = rng.random(n) < lc["drift_share"]
    drift_up = rng.random(n) < lc["drift_up_share"]
    drift_direction = np.where(is_drift, np.where(drift_up, "up", "down"), "none")

    return pd.DataFrame({
        "buyer_id": np.arange(1, n + 1),
        "signup_day": np.round(signup_day, 5),
        "area_id": rng.integers(1, n_areas + 1, size=n),
        "address_type": np.where(rng.random(n) < bc["condo_with_office_share"], "condo_with_office", "house"),
        "is_regular": is_regular,
        "monthly_rate": monthly_rate,
        "latent_aov": np.round(np.clip(bc["aov_median_thb"] * np.exp(rng.normal(0, bc["aov_sigma"], size=n)),
                              bc["aov_min_thb"], bc["aov_max_thb"]), 2),
        "latent_wont_base": latent_wont,
        "wont_group": wont_group,
        "latent_cant": latent_cant,
        "drift_direction": drift_direction,
    })


# ---------------------------------------------------------------------------
# 3. Orders (when each buyer orders)
# ---------------------------------------------------------------------------
def day_weights(cfg, n_days):
    """Relative chance that an order lands on each day. Campaign days get more."""
    dates = pd.date_range(cfg["dates"]["sim_start"], periods=n_days, freq="D")
    is_campaign = np.asarray(dates.month == dates.day)
    w = np.where(is_campaign, cfg["buyers"]["campaign_day_order_weight"], 1.0)
    return w, is_campaign


def sample_days(rng, cum_w, start_day, end_day):
    """Pick one whole day per row in [start_day, end_day] using the day weights."""
    lo = cum_w[start_day]
    hi = cum_w[end_day + 1]
    target = lo + rng.random(len(lo)) * (hi - lo)
    return np.searchsorted(cum_w, target, side="right") - 1


def make_orders(cfg, rng, buyers):
    bc = cfg["buyers"]
    sim_start = cfg["dates"]["sim_start"]
    end_day = day_number(cfg["dates"]["sim_end"], sim_start) + 1  # exclusive
    w, _ = day_weights(cfg, end_day)
    cum_w = np.concatenate([[0.0], np.cumsum(w)])

    n = len(buyers)
    # COD activity window: first COD order shortly after signup (or sim start),
    # then the buyer keeps using COD for a random time.
    cod_start = np.maximum(buyers["signup_day"].to_numpy(), 0) + rng.exponential(bc["first_cod_lag_mean_days"], n)
    active_mean = np.where(buyers["is_regular"], bc["active_days_mean_regular"], bc["active_days_mean_occasional"])
    cod_end = np.minimum(cod_start + rng.exponential(active_mean), end_day - 1e-6)
    has_orders = cod_start < end_day
    start_int = np.floor(cod_start).astype(int)
    end_int = np.floor(np.maximum(cod_end, cod_start)).astype(int)

    # How many orders: the first one, plus a Poisson number over the active window.
    active_months = np.maximum(cod_end - cod_start, 0) / 30.44
    n_more = rng.poisson(buyers["monthly_rate"].to_numpy() * active_months)

    parts = []
    idx = np.flatnonzero(has_orders)
    # (a) the first COD order, on the first active day
    parts.append(pd.DataFrame({"b": idx, "day": start_int[idx], "is_spike_order": False}))
    # (b) the normal orders, spread over the active window (campaign days weighted up)
    rep = np.repeat(idx, n_more[idx])
    parts.append(pd.DataFrame({"b": rep, "day": sample_days(rng, cum_w, start_int[rep], end_int[rep]),
                               "is_spike_order": False}))
    # (c) spikes: a few buyers squeeze 5-8 extra orders into one 30-day window
    spike = idx[rng.random(len(idx)) < bc["spike_share"]]
    spike_start = np.floor(rng.uniform(cod_start[spike], np.maximum(cod_end[spike], cod_start[spike]))).astype(int)
    spike_end = np.minimum(spike_start + bc["spike_window_days"] - 1, end_day - 1)
    k = rng.integers(bc["spike_extra_orders"][0], bc["spike_extra_orders"][1] + 1, size=len(spike))
    rep = np.repeat(np.arange(len(spike)), k)
    parts.append(pd.DataFrame({"b": spike[rep], "day": sample_days(rng, cum_w, spike_start[rep], spike_end[rep]),
                               "is_spike_order": True}))
    orders = pd.concat(parts, ignore_index=True)

    # Hour of the order: 8% between 01:00 and 04:59, the rest over other hours.
    other_hours = np.array([0] + list(range(5, 24)))
    hour_w = np.where(np.isin(other_hours, bc["evening_peak_hours"]), bc["evening_peak_weight"], 1.0)
    late = rng.random(len(orders)) < bc["late_night_share"]
    hour = np.where(late, rng.integers(1, 5, size=len(orders)),
                    rng.choice(other_hours, size=len(orders), p=hour_w / hour_w.sum()))
    orders["order_hour"] = hour
    # Time in days since sim start, rounded to about 1 second (keeps files small).
    orders["order_day"] = np.round(orders["day"] + (hour + rng.random(len(orders))) / 24, 5)

    # The first order cannot happen before the buyer signed up.
    signup = buyers["signup_day"].to_numpy()[orders["b"]]
    orders = orders[orders["order_day"] >= signup]
    orders = orders[orders["order_day"] < end_day]

    orders = orders.sort_values(["b", "order_day"]).reset_index(drop=True)
    orders = orders.merge(buyers, left_on="b", right_index=True).drop(columns="b")
    orders = orders.sort_values(["buyer_id", "order_day"]).reset_index(drop=True)
    return orders


# ---------------------------------------------------------------------------
# 4. Order details
# ---------------------------------------------------------------------------
def add_order_details(cfg, rng, orders, areas):
    bc, ac = cfg["buyers"], cfg["areas"]
    sim_start = cfg["dates"]["sim_start"]
    end_day = day_number(cfg["dates"]["sim_end"], sim_start) + 1
    n = len(orders)

    # Value: around the buyer's usual level, with some orders 3-6x higher.
    value = orders["latent_aov"].to_numpy() * np.exp(rng.normal(0, bc["order_value_sigma"], n))
    big = rng.random(n) < bc["big_order_share"]
    value = np.where(big, value * rng.uniform(*bc["big_order_multiplier"], n), value)
    orders["order_value"] = np.maximum(np.round(value), bc["min_order_value_thb"]).astype(int)

    # Calendar.
    _, is_campaign = day_weights(cfg, end_day)
    day_int = np.floor(orders["order_day"].to_numpy()).astype(int)
    orders["is_campaign_day"] = is_campaign[day_int].astype(int)
    dt = days_to_datetime(orders["order_day"].to_numpy(), sim_start)
    orders["order_datetime"] = dt.strftime("%Y-%m-%d %H:%M:%S")
    orders["order_month"] = dt.month
    orders["order_dom"] = dt.day

    # Delivery days depend on the area, plus a bit of randomness.
    area_days = areas.set_index("area_id")["base_delivery_days"]
    base = orders["area_id"].map(area_days).to_numpy()
    orders["expected_days_to_delivery"] = np.minimum(
        base + rng.integers(0, ac["extra_delivery_days_max"] + 1, n), ac["max_delivery_days"])

    # Same item ordered with COD at other shops in the last 48h (mostly 0).
    # Price shoppers / COD abusers (high latent_wont) do this more often.
    p_any = bc["same_item_base_prob"] + bc["same_item_wont_slope"] * orders["latent_wont_base"].to_numpy()
    counts = rng.choice(bc["same_item_count_values"], size=n, p=bc["same_item_count_probs"])
    orders["same_item_other_shops_48h"] = np.where(rng.random(n) < p_any, counts, 0)

    orders["account_age_days"] = np.floor(orders["order_day"] - orders["signup_day"]).astype(int)
    orders["latent_area_logistics"] = orders["area_id"].map(areas.set_index("area_id")["latent_area_logistics"])

    # Drift: in the last 90 days, drifting buyers' latent_wont is x3 or /3.
    lc = cfg["latent"]
    in_drift_window = orders["order_day"].to_numpy() >= end_day - lc["drift_last_days"]
    factor = np.select([orders["drift_direction"] == "up", orders["drift_direction"] == "down"],
                       [lc["drift_factor"], 1 / lc["drift_factor"]], 1.0)
    orders["latent_wont"] = np.clip(
        orders["latent_wont_base"].to_numpy() * np.where(in_drift_window, factor, 1.0), 1e-4, lc["max_latent"])

    # Order-context features that the hidden rule also uses (placed orders only).
    ctx = order_context(orders, cfg)
    orders["freq_change_ratio"] = ctx["freq_change_ratio"]
    return orders


# ---------------------------------------------------------------------------
# 5. Hidden scores (log-odds, before the intercept)
# ---------------------------------------------------------------------------
def score_orders(cfg, rng, orders):
    ew, ec, el = cfg["effects"]["wont"], cfg["effects"]["cant"], cfg["effects"]["logistics"]
    n = len(orders)

    # Won't: buyer refuses at the door.
    # "Usual value" here is the buyer's TRUE usual level (latent_aov); the model
    # only sees an estimate of it (value_vs_aov from past orders).
    true_ratio = orders["order_value"] / orders["latent_aov"]
    days = orders["expected_days_to_delivery"]
    s_wont = (
        logit(orders["latent_wont"])
        + ew["same_item_per_shop"] * np.minimum(orders["same_item_other_shops_48h"], ew["same_item_cap"])
        + np.select([true_ratio >= 3, true_ratio >= 2], [ew["value_3x_plus"], ew["value_2_to_3x"]], 0.0)
        + ew["new_account"] * (orders["account_age_days"] < ew["new_account_days"])
        + ew["late_night"] * orders["order_hour"].between(1, 4)
        + ew["freq_spike"] * (orders["freq_change_ratio"] >= ew["freq_spike_threshold"])
        + ew["campaign_day"] * orders["is_campaign_day"]
        + ew["per_delivery_day"] * np.maximum(days - ew["per_delivery_day_above"], 0)
        + rng.normal(0, ew["noise_sd"], n)
    )

    # Can't: buyer still wants it but is not home / has no cash.
    april_or_late_dec = (orders["order_month"] == 4) | (
        (orders["order_month"] == 12) & (orders["order_dom"] >= ec["late_december_from_day"]))
    s_cant = (
        logit(orders["latent_cant"])
        + ec["april_or_late_december"] * april_or_late_dec
        + ec["condo_with_office"] * (orders["address_type"] == "condo_with_office")
        + ec["per_delivery_day"] * days
        + rng.normal(0, ec["noise_sd"], n)
    )

    # Logistics: courier / network problem. Depends on the area only, not the buyer.
    s_log = logit(orders["latent_area_logistics"]) + rng.normal(0, el["noise_sd"], n)
    return s_wont.to_numpy(), s_cant.to_numpy(), s_log.to_numpy()


def calibrate(score, mask, target):
    """Find the intercept c so that mean(sigmoid(score + c)) over `mask` == target.
    Simple bisection: the mean goes up as c goes up."""
    lo, hi = -15.0, 15.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if sigmoid(score[mask] + mid).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# 6. Hidden-rules document (written from the config, so it never goes stale)
# ---------------------------------------------------------------------------
def write_hidden_rules(cfg, intercepts, summary):
    ew, ec = cfg["effects"]["wont"], cfg["effects"]["cant"]
    lc = cfg["latent"]
    mix = lambda comps: "; ".join(f"{c['share']:.0%} of buyers around {c['mean']:.1%}" for c in comps)
    text = f"""# Hidden rules of the generator

> {SYNTHETIC_NOTE}

This page lists the TRUE effects used by `src/generate_data.py`. The model never
sees these values. Use this page in the demo to compare what the model learns
with what really drives failures. All numbers come from `config.yaml`
(seed = {cfg['seed']}).

## How a chance is built

Each failure type gets a score on the log-odds scale:

    score = logit(hidden trait) + calibrated intercept + effects + random noise
    chance = 1 / (1 + exp(-score))

The three chances are then used to draw ONE outcome per order at random
(`wont`, `cant`, `logistics` or delivered). There are no hard cutoffs.

The buyer's **history is not in the rule**. Past refusals are only evidence
about the hidden trait `latent_wont`. That is why the features smooth the
refusal rate: 1 refusal out of 1 order is weak evidence, 15 out of 40 is strong.

## Hidden buyer traits

| Trait | Mixture (Beta distributions) |
| --- | --- |
| `latent_wont` | {mix(lc['wont_mixture'])}. The high group's share varies by signup period: {', '.join(f'{k} {v:.0%}' for k, v in lc['wont_high_share_by_period'].items())} (the low group absorbs the difference) |
| `latent_cant` | {mix(lc['cant_mixture'])} |
| Drift | {lc['drift_share']:.0%} of buyers: `latent_wont` x{lc['drift_factor']:g} ({lc['drift_up_share']:.0%} of them) or /{lc['drift_factor']:g} (the rest) in the last {lc['drift_last_days']} days |

## Won't (`p_wont`), buyer refuses at the door

Start from `logit(latent_wont)` + intercept **{intercepts['wont']:+.3f}**, then add:

| Factor | Effect (log-odds) |
| --- | --- |
| `same_item_other_shops_48h` | +{ew['same_item_per_shop']} per shop (max {ew['same_item_cap']}) |
| Order value 2 to 3 times the buyer's true usual value | +{ew['value_2_to_3x']} |
| Order value 3 times or more | +{ew['value_3x_plus']} |
| Account younger than {ew['new_account_days']} days | +{ew['new_account']} |
| Order 01:00 to 04:59 | +{ew['late_night']} |
| `freq_change_ratio` >= {ew['freq_spike_threshold']:g} | +{ew['freq_spike']} |
| Campaign day | +{ew['campaign_day']} |
| Each day of delivery wait above {ew['per_delivery_day_above']} | +{ew['per_delivery_day']} |
| Random noise | normal, sd {ew['noise_sd']} |

## Can't (`p_cant`), buyer not home or no cash

Start from `logit(latent_cant)` + intercept **{intercepts['cant']:+.3f}**, then add:

| Factor | Effect (log-odds) |
| --- | --- |
| April, or {ec['late_december_from_day']} to 31 December | +{ec['april_or_late_december']} |
| Condo with front office | {ec['condo_with_office']} |
| Each day of delivery wait | +{ec['per_delivery_day']} |
| Random noise | normal, sd {ec['noise_sd']} |

## Logistics (`p_logistics`), courier or network problem

`logit(latent_area_logistics)` of the buyer's area + intercept **{intercepts['logistics']:+.3f}**
+ noise (sd {cfg['effects']['logistics']['noise_sd']}). Not linked to the buyer at all.

## Calibration result (model rows, Feb to Jul 2026)

| Type | Target mean chance | Achieved mean chance | Realised rate |
| --- | --- | --- | --- |
{summary}

## Choices not spelled out in the brief (and why)

- **Beta mixtures.** Each mixture component is a Beta distribution around the
  stated mean, so buyers inside a group still differ a little. A point mass
  would make history features unrealistically clean.
- **Value effect uses the true usual value.** The hidden rule compares the
  order to the buyer's hidden `latent_aov`. The model only sees `value_vs_aov`,
  an estimate from past orders (and 1.0 for buyers with no history). This is
  how it works in real life: the platform only ever sees an estimate.
- **`same_item_other_shops_48h` is linked to `latent_wont`.** The chance of a
  non-zero count is {cfg['buyers']['same_item_base_prob']} + {cfg['buyers']['same_item_wont_slope']} x `latent_wont`.
  Price shopping and COD abuse are the behaviours behind refusals, so buyers
  who refuse more also shop around more. The +1.0 effect in the rule still applies.
- **Recent signups have a larger high-refusal group** (see table above).
  Throwaway accounts used for COD abuse tend to be new. Without this, the few
  long-standing heavy refusers pull the average up so much that new COD
  buyers look no riskier than average, which contradicts the case.
- **Regular vs occasional buyers depend on signup period**
  ({', '.join(f'{k} {v:.0%}' for k, v in cfg['buyers']['regular_share_by_period'].items())} regular).
  Brand-new buyers mostly try COD once or twice. This affects only how often
  buyers order, not the outcome rule. It is needed so that at least 25% of model
  rows are first-time COD buyers.
- **Poor courier areas are worse than the brief's starting value.** The brief
  starts at "a few areas up to 0.6%". At that level a clean buyer's own
  won't + can't chance (about 1.2%) is always larger than the area's logistics
  chance, so failures in a poor area could never be mostly `logistics`
  (edge case 6). The 3 poor areas start at
  {cfg['areas']['logistics_rate_groups'][-1]['low']:.1%} to {cfg['areas']['logistics_rate_groups'][-1]['high']:.1%}
  before calibration; the overall logistics rate still hits its target.
- **Condo effect counted once.** The brief says `latent_cant` is lower for
  condos with a front office AND lists a -0.7 effect. We apply only the -0.7
  effect so the condo effect is not counted twice.
- **Outcome timing.** An order's outcome becomes known on
  `order time + expected_days_to_delivery + 0 to {cfg['calibration']['resolution_extra_days_max']} day`.
  History features only count past orders whose outcome was already known.
- **Drift window.** Drift applies to orders in the last {lc['drift_last_days']} days of the
  simulation (May to Jul 2026), so drifting buyers appear in train and test.
- **Probability cap.** The three chances together are capped at
  {cfg['effects']['max_total_failure_prob']} (almost never reached).
"""
    write_text(DOCS_DIR / "hidden_rules.md", text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg = load_config()
    rng = np.random.default_rng(cfg["seed"])
    sim_start = cfg["dates"]["sim_start"]
    model_day = day_number(cfg["dates"]["model_start"], sim_start)

    print("Making areas and buyers ...")
    areas = make_areas(cfg, rng)
    buyers = make_buyers(cfg, rng, len(areas))

    print("Simulating orders ...")
    orders = make_orders(cfg, rng, buyers)
    orders = add_order_details(cfg, rng, orders, areas)
    orders["is_model_row"] = (orders["order_day"] >= model_day).astype(int)
    print(f"  {len(orders):,} orders from {orders['buyer_id'].nunique():,} buyers; "
          f"{orders['is_model_row'].sum():,} in the model window")

    print("Scoring and calibrating ...")
    s_wont, s_cant, s_log = score_orders(cfg, rng, orders)
    mask = orders["is_model_row"].to_numpy() == 1
    cal = cfg["calibration"]
    intercepts = {
        "wont": calibrate(s_wont, mask, cal["target_wont"]),
        "cant": calibrate(s_cant, mask, cal["target_cant"]),
        "logistics": calibrate(s_log, mask, cal["target_logistics"]),
    }
    p_wont = sigmoid(s_wont + intercepts["wont"])
    p_cant = sigmoid(s_cant + intercepts["cant"])
    p_log = sigmoid(s_log + intercepts["logistics"])

    # Cap the total chance (almost never needed), scaling all three down together.
    total = p_wont + p_cant + p_log
    scale = np.minimum(1.0, cfg["effects"]["max_total_failure_prob"] / total)
    p_wont, p_cant, p_log = p_wont * scale, p_cant * scale, p_log * scale

    # Draw ONE outcome per order at random from the three chances.
    u = rng.random(len(orders))
    orders["failure_type"] = np.select(
        [u < p_wont, u < p_wont + p_cant, u < p_wont + p_cant + p_log],
        ["wont", "cant", "logistics"], "none")
    orders["p_wont"], orders["p_cant"], orders["p_logistics"] = p_wont, p_cant, p_log

    # When the outcome becomes known (used to keep history features honest).
    orders["outcome_known_day"] = np.round(orders["order_day"] + orders["expected_days_to_delivery"]
                                          + rng.uniform(0, cal["resolution_extra_days_max"], len(orders)), 5)
    orders["outcome_known_datetime"] = days_to_datetime(
        orders["outcome_known_day"].to_numpy(), sim_start).strftime("%Y-%m-%d %H:%M:%S")
    orders["signup_date"] = days_to_datetime(orders["signup_day"].to_numpy(), sim_start).strftime("%Y-%m-%d")
    orders["order_id"] = np.arange(1, len(orders) + 1)

    # Summary for the hidden-rules page.
    m = orders[mask]
    rows = []
    for t, p in [("wont", p_wont), ("cant", p_cant), ("logistics", p_log)]:
        rows.append(f"| `{t}` | {cal['target_' + t]:.2%} | {p[mask].mean():.2%} | {(m['failure_type'] == t).mean():.2%} |")
    write_hidden_rules(cfg, intercepts, "\n".join(rows))

    cols = [
        "order_id", "buyer_id", "area_id", "order_datetime", "order_day", "order_month", "order_hour",
        "signup_date", "signup_day", "address_type", "order_value", "is_campaign_day",
        "expected_days_to_delivery", "same_item_other_shops_48h", "account_age_days",
        "outcome_known_datetime", "outcome_known_day", "is_model_row",
        # ---- hidden / audit-only columns below ----
        "latent_aov", "latent_wont_base", "latent_wont", "latent_cant", "latent_area_logistics",
        "drift_direction", "is_regular", "is_spike_order", "p_wont", "p_cant", "p_logistics", "failure_type",
    ]
    out = orders[cols].copy()
    for c in ["latent_wont_base", "latent_wont", "latent_cant", "latent_area_logistics", "p_wont", "p_cant", "p_logistics"]:
        out[c] = out[c].round(6)
    write_csv(out, DATA_DIR / "raw_events.csv")

    print(f"  intercepts: " + ", ".join(f"{k} {v:+.3f}" for k, v in intercepts.items()))
    print(f"  model-window failure rate: {(m['failure_type'] != 'none').mean():.2%}")
    print(f"  failure mix: " + ", ".join(
        f"{k} {v:.1%}" for k, v in m.loc[m['failure_type'] != 'none', 'failure_type'].value_counts(normalize=True).items()))
    print("Wrote data/raw_events.csv and docs/hidden_rules.md")


if __name__ == "__main__":
    main()
