"""Step 2: build as-of-time features, labels and the time split.

SYNTHETIC DATA - for illustration only. This is NOT real Shopee data.

Every feature for an order uses only information available when that order was
placed:
  * order-context features (value, frequency) use earlier PLACED orders;
  * history features (refusals, misses, courier failures) use earlier orders
    whose OUTCOME WAS ALREADY KNOWN at order time (delivered or failed before
    this checkout). An order placed yesterday that is still on the truck does
    not count yet.

Output: data/model_table.csv, data/train.csv, data/test.csv, docs/data_dictionary.md
"""
import numpy as np
import pandas as pd

from src.common import (
    DATA_DIR, DOCS_DIR, SYNTHETIC_NOTE, day_number, last_time_before,
    load_config, order_context, read_csv, sum_before, write_csv, write_text,
)

ID_COLS = ["order_id", "buyer_id", "order_datetime"]  # kept for splitting / audit, NOT features
LABEL_COLS = ["label_failed", "label_buyer_caused"]
FEATURE_COLS = [
    # A. buyer COD history
    "hist_cod_orders", "hist_refusals", "refusal_rate_smoothed", "recent_refusal_rate_smoothed_90d",
    "days_since_last_refusal", "has_ever_refused", "hist_buyer_caused_misses", "hist_courier_caused_failures",
    # B. order context
    "order_value", "value_vs_aov", "freq_change_ratio", "account_age_days", "is_new_cod_buyer",
    "is_late_night", "same_item_other_shops_48h", "is_campaign_day", "expected_days_to_delivery",
    # C. logistics context
    "area_logistics_failure_rate", "address_is_condo_with_office", "order_month",
]


def smoothed_rate(events, exposure, prior_rate, prior_strength):
    """Bayesian (Beta prior) smoothing: (events + a) / (exposure + a + b),
    where a = prior_rate * strength and b = (1 - prior_rate) * strength."""
    a = prior_rate * prior_strength
    b = (1 - prior_rate) * prior_strength
    return (events + a) / (exposure + a + b)


def build(cfg, raw):
    fc = cfg["features"]
    sim_start = cfg["dates"]["sim_start"]

    # ---- B. order context from placed orders (all 18 months feed the history) ----
    ctx = order_context(raw, cfg)
    raw = raw.assign(value_vs_aov=ctx["value_vs_aov"], freq_change_ratio=ctx["freq_change_ratio"])

    # Only orders in the model window become rows; all orders are history.
    rows = raw[raw["is_model_row"] == 1].copy()
    q_b, q_t = rows["buyer_id"].to_numpy(), rows["order_day"].to_numpy()

    # ---- A. buyer COD history from orders whose outcome was known before this order ----
    ref_b, ref_t = raw["buyer_id"].to_numpy(), raw["outcome_known_day"].to_numpy()
    ft = raw["failure_type"].to_numpy()
    flags = {"wont": (ft == "wont").astype(float), "cant": (ft == "cant").astype(float),
             "logistics": (ft == "logistics").astype(float)}

    life = sum_before(ref_b, ref_t, q_b, q_t, values=flags)
    recent = sum_before(ref_b, ref_t, q_b, q_t, values={"wont": flags["wont"]}, window=fc["recent_window_days"])

    rows["hist_cod_orders"] = life["count"]
    rows["hist_refusals"] = life["wont"].astype(int)
    rows["hist_buyer_caused_misses"] = life["cant"].astype(int)
    rows["hist_courier_caused_failures"] = life["logistics"].astype(int)
    # Courier failures are NOT refusals, so they never enter the refusal rates below.
    rows["refusal_rate_smoothed"] = smoothed_rate(
        life["wont"], life["count"], fc["refusal_prior_rate"], fc["refusal_prior_strength"])
    rows["recent_refusal_rate_smoothed_90d"] = smoothed_rate(
        recent["wont"], recent["count"], fc["refusal_prior_rate"], fc["refusal_prior_strength"])

    is_wont = ft == "wont"
    last_ref = last_time_before(ref_b[is_wont], ref_t[is_wont], q_b, q_t)
    rows["has_ever_refused"] = (~np.isnan(last_ref)).astype(int)
    rows["days_since_last_refusal"] = np.where(np.isnan(last_ref), -1, np.floor(q_t - np.nan_to_num(last_ref))).astype(int)

    # ---- B. rest of order context ----
    rows["is_new_cod_buyer"] = (rows["hist_cod_orders"] == 0).astype(int)
    rows["is_late_night"] = rows["order_hour"].between(1, 4).astype(int)

    # ---- C. logistics context: ONLY logistics failures in the area, last 90 days ----
    area = sum_before(raw["area_id"].to_numpy(), ref_t, rows["area_id"].to_numpy(), q_t,
                      values={"logistics": flags["logistics"]}, window=fc["area_window_days"])
    rows["area_logistics_failure_rate"] = smoothed_rate(
        area["logistics"], area["count"], fc["area_prior_rate"], fc["area_prior_strength"])
    rows["address_is_condo_with_office"] = (rows["address_type"] == "condo_with_office").astype(int)

    # ---- Labels ----
    rows["label_failed"] = (rows["failure_type"] != "none").astype(int)
    rows["label_buyer_caused"] = rows["failure_type"].isin(["wont", "cant"]).astype(int)

    # ---- Time split ----
    test_day = day_number(cfg["dates"]["test_start"], sim_start)
    rows["split"] = np.where(rows["order_day"] >= test_day, "test", "train")

    for c in ["refusal_rate_smoothed", "recent_refusal_rate_smoothed_90d", "area_logistics_failure_rate"]:
        rows[c] = rows[c].round(6)
    return rows[ID_COLS + FEATURE_COLS + LABEL_COLS + ["split"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Data dictionary (ranges are filled in from the actual data)
# ---------------------------------------------------------------------------
DESCRIPTIONS = {
    "order_id": ("ID", "Unique order id. Join key to raw_events.csv."),
    "buyer_id": ("ID", "Buyer id. Kept only for splitting and audit. Never a feature."),
    "order_datetime": ("ID", "When the order was placed (checkout time)."),
    "hist_cod_orders": ("feature", "Past COD orders whose outcome was known before this order. Exposure (how much evidence we have), not risk."),
    "hist_refusals": ("feature", "Past `wont` failures (buyer refused at the door). Raw count, kept for audit and for the model."),
    "refusal_rate_smoothed": ("feature", "Smoothed lifetime refusal rate, see note below."),
    "recent_refusal_rate_smoothed_90d": ("feature", "Same smoothing, using only orders whose outcome became known in the last 90 days."),
    "days_since_last_refusal": ("feature", "Days since the last refusal became known. -1 if the buyer never refused (see `has_ever_refused`)."),
    "has_ever_refused": ("feature", "1 if the buyer has at least one known past refusal."),
    "hist_buyer_caused_misses": ("feature", "Past `cant` failures (not home, no cash)."),
    "hist_courier_caused_failures": ("feature", "Past `logistics` failures. Courier's fault, so NOT counted in any refusal feature."),
    "order_value": ("feature", "Order value in THB."),
    "value_vs_aov": ("feature", "order_value / average value of the buyer's earlier placed COD orders. 1.0 if there are no earlier orders."),
    "freq_change_ratio": ("feature", "COD orders placed in the last 30 days / buyer's smoothed average monthly COD orders."),
    "account_age_days": ("feature", "Days since signup."),
    "is_new_cod_buyer": ("feature", "1 if hist_cod_orders == 0."),
    "is_late_night": ("feature", "1 if placed 01:00 to 04:59."),
    "same_item_other_shops_48h": ("feature", "Similar COD orders at other shops in the last 48 hours."),
    "is_campaign_day": ("feature", "1 on double-day campaigns (1.1, 2.2, ..., 11.11, 12.12)."),
    "expected_days_to_delivery": ("feature", "Promised delivery time in days, shown at checkout."),
    "area_logistics_failure_rate": ("feature", "Smoothed rate of LOGISTICS failures in the buyer's area over the 90 days before the order. Refusals are not included, so poorly served areas are not labelled as bad buyers."),
    "address_is_condo_with_office": ("feature", "1 if the address is a condo with a front office that can receive parcels."),
    "order_month": ("feature", "Month of the order, 1 to 12 (Songkran in April, December holidays)."),
    "label_failed": ("label (main)", "1 if the order failed for any reason (matches the case's failed delivery rate)."),
    "label_buyer_caused": ("label (secondary)", "1 if failure_type is `wont` or `cant`."),
    "split": ("split", "`train` = Feb to Jun 2026, `test` = Jul 2026."),
}

RAW_ONLY = {
    "failure_type": "`wont`, `cant`, `logistics` or `none`. The outcome itself, audit only.",
    "latent_wont / latent_wont_base / latent_cant / latent_area_logistics / latent_aov": "Hidden traits of the generator. Never features.",
    "p_wont / p_cant / p_logistics": "True chances used to draw the outcome. Never features.",
    "drift_direction, is_regular, is_spike_order": "Generator bookkeeping for audit.",
    "outcome_known_datetime / outcome_known_day": "When the outcome became known. Used only to decide what counts as history.",
    "signup_date, address_type, area_id, order_hour, order_day, signup_day": "Raw inputs that the features above are built from.",
}


def column_range(s):
    if not pd.api.types.is_numeric_dtype(s):
        vals = s.dropna().unique()
        if len(vals) > 4:
            return f"{s.min()} to {s.max()}"
        return ", ".join(sorted(map(str, vals)))
    lo, hi = s.min(), s.max()
    fmt = (lambda x: f"{x:,.0f}") if pd.api.types.is_integer_dtype(s) else (lambda x: f"{x:,.4g}")
    return f"{fmt(lo)} to {fmt(hi)}"


def write_dictionary(cfg, table):
    fc = cfg["features"]
    a = fc["refusal_prior_rate"] * fc["refusal_prior_strength"]
    b = (1 - fc["refusal_prior_rate"]) * fc["refusal_prior_strength"]
    lines = [
        "# Data dictionary", "", f"> {SYNTHETIC_NOTE}", "",
        "## `data/model_table.csv`, `data/train.csv`, `data/test.csv`", "",
        "One row per COD order placed Feb to Jul 2026. All features are computed as of order time.", "",
        "| Column | Role | Type | Range in data | Meaning |", "| --- | --- | --- | --- | --- |",
    ]
    for col in table.columns:
        role, desc = DESCRIPTIONS[col]
        dtype = "int" if pd.api.types.is_integer_dtype(table[col]) else (
            "float" if pd.api.types.is_float_dtype(table[col]) else "text")
        lines.append(f"| `{col}` | {role} | {dtype} | {column_range(table[col])} | {desc} |")

    lines += [
        "", "## Why the refusal rate is smoothed", "",
        f"`refusal_rate_smoothed = (hist_refusals + a) / (hist_cod_orders + a + b)` with "
        f"a = {a:.2f} and b = {b:.2f}.", "",
        f"This is a Beta prior centred on the population COD failure rate ({fc['refusal_prior_rate']:.1%}), "
        f"worth about {fc['refusal_prior_strength']} orders of evidence. We use the overall COD failure rate "
        "from the case as the prior, as the brief asks; the refusal-only rate (about 1.7%) would work almost the same.",
        "",
        "- A buyer with **no history** gets the prior (2.6%), not 0% and not 100%.",
        f"- **1 order, 1 refusal** gives (1 + {a:.2f}) / (1 + 10) = {(1 + a) / (1 + a + b):.1%}: suspicious, but not proven.",
        f"- **40 orders, 15 refusals** gives (15 + {a:.2f}) / (40 + 10) = {(15 + a) / (40 + a + b):.1%}: strong evidence.",
        "",
        "So more exposure makes us more confident, and one unlucky order does not brand a new buyer. "
        "`hist_cod_orders` is kept as a separate feature so the model can also see how much evidence there is.",
        "",
        f"`area_logistics_failure_rate` uses the same idea with prior {fc['area_prior_rate']:.2%} and strength "
        f"{fc['area_prior_strength']} orders, so small samples in an area do not swing the rate.",
        "",
        "## Timing rules (no leakage)", "",
        "- Order-context features (`value_vs_aov`, `freq_change_ratio`) use earlier **placed** orders: "
        "their values are known at checkout.",
        "- History features use earlier orders whose **outcome was known** before this checkout "
        "(order time + delivery days). An undelivered earlier order is not counted yet, so a buyer "
        "can still be `is_new_cod_buyer = 1` while `value_vs_aov` already uses the first pending order.",
        "- History starts on 2025-02-01 (the simulation start). The first 12 months only build history.",
        "",
        "## Audit-only columns (in `data/raw_events.csv`, never features)", "",
        "| Column(s) | Meaning |", "| --- | --- |",
    ]
    lines += [f"| {k} | {v} |" for k, v in RAW_ONLY.items()]
    lines += ["", "No protected traits (gender, age, religion, ethnicity, ...) exist in the data.", ""]
    write_text(DOCS_DIR / "data_dictionary.md", "\n".join(lines))


def main():
    cfg = load_config()
    print("Reading data/raw_events.csv ...")
    raw = read_csv(DATA_DIR / "raw_events.csv")
    print("Building as-of-time features ...")
    table = build(cfg, raw)

    write_csv(table, DATA_DIR / "model_table.csv")
    write_csv(table[table["split"] == "train"], DATA_DIR / "train.csv")
    write_csv(table[table["split"] == "test"], DATA_DIR / "test.csv")
    write_dictionary(cfg, table)

    counts = table["split"].value_counts()
    print(f"  model rows: {len(table):,} (train {counts.get('train', 0):,}, test {counts.get('test', 0):,})")
    print(f"  failure rate: {table['label_failed'].mean():.2%}; new COD buyers: {table['is_new_cod_buyer'].mean():.1%}")
    print("Wrote data/model_table.csv, data/train.csv, data/test.csv, docs/data_dictionary.md")


if __name__ == "__main__":
    main()
