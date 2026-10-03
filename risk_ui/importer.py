"""Import an order file (CSV or Excel), check it, and score it with the app's model.

The file is one row per COD order with RAW checkout data (the same fields as the
"Score an order" form). The app builds the 20 model features itself with
features_from_inputs, so the scores match the training pipeline.
"""
import io

import numpy as np
import pandas as pd

from risk_ui.model import TIER_NAMES, assign_tiers, features_from_inputs

# (column, kind, required, description, example). kind: int, num, datetime, flag
TEMPLATE = [
    ("order_id", "int", True, "Unique order number.", 900001),
    ("buyer_id", "int", True, "Buyer number (used for search and grouping, not for scoring).", 1250),
    ("order_datetime", "datetime", True,
     "When the order was placed: 2026-07-15 14:30 or 15/07/2026 14:30 (day first). Buddhist-era years are converted.",
     "2026-07-15 14:30"),
    ("area_id", "int", True, "Delivery area number. 1-60 are the areas the model knows.", 12),
    ("order_value", "num", True, "Order value in THB (paid in cash at the door).", 450),
    ("same_item_other_shops", "int", True,
     "How many OTHER shops this buyer ordered the same item from with COD in the last 48 hours.", 0),
    ("delivery_days", "int", True, "Delivery time promised at checkout, in days (1-6).", 2),
    ("account_age_days", "int", True, "Days since the buyer's account was created.", 400),
    ("condo_with_office", "flag", True, "1 if building staff can receive the parcel, else 0.", 0),
    ("area_logistics_failure_rate", "num", False,
     "Share of the area's orders that failed because of the courier in the last 90 days (0.0015 = 0.15%). "
     "Leave blank to use the area's latest rate (areas 1-60 only).", 0.0015),
    ("past_orders", "int", True, "Earlier COD orders of this buyer whose result is already known. 0 = first COD order.", 12),
    ("past_refusals", "int", True, "Of those, how many the buyer refused at the door.", 0),
    ("days_since_last_refusal", "int", False, "Days since the last refusal. Required only if past_refusals > 0.", ""),
    ("past_not_home", "int", True, "Of those, how many failed because the buyer was out or had no cash.", 0),
    ("past_courier_failures", "int", True, "Of those, how many failed because of the courier.", 0),
    ("orders_last_90d", "int", True, "Earlier COD orders (known result) from the last 90 days.", 4),
    ("refusals_last_90d", "int", True, "Refusals among those last-90-day orders.", 0),
    ("usual_order_value", "num", True, "The buyer's typical order value in THB. 0 if they have never ordered.", 450),
    ("orders_in_transit", "int", True, "Earlier orders still on the way (result not known yet).", 0),
    ("orders_last_30d", "int", True, "COD orders placed in the last 30 days, including ones still on the way.", 1),
    ("days_since_first_cod", "num", True, "Days since the buyer's first COD order. 0 if this is the first.", 300),
]
COLUMNS = [c[0] for c in TEMPLATE]
KIND = {c[0]: c[1] for c in TEMPLATE}
REQUIRED = [c[0] for c in TEMPLATE if c[2]]
MAX_ERRORS_SHOWN = 500


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------
def demo_as_template(orders, n=None):
    """Turn the app's demo orders into template rows (used for the example file and for testing)."""
    o = orders if n is None else orders.head(n)
    return pd.DataFrame({
        "order_id": o["order_id"], "buyer_id": o["buyer_id"],
        "order_datetime": pd.to_datetime(o["order_datetime"]).dt.strftime("%Y-%m-%d %H:%M"),
        "area_id": o["area_id"], "order_value": o["order_value"],
        "same_item_other_shops": o["same_item_other_shops_48h"], "delivery_days": o["expected_days_to_delivery"],
        "account_age_days": o["account_age_days"], "condo_with_office": o["address_is_condo_with_office"],
        "area_logistics_failure_rate": o["area_logistics_failure_rate"],
        "past_orders": o["hist_cod_orders"], "past_refusals": o["hist_refusals"],
        "days_since_last_refusal": o["days_since_last_refusal"].where(o["days_since_last_refusal"] >= 0),
        "past_not_home": o["hist_buyer_caused_misses"], "past_courier_failures": o["hist_courier_caused_failures"],
        "orders_last_90d": o["orders_last_90d"], "refusals_last_90d": o["refusals_last_90d"],
        "usual_order_value": o["usual_order_value"], "orders_in_transit": o["orders_in_transit"],
        "orders_last_30d": o["orders_last_30d"], "days_since_first_cod": o["days_since_first_cod"],
    })[COLUMNS]


def column_guide():
    return pd.DataFrame([{"Column": c, "Required": "yes" if req else "no",
                          "Type": {"int": "whole number", "num": "number", "datetime": "date + time",
                                   "flag": "0 or 1"}[k], "Meaning": desc, "Example": str(ex)}
                         for c, k, req, desc, ex in TEMPLATE])


def template_csv(examples):
    return examples.to_csv(index=False).encode("utf-8-sig")


def template_excel(examples):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        examples.to_excel(xl, sheet_name="orders", index=False)
        column_guide().to_excel(xl, sheet_name="columns", index=False)
        for ws in xl.book.worksheets:  # readable column widths (from the first 50 rows)
            for col in ws.iter_cols(max_row=min(ws.max_row, 50)):
                ws.column_dimensions[col[0].column_letter].width = min(max(len(str(c.value or "")) for c in col) + 2, 60)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------
def read_file(name, data):
    """Read an uploaded CSV / Excel file into a DataFrame of strings-or-values. Raises ValueError."""
    lower = name.lower()
    if lower.endswith(".csv"):
        for enc in ("utf-8-sig", "cp874", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(data), comment="#", encoding=enc, dtype=str, skip_blank_lines=True)
                break
            except UnicodeDecodeError:
                continue
    elif lower.endswith((".xlsx", ".xlsm")):
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, engine="openpyxl")
        df = sheets.get("orders", next(iter(sheets.values())))
    else:
        raise ValueError("Please upload a .csv or .xlsx file.")
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("The file has no data rows.")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------
def _blank(s):
    return s.isna() | s.astype(str).str.strip().isin(["", "nan", "NaN", "None", "NaT"])


def check(raw, area_rates):
    """Check an imported table. Returns a dict with the parsed table, per-row problems and a summary.

    area_rates: {area_id (str): latest courier failure rate} used to fill a blank area rate.
    """
    missing_cols = [c for c in REQUIRED if c not in raw.columns]
    extra_cols = [c for c in raw.columns if c not in COLUMNS]
    n = len(raw)
    errors = {}    # check name -> boolean mask of rows that fail (row cannot be scored)
    warnings = {}  # check name -> mask of rows that are scored but worth a look

    def fail(name, mask):
        mask = pd.Series(mask, index=raw.index).fillna(False).astype(bool)
        if mask.any():
            errors[name] = errors.get(name, False) | mask

    def warn(name, mask):
        mask = pd.Series(mask, index=raw.index).fillna(False).astype(bool)
        if mask.any():
            warnings[name] = mask

    # ---- parse every known column; blank vs. unreadable are separate problems ----
    t = pd.DataFrame(index=raw.index)
    for col in COLUMNS:
        if col not in raw.columns:
            t[col] = np.nan
            continue
        s = raw[col]
        blank = _blank(s)
        if KIND[col] == "datetime":
            # ISO first (2026-07-05 = 5 July); anything else is read day-first like Thai usage (05/07/2026 = 5 July).
            # (dayfirst would also flip ISO dates, so the two are parsed separately.)
            given = s.where(~blank)
            v = pd.to_datetime(given, errors="coerce", format="ISO8601") if not pd.api.types.is_datetime64_any_dtype(s) \
                else given
            rest = v.isna() & ~blank
            if rest.any():
                v = v.where(~rest, pd.to_datetime(given[rest], errors="coerce", format="mixed", dayfirst=True))
            be = v.dt.year > 2400  # Buddhist-era year (e.g. 2569) -> CE
            if be.any():
                v = v.where(~be, v - pd.DateOffset(years=543))
                warn("Buddhist-era year in order_datetime, converted to CE (minus 543)", be)
        else:
            v = pd.to_numeric(s, errors="coerce") if pd.api.types.is_numeric_dtype(s) else pd.to_numeric(
                s.where(~blank).astype(str).str.replace(",", "").str.strip(), errors="coerce")  # "1,200" -> 1200
        t[col] = v
        if col in REQUIRED:
            fail(f"Missing value: {col}", blank)
        fail(f"Not readable: {col} (expected {'a date' if KIND[col] == 'datetime' else 'a number'})", ~blank & v.isna())
        if KIND[col] in ("int", "flag"):
            fail(f"Not a whole number: {col}", v.notna() & (v % 1 != 0))

    if not missing_cols:
        # ---- ranges ----
        counts = ["same_item_other_shops", "account_age_days", "past_orders", "past_refusals", "past_not_home",
                  "past_courier_failures", "orders_last_90d", "refusals_last_90d", "orders_in_transit",
                  "orders_last_30d", "days_since_last_refusal", "usual_order_value", "days_since_first_cod"]
        for c in counts:
            fail(f"Negative value: {c}", t[c] < 0)
        fail("order_value must be more than 0", t["order_value"] <= 0)
        fail("delivery_days must be 1 or more", t["delivery_days"] < 1)
        fail("condo_with_office must be 0 or 1", t["condo_with_office"].notna() & ~t["condo_with_office"].isin([0, 1]))
        fail("area_logistics_failure_rate must be between 0 and 1",
             (t["area_logistics_failure_rate"] < 0) | (t["area_logistics_failure_rate"] > 1))
        known_area = t["area_id"].isin([int(a) for a in area_rates])
        fail("Blank area_logistics_failure_rate for an area outside 1-60 (no rate to fill in)",
             t["area_logistics_failure_rate"].isna() & t["area_id"].notna() & ~known_area)
        fail("Duplicate order_id", t["order_id"].notna() & t["order_id"].duplicated(keep="first"))

        # ---- the history must add up (same rules as the Score an order form) ----
        past, ref, ref90, o90 = t["past_orders"], t["past_refusals"], t["refusals_last_90d"], t["orders_last_90d"]
        last_ref, earlier = t["days_since_last_refusal"], t["past_orders"] + t["orders_in_transit"]
        fail("past_refusals + past_not_home + past_courier_failures is more than past_orders",
             ref + t["past_not_home"] + t["past_courier_failures"] > past)
        fail("orders_last_90d is more than past_orders", o90 > past)
        fail("refusals_last_90d is more than past_refusals or orders_last_90d", ref90 > np.minimum(ref, o90))
        fail("days_since_last_refusal is required when past_refusals > 0", (ref > 0) & last_ref.isna())
        fail("refusals_last_90d > 0 but the last refusal was over 90 days ago", (ref90 > 0) & (last_ref > 90))
        fail("Last refusal under 90 days ago but refusals_last_90d is 0", (ref > 0) & (ref90 == 0) & (last_ref < 90))
        fail("orders_last_30d is more than past_orders + orders_in_transit", t["orders_last_30d"] > earlier)
        fail("No earlier orders, but usual_order_value or days_since_first_cod is not 0",
             (earlier == 0) & ((t["usual_order_value"] > 0) | (t["days_since_first_cod"] > 0)))
        fail("Earlier orders, but usual_order_value is 0", (earlier > 0) & (t["usual_order_value"] == 0))
        fail("First COD order is before the account was created", t["days_since_first_cod"] > t["account_age_days"] + 1)

        # ---- scored, but outside what the model saw ----
        when = t["order_datetime"]
        warn("Order date outside Feb-Jul 2026 (the model only saw these months)",
             when.notna() & ((when.dt.year != 2026) | ~when.dt.month.between(2, 7)))
        warn("same_item_other_shops above 3 (the training data went up to 3)", t["same_item_other_shops"] > 3)
        warn("delivery_days above 6 (the training data went up to 6)", t["delivery_days"] > 6)
        warn("Area outside 1-60, scored with the courier failure rate given in the file", t["area_id"].notna() & ~known_area)
        warn("Blank area_logistics_failure_rate, filled with the area's latest rate",
             t["area_logistics_failure_rate"].isna() & known_area)

    bad = pd.Series(False, index=raw.index)
    for m in errors.values():
        bad |= m
    if missing_cols:
        bad[:] = True
    return {"table": t, "raw": raw, "rows": n, "bad": bad, "errors": errors, "warnings": warnings,
            "missing_cols": missing_cols, "extra_cols": extra_cols}


def problem_rows(report):
    """One line per failing row with all its reasons (first rows of the file only), for display / download."""
    raw, errors = report["raw"], report["errors"]
    idx = report["bad"][report["bad"]].index
    reasons = pd.Series("", index=idx)
    for name, mask in errors.items():
        hit = mask.reindex(idx, fill_value=False)
        reasons[hit] = reasons[hit].where(reasons[hit] == "", reasons[hit] + "; ") + name
    out = raw.loc[idx].copy()
    out.insert(0, "problem", reasons)
    out.insert(0, "file_row", idx + 2)  # +1 for the header, +1 for 1-based rows
    return out


def summary_table(report):
    rows = [{"Check": k, "Rows": int(v.sum()), "Effect": "row not imported"} for k, v in report["errors"].items()]
    rows += [{"Check": k, "Rows": int(v.sum()), "Effect": "imported, worth a look"} for k, v in report["warnings"].items()]
    return pd.DataFrame(rows, columns=["Check", "Rows", "Effect"]).sort_values("Rows", ascending=False)


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------
def score(report, params, model):
    """Score the rows that passed the checks. Returns a table with the same columns the app pages use."""
    t = report["table"][~report["bad"]].copy()
    rates = {int(k): v for k, v in params["area_latest_rate"].items()}
    t["area_logistics_failure_rate"] = t["area_logistics_failure_rate"].fillna(t["area_id"].map(rates))
    t["days_since_last_refusal"] = t["days_since_last_refusal"].fillna(0)
    raw = pd.DataFrame({
        "past_orders": t["past_orders"], "past_refusals": t["past_refusals"], "orders_last_90d": t["orders_last_90d"],
        "refusals_last_90d": t["refusals_last_90d"], "days_since_last_refusal": t["days_since_last_refusal"],
        "past_not_home": t["past_not_home"], "past_courier_failures": t["past_courier_failures"],
        "order_value": t["order_value"], "usual_order_value": t["usual_order_value"],
        "orders_in_transit": t["orders_in_transit"], "orders_last_30d": t["orders_last_30d"],
        "days_since_first_cod": t["days_since_first_cod"], "account_age_days": t["account_age_days"],
        "order_datetime": t["order_datetime"], "order_hour": t["order_datetime"].dt.hour,
        "same_item_other_shops": t["same_item_other_shops"], "delivery_days": t["delivery_days"],
        "condo_with_office": t["condo_with_office"], "area_logistics_failure_rate": t["area_logistics_failure_rate"],
    })
    feats = features_from_inputs(raw, params["feature_settings"])
    out = pd.concat([
        pd.DataFrame({"order_id": t["order_id"].astype("int64"), "buyer_id": t["buyer_id"].astype("int64"),
                      "order_datetime": t["order_datetime"], "split": "imported",
                      "order_hour": raw["order_hour"].astype(int), "area_id": t["area_id"].astype(int)}),
        feats,
        pd.DataFrame({"orders_in_transit": t["orders_in_transit"].astype(int),
                      "usual_order_value": t["usual_order_value"], "orders_last_30d": t["orders_last_30d"].astype(int),
                      "days_since_first_cod": t["days_since_first_cod"], "orders_last_90d": t["orders_last_90d"].astype(int),
                      "refusals_last_90d": t["refusals_last_90d"].astype(int)}),
    ], axis=1)
    out["order_value"] = out["order_value"].astype(float)
    out["risk"] = model.predict(out)
    out["tier"] = assign_tiers(out["risk"].to_numpy())
    out["tier_name"] = out["tier"].map(TIER_NAMES)
    out["risk_pct"] = out["risk"] * 100
    return out.reset_index(drop=True)
