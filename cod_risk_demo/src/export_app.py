"""Step 10: export what the Streamlit app needs.

SYNTHETIC DATA - for illustration only. This is NOT real Shopee data.

The Streamlit app (app.py at the repo root, package `risk_ui/`) does not load
the joblib models: both projects have a package called `src`, so unpickling
would clash. Instead we export:

  * app_data/model_params.json  - the recommended Logistic Regression as plain
    numbers (scaler, coefficients, Platt calibration, SHAP background mean),
    plus the feature settings, tier cut-offs and money assumptions.
  * app_data/orders.csv         - all 76,232 model rows (Feb-Jul 2026) with the
    calibrated score, tier, true outcome (audit), and the RAW history counts
    the app's input form needs to rebuild each order's features.

Then we CHECK that the app's own code (risk_ui.model) rebuilds every feature
and every score exactly from those raw inputs.
"""
import sys

import joblib
import numpy as np
import pandas as pd

from src.common import DATA_DIR, ROOT, load_config, read_csv, sum_before, write_csv
from src.decide import assign_tiers
from src.modeling import FEATURE_COLS, MODELS_DIR, read_json, write_json

APP_DIR = ROOT / "app_data"


def raw_inputs(cfg, raw, rows):
    """The raw counts behind each row's features (the same maths as build_features)."""
    fc = cfg["features"]
    q_b, q_t = rows["buyer_id"].to_numpy(), rows["order_day"].to_numpy()
    b, t, v = raw["buyer_id"].to_numpy(), raw["order_day"].to_numpy(), raw["order_value"].to_numpy()

    # Placed orders before this one (known at checkout).
    placed = sum_before(b, t, q_b, q_t, values={"value": v})
    last30 = sum_before(b, t, q_b, q_t, window=fc["freq_window_days"])["count"]
    first_day = raw.groupby("buyer_id")["order_day"].min()
    days_since_first = q_t - rows["buyer_id"].map(first_day).to_numpy()

    # Orders whose outcome was known in the last 90 days.
    known_t = raw["outcome_known_day"].to_numpy()
    wont = (raw["failure_type"] == "wont").to_numpy().astype(float)
    recent = sum_before(b, known_t, q_b, q_t, values={"wont": wont}, window=fc["recent_window_days"])

    n_placed = placed["count"]
    usual = np.divide(placed["value"], n_placed, out=np.zeros(len(rows)), where=n_placed > 0)
    return pd.DataFrame({
        "orders_in_transit": n_placed - rows["hist_cod_orders"].to_numpy(),
        "usual_order_value": np.round(usual, 6),
        "orders_last_30d": last30,
        "days_since_first_cod": np.round(days_since_first, 6),
        "orders_last_90d": recent["count"],
        "refusals_last_90d": recent["wont"].astype(int),
    }, index=rows.index)


def main():
    cfg = load_config()
    rec = read_json(MODELS_DIR / "recommended.json")
    model = joblib.load(MODELS_DIR / "logreg.joblib")
    assert model.method == "platt", "the app re-implements Platt calibration only"
    if rec["model"] != "logreg":
        print("NOTE: LightGBM is recommended, but the app uses Logistic Regression "
              "(a tree model cannot be re-implemented as plain numbers).")

    table = read_csv(DATA_DIR / "model_table.csv")
    raw = read_csv(DATA_DIR / "raw_events.csv")
    extra = raw[["order_id", "order_day", "order_hour", "area_id", "address_type", "failure_type",
                 "p_wont", "p_cant", "p_logistics"]]
    rows = table.merge(extra, on="order_id", how="left")
    rows = pd.concat([rows, raw_inputs(cfg, raw, rows)], axis=1)

    # ---- Model as plain numbers ----
    scaler, clf = model.model[0], model.model[-1]
    train = read_csv(DATA_DIR / "train.csv").sort_values("order_datetime").reset_index(drop=True)
    background = train.sample(2000, random_state=cfg["seed"])  # same background as src/explain.py
    bg_mean_scaled = scaler.transform(background[FEATURE_COLS]).mean(axis=0)
    params = {
        "note": "Synthetic data, for illustration only. Not Shopee data.",
        "model": "Logistic Regression (calibrated)",
        "features": FEATURE_COLS,
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
        "platt_a": float(model.calibrator.coef_[0][0]), "platt_b": float(model.calibrator.intercept_[0]),
        "background_mean_scaled": bg_mean_scaled.tolist(),
        "feature_settings": cfg["features"],
        "decision": cfg["decision"],
        "area_latest_rate": (rows.sort_values("order_datetime").groupby("area_id")["area_logistics_failure_rate"]
                             .last().round(6).to_dict()),
        "population_failure_rate": float(rows["label_failed"].mean()),
    }
    APP_DIR.mkdir(exist_ok=True)
    write_json(APP_DIR / "model_params.json", {**params, "area_latest_rate": {
        str(k): v for k, v in params["area_latest_rate"].items()}})

    # ---- Scores + tiers for every row ----
    rows["risk"] = model.predict(rows).round(6)
    rows["tier"] = assign_tiers(rows["risk"].to_numpy(), cfg["decision"])
    rows["p_true"] = (rows["p_wont"] + rows["p_cant"] + rows["p_logistics"]).round(6)

    # ---- Check: the app's own code rebuilds features and scores exactly ----
    sys.path.insert(0, str(ROOT.parent))
    from risk_ui.model import Model, features_from_inputs  # noqa: E402

    app_model = Model(params)
    rebuilt = features_from_inputs(rows.rename(columns={
        "hist_cod_orders": "past_orders", "hist_refusals": "past_refusals",
        "hist_buyer_caused_misses": "past_not_home", "hist_courier_caused_failures": "past_courier_failures",
        "same_item_other_shops_48h": "same_item_other_shops", "address_is_condo_with_office": "condo_with_office",
        "expected_days_to_delivery": "delivery_days"}), params["feature_settings"])
    # Features are rounded (4-6 decimals); a value sitting exactly on a rounding
    # edge can land one unit apart, so allow one unit in the last decimal.
    edge_rows = 0
    for f in FEATURE_COLS:
        diff = np.abs(rebuilt[f].to_numpy(dtype=float) - rows[f].to_numpy(dtype=float))
        assert diff.max() < 1.5e-4, f"app rebuilds `{f}` differently (max gap {diff.max()})"
        edge_rows += int((diff > 1e-9).sum())
    same_features_gap = np.abs(app_model.predict(rows[FEATURE_COLS]) - model.predict(rows)).max()
    rebuilt_gap = np.abs(app_model.predict(rebuilt) - model.predict(rows)).max()
    assert same_features_gap < 1e-9, f"app score differs from the model (max gap {same_features_gap})"
    assert rebuilt_gap < 1e-5, f"score from rebuilt features differs (max gap {rebuilt_gap})"
    print(f"Checked {len(rows):,} orders: app score = model score (max gap {same_features_gap:.1e}); "
          f"score from app-rebuilt features max gap {rebuilt_gap:.1e} "
          f"({edge_rows} feature values differ by one rounding unit).")

    keep = (["order_id", "buyer_id", "order_datetime", "split", "order_hour", "area_id"] + FEATURE_COLS
            + ["orders_in_transit", "usual_order_value", "orders_last_30d", "days_since_first_cod",
               "orders_last_90d", "refusals_last_90d", "risk", "tier", "label_failed", "failure_type", "p_true"])
    write_csv(rows[keep], APP_DIR / "orders.csv")
    print(f"Wrote app_data/model_params.json and app_data/orders.csv ({len(rows):,} orders)")


if __name__ == "__main__":
    main()
