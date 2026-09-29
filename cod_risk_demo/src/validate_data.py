"""Step 3: check the synthetic data against the brief and write the report.

SYNTHETIC DATA - for illustration only. This is NOT real Shopee data.

Runs every check from section 7 of the brief, counts the 8 edge cases,
fits a small sanity-check logistic regression (NOT the final model), and
writes reports/validation_report.md.
"""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.build_features import FEATURE_COLS, ID_COLS, LABEL_COLS
from src.common import DATA_DIR, REPORTS_DIR, SYNTHETIC_NOTE, load_config, read_csv, write_text

FORBIDDEN = ["failure_type", "p_wont", "p_cant", "p_logistics", "buyer_id", "outcome_known_datetime",
             "outcome_known_day", "drift_direction", "is_spike_order", "is_regular"]

# Rough expected importance order from the brief (group -> features in it).
EXPECTED_ORDER = [
    ("refusal history", ["hist_refusals", "refusal_rate_smoothed", "has_ever_refused"]),
    ("recent refusal rate", ["recent_refusal_rate_smoothed_90d"]),
    ("days since last refusal", ["days_since_last_refusal"]),
    ("value vs usual / same item / late night", ["value_vs_aov", "same_item_other_shops_48h", "is_late_night"]),
    ("frequency change", ["freq_change_ratio"]),
    ("area logistics / courier failures", ["area_logistics_failure_rate", "hist_courier_caused_failures"]),
]


def pf(ok):
    return "PASS" if ok else "FAIL"


def auc_either_way(y, x):
    """AUC of one feature used as a score, in whichever direction is better."""
    a = roc_auc_score(y, x)
    return max(a, 1 - a)


def main():
    cfg = load_config()
    v = cfg["validation"]
    table = read_csv(DATA_DIR / "model_table.csv")
    raw = read_csv(DATA_DIR / "raw_events.csv")
    audit = raw[["order_id", "failure_type", "p_wont", "p_cant", "p_logistics", "drift_direction",
                 "latent_area_logistics"]]
    df = table.merge(audit, on="order_id", how="left")
    df["p_true"] = df["p_wont"] + df["p_cant"] + df["p_logistics"]
    y = df["label_failed"]

    checks = []   # (name, target, result text, pass?)
    lines = []    # extra report sections

    # 1. Failure rate
    rate = y.mean()
    lo, hi = v["failure_rate_range"]
    checks.append(("COD failure rate, model rows", f"{lo:.1%} to {hi:.1%}", f"{rate:.2%}", lo <= rate <= hi))

    # 2. Failure-type mix
    mix = df.loc[y == 1, "failure_type"].value_counts(normalize=True)
    mix_ok = all(abs(mix.get(k, 0) - t) <= v["mix_tolerance"] for k, t in v["mix_targets"].items())
    checks.append(("Failure-type mix (wont / cant / logistics)",
                   " / ".join(f"{t:.0%}" for t in v["mix_targets"].values()) + f" (+/-{v['mix_tolerance']*100:.0f} pts)",
                   " / ".join(f"{mix.get(k, 0):.1%}" for k in v["mix_targets"]), mix_ok))

    # 3. Top 30% of true risk holds what share of failures?
    cut = df["p_true"].quantile(1 - v["top_share"])
    capture = y[df["p_true"] >= cut].sum() / y.sum()
    lo, hi = v["top_share_capture_range"]
    checks.append((f"Top {v['top_share']:.0%} of true risk holds this share of failures",
                   f"{lo:.0%} to {hi:.0%}", f"{capture:.1%}", lo <= capture <= hi))

    # 4. New COD buyers vs average
    new_rate = y[df["is_new_cod_buyer"] == 1].mean()
    ratio = new_rate / rate
    lo, hi = v["new_buyer_ratio_range"]
    checks.append(("New COD buyers: failure rate vs average", f"{lo} to {hi} times",
                   f"{ratio:.2f}x ({new_rate:.2%} vs {rate:.2%})", lo <= ratio <= hi))

    # 5. Sanity model: logistic regression on features only, train -> test
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(train[FEATURE_COLS], train["label_failed"])
    test_auc = roc_auc_score(test["label_failed"], model.predict_proba(test[FEATURE_COLS])[:, 1])
    lo, hi = v["sanity_auc_range"]
    checks.append(("Sanity model (logistic regression) AUC, test set", f"{lo} to {hi}",
                   f"{test_auc:.3f}", lo <= test_auc <= hi))

    # 6. Best single feature AUC (all model rows)
    single = pd.Series({f: auc_either_way(y, df[f]) for f in FEATURE_COLS}).sort_values(ascending=False)
    checks.append(("Best single feature AUC", f"below {v['best_single_feature_auc_max']}",
                   f"{single.iloc[0]:.3f} (`{single.index[0]}`)", single.iloc[0] < v["best_single_feature_auc_max"]))

    # 7. Smoothing check + 8. edge cases
    raw_rate = np.where(df["hist_cod_orders"] > 0, df["hist_refusals"] / df["hist_cod_orders"].clip(lower=1), 0)
    low_exp = (df["hist_cod_orders"].between(1, v["low_exposure_max_orders"])) & (df["hist_refusals"] == df["hist_cod_orders"])
    one_one = (df["hist_cod_orders"] == 1) & (df["hist_refusals"] == 1)
    high_exp = (df["hist_cod_orders"] >= v["high_exposure_min_orders"]) & (raw_rate >= v["high_exposure_min_raw_rate"])
    drift = ((df["refusal_rate_smoothed"] <= v["drift_lifetime_max"])
             & (df["recent_refusal_rate_smoothed_90d"] >= v["drift_recent_min"]))
    high_area = df["area_logistics_failure_rate"] >= df["area_logistics_failure_rate"].quantile(v["high_area_quantile"])
    clean = ((df["hist_cod_orders"] >= v["clean_buyer_min_orders"]) & (df["hist_refusals"] == 0)
             & (df["hist_buyer_caused_misses"] == 0))
    area_clean = high_area & clean

    smooth_a, smooth_b = df.loc[one_one, "p_true"].mean(), df.loc[high_exp, "p_true"].mean()
    smooth_ok = one_one.sum() > 0 and high_exp.sum() > 0 and smooth_a < smooth_b
    checks.append(("Smoothing check: '1 order, 1 refusal' less risky than high-exposure refusers", "PASS",
                   f"true risk {smooth_a:.1%} (n={one_one.sum():,}) vs {smooth_b:.1%} (n={high_exp.sum():,})",
                   smooth_ok))

    # Expected mix from the true chances (stable), plus the realised mix (few failures, so noisy).
    area_exp = df.loc[area_clean, ["p_logistics", "p_wont", "p_cant"]].sum()
    area_exp = area_exp / area_exp.sum()
    area_fail = df.loc[area_clean & (y == 1), "failure_type"].value_counts(normalize=True)
    edge = [
        ("1. New COD buyers, no history", df["is_new_cod_buyer"] == 1,
         f"{(df['is_new_cod_buyer'] == 1).mean():.1%} of rows (need >= {v['min_new_buyer_share']:.0%})",
         (df["is_new_cod_buyer"] == 1).mean() >= v["min_new_buyer_share"]),
        (f"2. High refusal rate, low exposure (1-{v['low_exposure_max_orders']} orders, all refused)", low_exp, "", None),
        (f"3. High refusal rate, high exposure (>= {v['high_exposure_min_orders']} orders, >= {v['high_exposure_min_raw_rate']:.0%} refused)", high_exp, "", None),
        (f"4. Low lifetime rate (<= {v['drift_lifetime_max']:.0%}) but high 90-day rate (>= {v['drift_recent_min']:.0%})", drift,
         f"{df.loc[drift, 'drift_direction'].eq('up').mean():.0%} are true 'drift up' buyers", None),
        (f"5. Order value >= {v['big_value_ratio']:g}x buyer's usual", df["value_vs_aov"] >= v["big_value_ratio"], "", None),
        (f"6. High-logistics area (top {1 - v['high_area_quantile']:.0%}) + clean buyer (>= {v['clean_buyer_min_orders']} orders, no buyer-caused failures)", area_clean,
         f"expected failure mix: logistics {area_exp['p_logistics']:.0%}, wont {area_exp['p_wont']:.0%}, "
         f"cant {area_exp['p_cant']:.0%}; realised: logistics {area_fail.get('logistics', 0):.0%}, "
         f"wont {area_fail.get('wont', 0):.0%}, cant {area_fail.get('cant', 0):.0%} "
         f"(only {int((area_clean & (y == 1)).sum())} failures, so noisy). Pass = logistics is the largest expected type",
         area_exp["p_logistics"] == area_exp.max()),
        (f"7. Frequency spike (freq_change_ratio >= {v['freq_spike_ratio']:g})", df["freq_change_ratio"] >= v["freq_spike_ratio"], "", None),
        ("8. Same item at other shops in 48h", df["same_item_other_shops_48h"] >= 1, "", None),
    ]
    edge_rows, edge_ok = [], True
    for name, mask, note, extra_ok in edge:
        n = int(mask.sum())
        ok = n > 0 and (extra_ok is None or extra_ok)
        edge_ok &= ok
        edge_rows.append(f"| {name} | {n:,} | {y[mask].mean():.2%} | {df.loc[mask, 'p_true'].mean():.2%} | {note} | {pf(ok)} |")
    checks.append(("All 8 edge cases present, with counts", "PASS", "see table below", edge_ok))

    # 9. Leakage
    leaked = [c for c in table.columns if c in FORBIDDEN and c not in ID_COLS or c.startswith("latent_")]
    bad_features = [c for c in FEATURE_COLS if c in FORBIDDEN or c in ID_COLS or c.startswith(("latent_", "p_"))]
    unexpected = [c for c in table.columns if c not in ID_COLS + FEATURE_COLS + LABEL_COLS + ["split"]]
    leak_ok = not leaked and not bad_features and not unexpected
    checks.append(("No leakage columns in model_table.csv", "PASS",
                   "none found" if leak_ok else f"leaked={leaked} bad_features={bad_features} unexpected={unexpected}",
                   leak_ok))

    # 10. April is higher: April's actual rate is above the average of the other months.
    # (Each month has ~10-20k rows, so month-to-month noise is about +-0.15 points;
    # the expected rate from the true chances is shown too.)
    by_month = df.groupby("order_month")["label_failed"].mean()
    expected_by_month = df.groupby("order_month")["p_true"].mean()
    others = by_month.drop(4, errors="ignore").mean()
    april_ok = 4 in by_month and by_month[4] > others
    checks.append(("Failure rate by month is higher in April", "April above the other months' average",
                   f"April {by_month.get(4, np.nan):.2%} vs others {others:.2%}", april_ok))

    # ---- Importance ranking of the sanity model (permutation, test AUC drop) ----
    perm = permutation_importance(model, test[FEATURE_COLS], test["label_failed"], scoring="roc_auc",
                                  n_repeats=5, random_state=cfg["seed"])
    coefs = model[-1].coef_[0]
    imp = pd.DataFrame({"feature": FEATURE_COLS, "auc_drop": perm.importances_mean,
                        "std_coef": coefs}).sort_values("auc_drop", ascending=False).reset_index(drop=True)
    rank = {f: i + 1 for i, f in enumerate(imp["feature"])}
    group_best = [(g, min(rank[f] for f in fs)) for g, fs in EXPECTED_ORDER]
    in_order = all(group_best[i][1] <= group_best[i + 1][1] for i in range(len(group_best) - 1))

    # ---- Print + write report ----
    passed = sum(c[3] for c in checks)
    out = [
        "# Validation report", "", f"> {SYNTHETIC_NOTE}", "",
        f"Model rows: {len(df):,} (train {len(train):,}, test {len(test):,}). Failures: {int(y.sum()):,}.",
        f"**{passed} of {len(checks)} checks passed.**", "",
        "## Checks", "", "| Check | Target | Result | Status |", "| --- | --- | --- | --- |",
    ]
    out += [f"| {n} | {t} | {r} | {pf(ok)} |" for n, t, r, ok in checks]
    out += ["", "## Edge cases (model rows)", "",
            "| Edge case | Rows | Actual failure rate | Mean true risk | Note | Status |",
            "| --- | --- | --- | --- | --- | --- |"] + edge_rows
    out += ["", "## Failure rate by month", "", "| Month | Rows | Actual failure rate | Expected (true chances) |",
            "| --- | --- | --- | --- |"]
    out += [f"| {m} | {(df['order_month'] == m).sum():,} | {r:.2%} | {expected_by_month[m]:.2%} |"
            for m, r in by_month.items()]
    out += ["", "## Single-feature AUC (all model rows, best direction)", "", "| Feature | AUC |", "| --- | --- |"]
    out += [f"| `{f}` | {a:.3f} |" for f, a in single.items()]
    out += ["", "## Sanity model importance ranking", "",
            "Permutation importance on the test set (drop in AUC when the feature is shuffled), "
            "plus the standardised logistic-regression coefficient. This is only a sanity check, not the final model.", "",
            "| Rank | Feature | AUC drop | Std. coefficient |", "| --- | --- | --- | --- |"]
    out += [f"| {i + 1} | `{r.feature}` | {r.auc_drop:.4f} | {r.std_coef:+.3f} |" for i, r in imp.iterrows()]
    out += ["", "Expected rough order (a check, not a hard rule): best rank reached by each group:", ""]
    out += [f"- {g}: rank {r}" for g, r in group_best]
    out += ["", f"Groups appear in the expected order: **{'yes' if in_order else 'not exactly'}**.", "",
            "Note: several refusal features carry the same information (lifetime vs 90-day smoothed rate, "
            "`has_ever_refused` vs `days_since_last_refusal`). A linear model gives most of the credit to one "
            "of them, so its twin can rank low even though it is informative on its own "
            "(see the single-feature AUC table).", ""]
    report = "\n".join(out)
    write_text(REPORTS_DIR / "validation_report.md", report)
    print(report)


if __name__ == "__main__":
    main()
