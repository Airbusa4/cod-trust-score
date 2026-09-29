"""Step 7: explain the models with SHAP and pick the two demo orders.

SYNTHETIC DATA - for illustration only. This is NOT real Shopee data.

  * Overall charts (importance bar, beeswarm) use LightGBM + shap.TreeExplainer
    on 5,000 test rows, as the brief asks.
  * The two example orders (waterfalls, demo reasons) are explained with the
    RECOMMENDED model, so the reasons always match the score shown in the demo.
    If that is Logistic Regression, SHAP uses LinearExplainer (exact for linear models).
  * SHAP values are multiplied by the Platt calibration slope, so each order's
    waterfall adds up to the log-odds of its CALIBRATED score.

Writes reports/figures/shap_*.png, reports/demo_orders.json and adds a SHAP
section to reports/model_report.md.
"""
import calendar
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.common import REPORTS_DIR, load_config
from src.decide import TIER_ACTIONS, assign_tiers
from src.modeling import (CHART_NOTE, FIG_DIR, MODELS_DIR, NICE_NAMES, add_chart_note, load_audit, load_split,
                          read_json, write_json)

SAMPLE_SIZE = 5000
warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer")

# What the hidden rule really does with each feature (from docs/hidden_rules.md).
# sign: +1 = higher value should raise risk, -1 = lower it, 0 = no direct effect in the rule.
HIDDEN_RULE = {
    "refusal_rate_smoothed": (+1, "Evidence of the hidden refusal trait `latent_wont` (the main driver)"),
    "recent_refusal_rate_smoothed_90d": (+1, "Evidence of `latent_wont`, incl. drift in the last 90 days"),
    "hist_refusals": (+1, "Evidence of `latent_wont`"),
    "has_ever_refused": (+1, "Evidence of `latent_wont`"),
    "days_since_last_refusal": (0, "Evidence of `latent_wont` (-1 = never refused)"),
    "hist_buyer_caused_misses": (+1, "Evidence of the hidden `latent_cant` trait"),
    "hist_cod_orders": (0, "No direct effect: exposure only (but long-time buyers are less often in the high-refusal group)"),
    "same_item_other_shops_48h": (+1, "+1.0 log-odds per shop (max 2); also more common for high-refusal buyers"),
    "value_vs_aov": (+1, "+0.35 at 2-3x, +0.7 at 3x+ the buyer's TRUE usual value"),
    "order_value": (0, "No direct effect (only the value vs usual matters)"),
    "account_age_days": (-1, "+0.6 if the account is younger than 30 days; recent signups have more high-refusal buyers"),
    "is_new_cod_buyer": (0, "No direct effect; proxy for young accounts"),
    "is_late_night": (+1, "+0.5 for 01:00-04:59"),
    "freq_change_ratio": (+1, "+0.4 if >= 3"),
    "is_campaign_day": (+1, "+0.3"),
    "expected_days_to_delivery": (+1, "+0.1 per day above 2 (won't), +0.05 per day (can't)"),
    "order_month": (0, "+0.5 (can't) in April and late December. All test orders are July, so here it is only a constant shift"),
    "address_is_condo_with_office": (-1, "-0.7 (can't)"),
    "area_logistics_failure_rate": (+1, "Evidence of the area's hidden courier failure rate"),
    "hist_courier_caused_failures": (0, "No link to the buyer (only weakly to the area)"),
}

EXPECTED_GROUPS = [
    ("Refusal history", ["refusal_rate_smoothed", "hist_refusals", "has_ever_refused"]),
    ("Recent refusal rate", ["recent_refusal_rate_smoothed_90d"]),
    ("Days since last refusal", ["days_since_last_refusal"]),
    ("Order factors", ["value_vs_aov", "same_item_other_shops_48h", "is_late_night", "freq_change_ratio",
                       "account_age_days", "is_new_cod_buyer", "is_campaign_day", "order_value"]),
    ("Area and courier factors", ["area_logistics_failure_rate", "hist_courier_caused_failures"]),
]


# ---------------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------------
def platt_slope(cal_model):
    """Calibrated log-odds = slope * raw log-odds + intercept (Platt). 1.0 for isotonic."""
    return float(cal_model.calibrator.coef_[0][0]) if cal_model.method == "platt" else 1.0


def shap_values(cal_model, X, background):
    """SHAP values (log-odds, scaled to the calibrated score) for rows X."""
    feats = cal_model.features
    slope = platt_slope(cal_model)
    est = cal_model.model
    if hasattr(est, "booster_"):  # LightGBM
        explainer = shap.TreeExplainer(est)
        values = explainer.shap_values(X[feats])
        values = values[1] if isinstance(values, list) else values
        base = float(np.ravel(explainer.expected_value)[-1])
    else:  # Pipeline(StandardScaler, LogisticRegression)
        scaler, clf = est[0], est[-1]
        masker = shap.maskers.Independent(scaler.transform(background[feats]), max_samples=len(background))
        explainer = shap.LinearExplainer(clf, masker)
        values = explainer.shap_values(scaler.transform(X[feats]))
        base = float(np.ravel(explainer.expected_value)[0])
    # Move the base value so that base + sum(values) equals the calibrated log-odds.
    calibrated_logodds = np.log(cal_model.predict(X) / (1 - cal_model.predict(X)))
    values = values * slope
    base_cal = float(np.mean(calibrated_logodds - values.sum(axis=1)))
    return values, base_cal


def explanation(values, base, X, feats):
    return shap.Explanation(values=values, base_values=np.full(len(X), base), data=X[feats].to_numpy(),
                            feature_names=[NICE_NAMES[f] for f in feats])


def save(fig, name, title):
    fig.suptitle(title, fontsize=14, y=0.995)
    add_chart_note(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plain-word reasons for the demo
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
        "account_age_days": lambda: f"Account is {int(v)} days old",
        "is_new_cod_buyer": lambda: "First-ever COD order" if v else "Has COD history",
        "is_late_night": lambda: f"Ordered late at night ({int(r['order_hour']):02d}:00-{int(r['order_hour']):02d}:59)" if v else "Ordered in normal hours",
        "same_item_other_shops_48h": lambda: (f"Same item ordered with COD at {plural(v, 'other shop')} in 48h" if v
                                              else "No similar COD orders at other shops"),
        "is_campaign_day": lambda: "Campaign day order" if v else "Not a campaign day",
        "expected_days_to_delivery": lambda: f"Delivery expected in {plural(v, 'day')}",
        "area_logistics_failure_rate": lambda: f"Area courier failure rate {v:.2%} (last 90 days)",
        "address_is_condo_with_office": lambda: "Condo with a front office to receive parcels" if v else "House address (no front office)",
        "order_month": lambda: f"Ordered in {calendar.month_name[int(v)]}",
    }
    return texts[f]()


def pick_orders(test, p, tier):
    """Risky: young account, late night, same item elsewhere, value far above usual. Safe: long clean history."""
    d = test.assign(p=p, tier=tier)
    risky_score = ((d["account_age_days"] < 90).astype(int) + d["is_late_night"]
                   + (d["same_item_other_shops_48h"] >= 1) + (d["value_vs_aov"] >= 2) + (d["tier"] == 2))
    risky = d.assign(s=risky_score).sort_values(["s", "p"], ascending=False).iloc[0]
    safe_pool = d[(d["hist_cod_orders"] >= 20) & (d["hist_refusals"] == 0) & d["value_vs_aov"].between(0.8, 1.25)
                  & d["order_hour"].between(9, 20) & (d["same_item_other_shops_48h"] == 0) & (d["tier"] == 0)]
    safe = safe_pool.sort_values("p").iloc[0]
    return risky.name, safe.name


def main():
    cfg = load_config()
    seed = cfg["seed"]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    test = load_split("test").merge(load_audit()[["order_id", "order_hour"]], on="order_id")
    train = load_split("train")
    lgbm = joblib.load(MODELS_DIR / "lgbm.joblib")
    logreg = joblib.load(MODELS_DIR / "logreg.joblib")
    rec_info = read_json(MODELS_DIR / "recommended.json")
    rec = joblib.load(MODELS_DIR / rec_info["file"])
    feats = lgbm.features
    background = train.sample(2000, random_state=seed)

    # ---- 1-2. Overall charts: LightGBM on 5,000 test rows ----
    sample = test.sample(min(SAMPLE_SIZE, len(test)), random_state=seed)
    lg_vals, lg_base = shap_values(lgbm, sample, background)
    exp_lg = explanation(lg_vals, lg_base, sample, feats)

    plt.figure(figsize=(9, 7))
    shap.plots.bar(exp_lg, max_display=15, show=False)
    save(plt.gcf(), "shap_importance_bar.png", "What drives COD failure risk (LightGBM, mean |SHAP|)")

    plt.figure(figsize=(9, 7))
    shap.plots.beeswarm(exp_lg, max_display=15, show=False)
    save(plt.gcf(), "shap_beeswarm.png", "How high / low values push risk (LightGBM)")

    lr_vals, _ = shap_values(logreg, sample, background)
    imp = pd.DataFrame({"feature": feats, "lgbm": np.abs(lg_vals).mean(0), "logreg": np.abs(lr_vals).mean(0)})
    # Direction the model learned: correlation between the feature's value and its SHAP value.
    with np.errstate(invalid="ignore", divide="ignore"):
        imp["lgbm_dir"] = [np.nan_to_num(np.sign(np.corrcoef(sample[f], lg_vals[:, i])[0, 1]))
                           if sample[f].std() > 0 and lg_vals[:, i].std() > 0 else 0 for i, f in enumerate(feats)]
    imp["lgbm_rank"] = imp["lgbm"].rank(ascending=False).astype(int)
    imp["logreg_rank"] = imp["logreg"].rank(ascending=False).astype(int)
    imp = imp.sort_values("lgbm_rank")

    # ---- 3-4. Two example orders, explained with the recommended model ----
    p_all = rec.predict(test)
    tier_all = assign_tiers(p_all, cfg["decision"])
    i_risky, i_safe = pick_orders(test, p_all, tier_all)
    demo = {"note": CHART_NOTE, "model": rec_info["name"],
            "tier_cutoffs": {"tier1": cfg["decision"]["tier1_cutoff"], "tier2": cfg["decision"]["tier2_cutoff"]},
            "orders": {}}
    for key, idx, fname, label in [("A", i_risky, "shap_order_risky.png", "risky"),
                                   ("B", i_safe, "shap_order_safe.png", "safe")]:
        row = test.loc[[idx]]
        vals, base = shap_values(rec, row, background)
        plt.figure(figsize=(9, 6.5))
        shap.plots.waterfall(explanation(vals, base, row, feats)[0], max_display=10, show=False)
        p = float(p_all[idx])
        t = int(tier_all[idx])
        save(plt.gcf(), fname, f"Order {key} ({label}): risk {p:.1%}, Tier {t} ({rec_info['name']}, log-odds)")

        r = test.loc[idx]
        top = np.argsort(-np.abs(vals[0]))[:3]
        demo["orders"][key] = {
            "label": label, "order_id": int(r["order_id"]), "order_datetime": r["order_datetime"],
            "risk": round(p, 4), "tier": t, "action": TIER_ACTIONS[t],
            "facts": {
                "Order value": f"฿{r['order_value']:,.0f}",
                "Value vs usual": f"{r['value_vs_aov']:.1f}x",
                "Order time": f"{r['order_datetime'][11:16]}",
                "Account age": plural(r["account_age_days"], "day"),
                "Past COD orders": f"{int(r['hist_cod_orders'])}",
                "Past refusals": f"{int(r['hist_refusals'])}",
                "Same item at other shops (48h)": f"{int(r['same_item_other_shops_48h'])}",
                "Expected delivery": plural(r["expected_days_to_delivery"], "day"),
            },
            "features": {f: (float(r[f]) if isinstance(r[f], float) else int(r[f])) for f in feats},
            "reasons": [{"feature": feats[j], "name": NICE_NAMES[feats[j]], "text": plain_reason(feats[j], r),
                         "shap": round(float(vals[0][j]), 3), "direction": "up" if vals[0][j] > 0 else "down"}
                        for j in top],
        }
    write_json(REPORTS_DIR / "demo_orders.json", demo)

    # ---- Add the SHAP section to the model report ----
    report_path = REPORTS_DIR / "model_report.md"
    text = report_path.read_text(encoding="utf-8").split("\n## Explanations (SHAP)")[0].rstrip() + "\n"
    out = ["", "## Explanations (SHAP)", "",
           f"Overall charts: LightGBM + `shap.TreeExplainer` on {len(sample):,} test orders. "
           f"The two example orders use the recommended model ({rec_info['name']}), so they match the demo scores.", "",
           "![SHAP importance](figures/shap_importance_bar.png)", "", "![SHAP beeswarm](figures/shap_beeswarm.png)", "",
           "### Top 10 factors vs the hidden rule", "",
           "\"Found?\" = the factor is in the top 10 and pushes risk in the same direction as the hidden rule "
           "(direction = sign of the correlation between the value and its SHAP value).", "",
           "| LightGBM rank | Factor | Mean abs SHAP | Learned direction | LR rank | True effect in the hidden rule | Found? |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    found_count = 0
    for _, r in imp.head(10).iterrows():
        sign, truth = HIDDEN_RULE[r["feature"]]
        learned = {1: "higher = riskier", -1: "higher = safer", 0: "flat"}[int(r["lgbm_dir"])]
        if sign == 0:
            found = "no direct effect; picked up as a proxy"
        elif sign == r["lgbm_dir"]:
            found = "yes"
            found_count += 1
        else:
            found = "direction differs"
        out.append(f"| {r['lgbm_rank']} | {NICE_NAMES[r['feature']]} | {r['lgbm']:.3f} | {learned} | "
                   f"{r['logreg_rank']} | {truth} | {found} |")
    missed = [NICE_NAMES[f] for f, (s, _) in HIDDEN_RULE.items()
              if s != 0 and f not in set(imp.head(10)["feature"])]
    out += ["", f"The model found {found_count} of the true drivers in its top 10 with the right direction. "
            f"True drivers outside the top 10: {', '.join(missed) if missed else 'none'}. "
            "These are either rare (late-night orders are 8% of orders, frequency spikes under 1%) or affect only "
            "the smaller `cant` / `logistics` failure types, so their total contribution is small.", "",
            "### Expected order", "",
            "Expected: refusal history > recent refusal rate > days since last refusal > order factors > area and courier factors.", "",
            "| Group | Best LightGBM rank | Best LR rank |", "| --- | --- | --- |"]
    rank_of = imp.set_index("feature")
    group_ranks = []
    for g, fs in EXPECTED_GROUPS:
        gl, gr = rank_of.loc[fs, "lgbm_rank"].min(), rank_of.loc[fs, "logreg_rank"].min()
        group_ranks.append(gl)
        out.append(f"| {g} | {gl} | {gr} |")
    in_order = all(a <= b for a, b in zip(group_ranks, group_ranks[1:]))
    out += ["", ("The groups appear in the expected order." if in_order else
                 "The groups are **not exactly** in the expected order. Refusal history is on top as expected. "
                 "The recent rate and days since last refusal carry mostly the same information as the lifetime "
                 "refusal rate, so the model gives the credit to one of them and the others rank lower. "
                 "Order factors, especially the same item at other shops, are stronger than the brief expected "
                 "because about a quarter of orders come from buyers with no history, where order factors are "
                 "the only signal."), "",
            "### Example orders", "",
            "![Risky order](figures/shap_order_risky.png)", "", "![Safe order](figures/shap_order_safe.png)", ""]
    for key, o in demo["orders"].items():
        out.append(f"- **Order {key} ({o['label']})**: risk {o['risk']:.1%}, Tier {o['tier']} ({o['action']}). Top reasons: "
                   + "; ".join(f"{x['text']} ({'raises' if x['direction'] == 'up' else 'lowers'} risk)" for x in o["reasons"]))
    report_path.write_text(text + "\n".join(out) + "\n", encoding="utf-8")

    summary = [{"rank": int(r["lgbm_rank"]), "feature": r["feature"], "name": NICE_NAMES[r["feature"]],
                "matches_hidden_rule": (HIDDEN_RULE[r["feature"]][0] != 0 and HIDDEN_RULE[r["feature"]][0] == r["lgbm_dir"])}
               for _, r in imp.head(10).iterrows()]
    write_json(REPORTS_DIR / "shap_summary.json", {"note": CHART_NOTE, "top10_lgbm": summary})

    print("Top 5 SHAP factors (LightGBM):")
    for s in summary[:5]:
        print(f"  {s['rank']}. {s['name']:<35} matches hidden rule: {'yes' if s['matches_hidden_rule'] else 'proxy / no direct effect'}")
    for key, o in demo["orders"].items():
        print(f"Order {key} ({o['label']}): id {o['order_id']}, risk {o['risk']:.1%}, Tier {o['tier']}")
    print("Wrote reports/figures/shap_*.png, reports/demo_orders.json, SHAP section of reports/model_report.md")


if __name__ == "__main__":
    main()
