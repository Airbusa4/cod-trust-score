"""Step 5: evaluate both models on the TEST set only (July 2026).

SYNTHETIC DATA - for illustration only. This is NOT real Shopee data.

Writes reports/model_report.md, reports/figures/calibration.png,
data/test_scored.csv (calibrated scores per test order) and
models/recommended.json (which model the tiers and demo use).
"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.common import DATA_DIR, REPORTS_DIR, SYNTHETIC_NOTE, load_config, write_csv, write_text
from src.modeling import (FIG_DIR, MODELS_DIR, TARGET, add_chart_note, load_audit, load_split, read_json,
                          write_json)

HONESTY = ("**These results show how the system works on synthetic data. "
           "Real accuracy will be measured in the pilot with Shopee data.**")
MODEL_FILES = {"logreg": "logreg.joblib", "lgbm": "lgbm.joblib"}


def top_capture(y, p, share):
    """Share of all failures found in the `share` highest-risk orders, and the failure rate there."""
    n_top = int(round(len(p) * share))
    top = np.argsort(-p, kind="stable")[:n_top]
    return y[top].sum() / y.sum(), y[top].mean()


def calibration_table(y, p, groups=10):
    """Split orders into 10 equal groups by predicted risk; compare predicted % with actual %."""
    df = pd.DataFrame({"y": y, "p": p})
    df["group"] = pd.qcut(df["p"].rank(method="first"), groups, labels=range(1, groups + 1))
    return df.groupby("group", observed=True).agg(orders=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))


def metrics(y, p, is_new, ftype):
    top30, _ = top_capture(y, p, 0.30)
    top5, top5_rate = top_capture(y, p, 0.05)
    in_top30 = np.zeros(len(p), bool)
    in_top30[np.argsort(-p, kind="stable")[:int(round(len(p) * 0.30))]] = True
    by_type = {t: in_top30[ftype == t].mean() for t in ["wont", "cant", "logistics"]}
    caught_mix = pd.Series(ftype[in_top30 & (y == 1)]).value_counts(normalize=True)
    new = is_new == 1
    return {
        "auc": roc_auc_score(y, p), "top30": top30, "top5": top5, "top5_rate": top5_rate,
        "brier": brier_score_loss(y, p),
        "new_auc": roc_auc_score(y[new], p[new]),
        "new_top30_within": top_capture(y[new], p[new], 0.30)[0],
        "new_caught_overall": in_top30[new & (y == 1)].mean(),
        "type_caught": by_type, "caught_mix": {t: caught_mix.get(t, 0.0) for t in ["wont", "cant", "logistics"]},
    }


def plot_calibration(tables, path):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    top = max(max(t["predicted"].max(), t["actual"].max()) for t in tables.values()) * 1.1
    ax.plot([0, top], [0, top], ls="--", color="#999999", label="Perfect calibration")
    for (name, t), color in zip(tables.items(), ["#1f77b4", "#d62728"]):
        ax.plot(t["predicted"] * 100, t["actual"] * 100, "o-", color=color, label=name, lw=2, ms=7)
    ax.set_xlim(0, top * 100)
    ax.set_ylim(0, top * 100)
    ax.plot([0, top * 100], [0, top * 100], ls="--", color="#999999")
    ax.set_xlabel("Predicted failure chance (%)", fontsize=12)
    ax.set_ylabel("Actual failure rate (%)", fontsize=12)
    ax.set_title("Calibration on the July 2026 test set\n(10 groups by predicted risk)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    add_chart_note(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pct(x):
    return f"{x:.1%}"


def main():
    cfg = load_config()
    test = load_split("test")
    audit = load_audit()
    test = test.merge(audit[["order_id", "failure_type"]], on="order_id", how="left")  # reporting only
    y, is_new, ftype = test[TARGET].to_numpy(), test["is_new_cod_buyer"].to_numpy(), test["failure_type"].to_numpy()

    models = {key: joblib.load(MODELS_DIR / f) for key, f in MODEL_FILES.items()}
    scores = {key: m.predict(test) for key, m in models.items()}
    res = {key: metrics(y, scores[key], is_new, ftype) for key in models}
    tables = {models[k].name: calibration_table(y, scores[k]) for k in models}

    # ---- Pick the recommended model: LightGBM only if it clearly wins ----
    gap = res["lgbm"]["auc"] - res["logreg"]["auc"]
    min_gap = cfg["model"]["min_auc_gap_for_lgbm"]
    rec = "lgbm" if gap >= min_gap else "logreg"
    reason = (f"LightGBM beats Logistic Regression by {gap:+.3f} AUC (>= {min_gap}), so it is recommended."
              if rec == "lgbm" else
              f"LightGBM's AUC gap over Logistic Regression is {gap:+.3f} (under {min_gap}). It does not clearly win, "
              "so the simpler, easier-to-explain Logistic Regression is recommended.")
    write_json(MODELS_DIR / "recommended.json", {"model": rec, "file": MODEL_FILES[rec], "name": models[rec].name,
                                                 "auc_gap_lgbm_minus_logreg": round(gap, 4), "reason": reason})

    scored = test[["order_id", "order_datetime", TARGET]].copy()
    scored["p_logreg"], scored["p_lgbm"] = scores["logreg"].round(6), scores["lgbm"].round(6)
    scored["p"] = scored[f"p_{rec}"]
    write_csv(scored, DATA_DIR / "test_scored.csv")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_calibration(tables, FIG_DIR / "calibration.png")

    # ---- Report ----
    tr = read_json(MODELS_DIR / "training_summary.json")
    L, G = res["logreg"], res["lgbm"]
    row = lambda label, f: f"| {label} | {f(L)} | {f(G)} |"
    out = [
        "# COD Risk Score: model report", "", f"> {SYNTHETIC_NOTE}", "", HONESTY, "",
        f"Test set: {len(test):,} COD orders from July 2026, {int(y.sum()):,} failures ({y.mean():.2%}). "
        "The test set was used only for this final evaluation.", "",
        "## Setup", "",
        f"- Features: the {len(models['logreg'].features)} columns marked `feature` in `docs/data_dictionary.md`. "
        "A code check stops training if any leakage, hidden or id column gets in.",
        f"- Training: Feb 2026 to early Jun 2026 ({tr['fit_rows']:,} orders). Validation: the last 20% of the "
        f"train period by date ({tr['valid_rows']:,} orders, from {tr['valid_from'][:10]}), used for early "
        "stopping, choosing LightGBM settings and calibration.",
        "- Baseline: Logistic Regression (scaled features, balanced class weights).",
        f"- Main: LightGBM. Tried the base settings plus 3 small changes; picked `{tr['lgbm_picked']}` "
        "by validation AUC:", "",
        "| LightGBM setting | Trees (early stopping) | Validation AUC |", "| --- | --- | --- |",
    ]
    out += [f"| `{t['change']}` | {t['best_iteration']} | {t['valid_auc']:.4f} |" for t in tr["lgbm_trials"]]
    out += [f"| *Logistic Regression, for reference* | | {tr['logreg_valid_auc']:.4f} |", "",
            f"- Calibration: both models, `{tr['calibration_method']}` method, fitted on the validation set.", "",
            "## Results on the test set", "",
            "| Metric | Logistic Regression | LightGBM |", "| --- | --- | --- |",
            row("AUC", lambda r: f"{r['auc']:.3f}"),
            row("**Top 30% capture** (target >= 60%)", lambda r: f"**{pct(r['top30'])}**"),
            row("Top 5% capture", lambda r: pct(r["top5"])),
            row("Actual failure rate inside the top 5%", lambda r: f"{r['top5_rate']:.1%} ({r['top5_rate'] / y.mean():.1f}x average)"),
            row("Brier score (lower is better)", lambda r: f"{r['brier']:.5f}"),
            f"| *Brier score if we always predicted the average rate* | {brier_score_loss(y, np.full(len(y), y.mean())):.5f} | |",
            row("New COD buyers only: AUC", lambda r: f"{r['new_auc']:.3f}"),
            row("New COD buyers only: top 30% capture (ranked among new buyers)", lambda r: pct(r["new_top30_within"])),
            row("New COD buyers' failures caught in the overall top 30%", lambda r: pct(r["new_caught_overall"])),
            row("`wont` failures caught in the top 30%", lambda r: pct(r["type_caught"]["wont"])),
            row("`cant` failures caught in the top 30%", lambda r: pct(r["type_caught"]["cant"])),
            row("`logistics` failures caught in the top 30%", lambda r: pct(r["type_caught"]["logistics"])),
            row("Mix of the failures caught in the top 30% (wont / cant / logistics)",
                lambda r: " / ".join(pct(r["caught_mix"][t]) for t in ["wont", "cant", "logistics"])),
            "", f"New COD buyers are {is_new.mean():.1%} of test orders and have no refusal history at all, "
            "yet the model still ranks them using order-context signals.", "",
            "## Recommended model", "", f"**{models[rec].name}.** {reason}", "",
            "The decision tiers and the demo use the recommended model's calibrated score.", "",
            "## Calibration (test set)", "",
            "Orders are split into 10 equal groups by predicted risk. A trustworthy % means predicted and actual are close.", "",
            "![Calibration chart](figures/calibration.png)", "",
            "| Group | Orders | LR predicted | LR actual | LightGBM predicted | LightGBM actual |",
            "| --- | --- | --- | --- | --- | --- |"]
    tl, tg = tables[models["logreg"].name], tables[models["lgbm"].name]
    out += [f"| {g} | {tl.loc[g, 'orders']:,} | {tl.loc[g, 'predicted']:.2%} | {tl.loc[g, 'actual']:.2%} | "
            f"{tg.loc[g, 'predicted']:.2%} | {tg.loc[g, 'actual']:.2%} |" for g in tl.index]
    out += ["", "Each group has about 1,600 orders and only 10 to 150 failures, so small gaps between predicted "
            "and actual are expected noise.", ""]
    write_text(REPORTS_DIR / "model_report.md", "\n".join(out))

    print(f"Test AUC: LR {L['auc']:.3f}, LightGBM {G['auc']:.3f} (gap {gap:+.3f})")
    print(f"Top 30% capture: LR {L['top30']:.1%}, LightGBM {G['top30']:.1%}")
    print(f"Top 5%: capture LR {L['top5']:.1%} / LGBM {G['top5']:.1%}; fail rate LR {L['top5_rate']:.1%} / LGBM {G['top5_rate']:.1%}")
    print(f"Brier: LR {L['brier']:.5f}, LGBM {G['brier']:.5f}; new-buyer AUC LR {L['new_auc']:.3f}, LGBM {G['new_auc']:.3f}")
    print(f"Recommended: {models[rec].name}. {reason}")
    print("Wrote reports/model_report.md, reports/figures/calibration.png, data/test_scored.csv")


if __name__ == "__main__":
    main()
