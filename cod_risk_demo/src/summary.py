"""Step 9: print the final summary of the whole run.

SYNTHETIC DATA - for illustration only. This is NOT real Shopee data.
"""
import joblib
from sklearn.metrics import roc_auc_score

from src.common import REPORTS_DIR, ROOT, read_csv, DATA_DIR
from src.evaluate import top_capture
from src.modeling import FIG_DIR, MODELS_DIR, TARGET, read_json


def main():
    rec = read_json(MODELS_DIR / "recommended.json")
    dec = read_json(REPORTS_DIR / "decision_summary.json")
    shp = read_json(REPORTS_DIR / "shap_summary.json")
    scored = read_csv(DATA_DIR / "test_scored.csv")
    y = scored[TARGET].to_numpy()

    print("=" * 72)
    print("COD RISK SCORE: FINAL SUMMARY  (synthetic data, for illustration only; not Shopee data)")
    print("=" * 72)
    print("1. Models on the July 2026 test set")
    for key, name in [("logreg", "Logistic Regression"), ("lgbm", "LightGBM")]:
        p = scored[f"p_{key}"].to_numpy()
        print(f"   {name:<20} AUC {roc_auc_score(y, p):.3f}   top 30% capture {top_capture(y, p, 0.30)[0]:.1%}")
    print(f"   Recommended: {rec['name']}. {rec['reason']}")
    print("2. Tiers (share of orders / share of failures / actual failure rate)")
    for t in dec["tiers"]:
        print(f"   Tier {t['tier']}: {t['order_share']:.1%} / {t['failure_share']:.1%} / {t['failure_rate']:.2%}")
    print("3. Net value per year (THB, 142.7M COD orders)")
    for k, v in dec["net_per_year_thb"].items():
        print(f"   {k:<13} THB {v / 1e6:,.1f}M")
    print(f"   Sensitivity: our tiers beat 'ask everyone' in {dec['sensitivity_wins']} scenarios")
    print("4. Top 5 SHAP factors (LightGBM)")
    for s in shp["top10_lgbm"][:5]:
        print(f"   {s['rank']}. {s['name']:<34} matches hidden rule: {'yes' if s['matches_hidden_rule'] else 'no direct effect (proxy)'}")
    print("5. Outputs")
    paths = [REPORTS_DIR / "model_report.md", REPORTS_DIR / "decision_report.md", REPORTS_DIR / "demo_orders.json",
             REPORTS_DIR / "validation_report.md", *sorted(FIG_DIR.glob("*.png")), ROOT / "demo" / "index.html"]
    for p in paths:
        print(f"   {p.relative_to(ROOT).as_posix()}")
    print("These results show how the system works on synthetic data. "
          "Real accuracy will be measured in the pilot with Shopee data.")


if __name__ == "__main__":
    main()
