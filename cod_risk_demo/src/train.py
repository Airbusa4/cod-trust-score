"""Step 4: train the baseline (Logistic Regression) and the main model (LightGBM).

SYNTHETIC DATA - for illustration only. This is NOT real Shopee data.

  * train.csv is split by time: the first 80% of orders fit the models, the
    last 20% (validation) is used for early stopping, picking LightGBM settings
    and calibration. The test set (July 2026) is NOT touched here.
  * Both models are calibrated on the validation set, so a score of 10% means
    about 10% of such orders really fail.

Output: models/logreg.joblib, models/lgbm.joblib, models/feature_list.json,
        models/training_summary.json
"""
import joblib
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common import REPORTS_DIR, load_config
from src.modeling import MODELS_DIR, TARGET, CalibratedModel, check_features, feature_list, load_split, write_json


def time_split(train, valid_share):
    """Last `valid_share` of the train rows (by order time) become the validation set."""
    cut = int(len(train) * (1 - valid_share))
    return train.iloc[:cut], train.iloc[cut:]


def train_logreg(fit, valid, feats, cfg):
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(class_weight="balanced", max_iter=2000, random_state=cfg["seed"]))
    model.fit(fit[feats], fit[TARGET])
    return CalibratedModel("Logistic Regression", model, feats, cfg["model"]["calibration_method"])


def train_lgbm(fit, valid, feats, params, cfg):
    model = lgb.LGBMClassifier(**params, random_state=cfg["seed"], verbose=-1)
    model.fit(fit[feats], fit[TARGET], eval_X=(valid[feats],), eval_y=(valid[TARGET],), eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(cfg["model"]["early_stopping_rounds"], verbose=False)])
    return model


def main():
    cfg = load_config()
    mc = cfg["model"]
    np.random.seed(cfg["seed"])

    # Stop if the data checks did not all pass (brief, step 0).
    report = (REPORTS_DIR / "validation_report.md").read_text(encoding="utf-8")
    failed = [line.split("|")[1].strip() for line in report.splitlines() if line.rstrip().endswith("| FAIL |")]
    if failed:
        raise SystemExit("Data checks FAILED, not training. Fix the data first:\n  " + "\n  ".join(failed))

    feats = feature_list()
    check_features(feats)
    print(f"Features ({len(feats)}): {', '.join(feats)}")

    train = load_split("train")
    fit, valid = time_split(train, mc["valid_share"])
    print(f"Fit rows {len(fit):,} (to {fit['order_datetime'].iloc[-1][:10]}), "
          f"validation rows {len(valid):,} (from {valid['order_datetime'].iloc[0][:10]})")

    # ---- Baseline ----
    logreg = train_logreg(fit, valid, feats, cfg)
    lr_auc = roc_auc_score(valid[TARGET], logreg.raw_proba(valid))
    print(f"Logistic Regression: validation AUC {lr_auc:.4f}")

    # ---- LightGBM: base settings + at most 3 small changes, pick by validation AUC ----
    trials = []
    for change in [{}] + mc["lgbm_variants"]:
        params = {**mc["lgbm_base"], **change}
        model = train_lgbm(fit, valid, feats, params, cfg)
        auc = roc_auc_score(valid[TARGET], model.predict_proba(valid[feats])[:, 1])
        trials.append({"change": change or "base", "params": params, "best_iteration": int(model.best_iteration_),
                       "valid_auc": round(auc, 4), "model": model})
        print(f"LightGBM {str(change or 'base'):<60} trees {model.best_iteration_:>3}  validation AUC {auc:.4f}")
    best = max(trials, key=lambda t: t["valid_auc"])
    lgbm = CalibratedModel("LightGBM", best["model"], feats, mc["calibration_method"])
    print(f"Picked LightGBM setting: {best['change']}")

    # ---- Calibrate both on the validation set ----
    for m in (logreg, lgbm):
        m.fit_calibration(valid, valid[TARGET])
        p = m.predict(valid)
        print(f"{m.name}: calibrated mean on validation {p.mean():.2%} vs actual {valid[TARGET].mean():.2%}")

    # ---- Save ----
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(logreg, MODELS_DIR / "logreg.joblib")
    joblib.dump(lgbm, MODELS_DIR / "lgbm.joblib")
    write_json(MODELS_DIR / "feature_list.json", feats)
    write_json(MODELS_DIR / "training_summary.json", {
        "note": "Synthetic data, for illustration only. Not Shopee data.",
        "fit_rows": len(fit), "valid_rows": len(valid),
        "valid_from": valid["order_datetime"].iloc[0],
        "calibration_method": mc["calibration_method"],
        "logreg_valid_auc": round(lr_auc, 4),
        "lgbm_trials": [{k: v for k, v in t.items() if k != "model"} for t in trials],
        "lgbm_picked": best["change"],
    })
    print("Saved models/logreg.joblib, models/lgbm.joblib, models/feature_list.json, models/training_summary.json")


if __name__ == "__main__":
    main()
