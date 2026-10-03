"""Model performance maths for the "Model performance" page (numpy / pandas only).

Compares the model's predicted risk with what actually happened to each order in
the synthetic data (label_failed = 1 if the order failed for any reason).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent / "cod_risk_demo" / "models"
FAILURE_TYPES = {"wont": "Refused (won't)", "cant": "Not home / no cash (can't)", "logistics": "Courier problem"}


def auc(y, p):
    """Area under the ROC curve (Mann-Whitney U with tied scores averaged)."""
    y, ranks = np.asarray(y), pd.Series(np.asarray(p)).rank(method="average").to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def roc_points(y, p, n=300):
    """(false positive rate, true positive rate) at n thresholds, highest risk first."""
    order = np.argsort(-np.asarray(p), kind="stable")
    y = np.asarray(y)[order]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    idx = np.unique(np.linspace(0, len(y) - 1, n).astype(int))
    return np.r_[0, fp[idx] / max(fp[-1], 1)], np.r_[0, tp[idx] / max(tp[-1], 1)]


def capture_points(y, p, n=300):
    """(share of orders asked, share of failures found) when asking the highest risk first."""
    order = np.argsort(-np.asarray(p), kind="stable")
    found = np.cumsum(np.asarray(y)[order]) / max(np.sum(y), 1)
    idx = np.unique(np.linspace(0, len(found) - 1, n).astype(int))
    return np.r_[0, (idx + 1) / len(found)], np.r_[0, found[idx]]


def capture_at(y, p, share):
    """Share of all failures that sit in the top `share` of orders by predicted risk."""
    k = max(int(round(len(y) * share)), 1)
    top = np.argsort(-np.asarray(p), kind="stable")[:k]
    return float(np.asarray(y)[top].sum() / max(np.sum(y), 1))


def calibration(y, p, groups=10):
    """Orders split into equal groups by predicted risk: mean predicted vs actual failure rate."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["group"] = pd.qcut(d["p"].rank(method="first"), groups, labels=False) + 1
    return d.groupby("group").agg(orders=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))


def summary(d):
    """The headline numbers for one set of orders (needs columns label_failed and risk)."""
    y, p = d["label_failed"].to_numpy(), d["risk"].to_numpy()
    base = y.mean()
    top5 = capture_at(y, p, 0.05)
    return {
        "Orders": len(d),
        "Failures": int(y.sum()),
        "Actual failure rate": base,
        "Mean predicted risk": p.mean(),
        "AUC": auc(y, p),
        "Brier score": float(np.mean((p - y) ** 2)),
        "Brier score, always predicting the average": float(np.mean((base - y) ** 2)),
        "Top 30% capture": capture_at(y, p, 0.30),
        "Top 5% capture": top5,
        "Failure rate inside the top 5% (x average)": top5 * y.sum() / max(round(len(y) * 0.05), 1) / max(base, 1e-9),
    }


def report_test_table():
    """The 'Results on the test set' table of reports/model_report.md (Logistic Regression vs LightGBM),
    or None if the report is missing or its layout changed."""
    path = MODELS_DIR.parent / "reports" / "model_report.md"
    try:
        text = path.read_text(encoding="utf-8")
        section = text.split("## Results on the test set", 1)[1].split("\n## ", 1)[0]
    except (OSError, IndexError):
        return None
    rows = [[c.replace("**", "").replace("*", "").replace("`", "").strip() for c in line.strip().strip("|").split("|")]
            for line in section.splitlines() if line.startswith("|") and "---" not in line]
    if len(rows) < 2 or len(rows[0]) != 3:
        return None
    return pd.DataFrame(rows[1:], columns=["Metric (test set)", "Logistic Regression", "LightGBM"])


def training_record():
    """Validation AUCs saved by the training pipeline + which model was recommended and why."""
    s = json.loads((MODELS_DIR / "training_summary.json").read_text(encoding="utf-8"))
    rec = json.loads((MODELS_DIR / "recommended.json").read_text(encoding="utf-8"))
    rows = [{"Model": "Logistic Regression (used in the app)", "Setting": "scaled features, balanced classes",
             "Validation AUC": s["logreg_valid_auc"], "Trees": None}]
    for t in s["lgbm_trials"]:
        change = "base settings" if t["change"] == "base" else ", ".join(f"{k}={v}" for k, v in t["change"].items())
        rows.append({"Model": "LightGBM" + (" (picked)" if t["change"] == s["lgbm_picked"] else ""),
                     "Setting": change, "Validation AUC": t["valid_auc"], "Trees": t["best_iteration"]})
    return pd.DataFrame(rows), s, rec
