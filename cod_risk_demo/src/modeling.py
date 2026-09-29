"""Shared helpers for training, evaluation, decisions and explanations.

SYNTHETIC DATA - for illustration only. This is NOT real Shopee data.
"""
import json

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.build_features import FEATURE_COLS
from src.common import DATA_DIR, DOCS_DIR, ROOT, read_csv

MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
CHART_NOTE = "Synthetic data, for illustration only. Not Shopee data."
TARGET = "label_failed"

# Columns that must NEVER be model inputs (leakage, hidden values, ids, labels).
FORBIDDEN_INPUTS = {"failure_type", "p_wont", "p_cant", "p_logistics", "buyer_id", "order_id",
                    "order_datetime", "split", "label_failed", "label_buyer_caused"}

# Readable names for charts and the demo page.
NICE_NAMES = {
    "hist_cod_orders": "Past COD orders",
    "hist_refusals": "Past refusals",
    "refusal_rate_smoothed": "Refusal rate (smoothed)",
    "recent_refusal_rate_smoothed_90d": "Refusal rate, last 90 days",
    "days_since_last_refusal": "Days since last refusal",
    "has_ever_refused": "Has ever refused",
    "hist_buyer_caused_misses": "Past 'not home / no cash' misses",
    "hist_courier_caused_failures": "Past courier failures",
    "order_value": "Order value (THB)",
    "value_vs_aov": "Value vs buyer's usual",
    "freq_change_ratio": "Order frequency change (30d)",
    "account_age_days": "Account age (days)",
    "is_new_cod_buyer": "First COD order",
    "is_late_night": "Ordered 01:00-04:59",
    "same_item_other_shops_48h": "Same item at other shops (48h)",
    "is_campaign_day": "Campaign day",
    "expected_days_to_delivery": "Expected delivery days",
    "area_logistics_failure_rate": "Area courier failure rate (90d)",
    "address_is_condo_with_office": "Condo with front office",
    "order_month": "Order month",
}


def feature_list():
    """Read every column marked `feature` in docs/data_dictionary.md, then check it."""
    feats = []
    for line in (DOCS_DIR / "data_dictionary.md").read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 3 and cells[2] == "feature":
            feats.append(cells[1].strip("`"))
    check_features(feats)
    assert feats == FEATURE_COLS, "data dictionary and build_features disagree on the feature list"
    return feats


def check_features(feats):
    """Fail loudly if a leakage / hidden / id / label column is in the feature list."""
    bad = [f for f in feats if f in FORBIDDEN_INPUTS or f.startswith(("latent_", "label_", "p_"))]
    assert not bad, f"Forbidden columns in the feature list: {bad}"


def load_split(name):
    df = read_csv(DATA_DIR / f"{name}.csv")
    return df.sort_values("order_datetime").reset_index(drop=True)


def load_audit():
    """failure_type per order, from the audit file. Used ONLY to report results, never as input."""
    raw = read_csv(DATA_DIR / "raw_events.csv")
    return raw[["order_id", "failure_type", "order_hour"]]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class CalibratedModel:
    """A trained model plus a calibrator, so that `predict(X)` returns a real
    percentage: of orders scored 10%, about 10% really fail."""

    def __init__(self, name, model, features, method):
        self.name, self.model, self.features, self.method = name, model, features, method
        self.calibrator = None

    def raw_proba(self, X):
        return self.model.predict_proba(X[self.features])[:, 1]

    def fit_calibration(self, X, y):
        raw = self.raw_proba(X)
        if self.method == "platt":
            # Logistic regression on the model's log-odds: keeps the ranking, fixes the scale.
            self.calibrator = LogisticRegression(C=1e6, max_iter=1000).fit(logit(raw).reshape(-1, 1), y)
        else:
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw, y)
        return self

    def predict(self, X):
        raw = self.raw_proba(X)
        if self.method == "platt":
            return self.calibrator.predict_proba(logit(raw).reshape(-1, 1))[:, 1]
        return self.calibrator.predict(raw)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def add_chart_note(fig):
    """Put the synthetic-data note at the bottom of a matplotlib figure."""
    fig.text(0.99, 0.005, CHART_NOTE, ha="right", va="bottom", fontsize=9, color="#666666", style="italic")
