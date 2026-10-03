# COD Risk Score demo: synthetic data pipeline

> **SYNTHETIC DATA, for illustration only. This is NOT real Shopee data.**
> Every number here comes from a simulation built for a case competition demo.

This folder builds a synthetic dataset of Shopee Thailand cash-on-delivery (COD)
orders, ready for training a failed-delivery risk model. It matches the facts
in the case:

- COD failure rate about 2.61%;
- failures split into buyer won't (about 65%), buyer can't (about 28%) and logistics (about 7%);
- many orders come from buyers with no COD history;
- failure rates are higher in April and late December.

## How to run

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python -m src.generate_data && python -m src.build_features && python -m src.validate_data
```

In Windows PowerShell 5.1, `&&` does not work. Use `;` instead:
`python -m src.generate_data; python -m src.build_features; python -m src.validate_data`

The whole run takes about 30 seconds. The seed is fixed (`seed: 42` in
`config.yaml`), so every run gives exactly the same files.

CSV files start with a one-line `#` note saying the data is synthetic. Read them with:

```python
pd.read_csv("data/model_table.csv", comment="#")
```

## Model, decisions and demo (one command)

```bash
./run_all.sh        # macOS / Linux / Git Bash
run_all.bat         # Windows
```

This rebuilds the data (same files every time), then trains the models, evaluates
them on the July 2026 test set, applies the decision tiers and money simulation,
explains the model with SHAP, rebuilds the demo page and prints a summary.
It takes about a minute.

| Step | File | Output |
| --- | --- | --- |
| Train | `src/train.py` | Logistic Regression (baseline) and LightGBM (main; base settings + 3 small changes). Tuning uses the last 20% of the train period; both are calibrated there. Stops if any data check failed. Saves `models/*.joblib`, `models/feature_list.json` |
| Evaluate | `src/evaluate.py` | Test-set-only metrics side by side, calibration chart, picks the recommended model (LightGBM only if it wins by >= 0.01 AUC). `reports/model_report.md` |
| Decide | `src/decide.py` | Tiers 0/1/2, three policies (do nothing / ask everyone / our tiers), sensitivity. `reports/decision_report.md` |
| Explain | `src/explain.py` | SHAP charts, comparison with the hidden rule, two example orders. `reports/figures/`, `reports/demo_orders.json` |
| Demo | `src/build_demo.py` | `demo/index.html`, one offline file. Open it in a browser; add `#B` to the URL to start on Order B |

| App data | `src/export_app.py` | `app_data/model_params.json` (the Logistic Regression as plain numbers) and `app_data/orders.csv` (all 76,232 scored orders + the raw counts behind each feature). Checks that the app rebuilds every feature and score exactly |

Shared model helpers (feature list + leakage assert, calibration, readable names) are in `src/modeling.py`.

## Interactive app (Streamlit)

The Streamlit app at the repo root (`app.py`) shows the **COD Risk Score (ML model)**
(code in `risk_ui/`). From the repo root:

```bash
python -m venv .venv
.venv/Scripts/activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt   # the root requirements (streamlit, plotly, ...)
streamlit run app.py
```

Open http://localhost:8501 (add `?page=orders`, `?page=score` or `?page=simulator`
to start on another page; `?page=performance` / `?page=import` open those pages). Pages:

| Page | What it does |
| --- | --- |
| Dashboard | Orders, mean predicted risk and predicted cost lost ((฿24 shipping out + ฿20 shipping back + 10% of the order value) x risk), risk distribution with cut-offs, tier table, predicted-risk trend (daily / weekly / monthly) |
| Orders | All 76,232 orders (sortable, downloadable). Click a row: score, tier, top 3 reasons, every factor's push, phone mockup |
| Score an order | Start from an example (typical buyer, new buyer, past refuser) or load any order, then edit the fields (each has a short description); the app builds the 20 features and scores live |
| Policy simulator | Move the 3 tier cut-offs and each tier's assumed "failures prevented" and "good orders lost"; see tier shape, three policies, net saving vs Tier 2 cut-off and a sensitivity table |
| Model performance | Predicted risk vs actual outcomes (synthetic data), Train (Feb-Jun) vs Test (Jul): AUC, Brier, top 30% / 5% capture, actual vs predicted by tier, calibration, capture and ROC curves, weekly actual vs predicted, new vs returning buyers, failure types caught, and Logistic Regression vs LightGBM. Each section says which data it compares. Always uses the demo data (imported files have no outcomes) |
| Import data | Download the template (Excel or CSV, 100 example rows), upload your own orders (.csv / .xlsx). A pop-up checks the file (column completeness, every check, the problem rows with reasons) and lets you import the rows that pass. The orders are scored with the same model; pick **Data → Imported file** in the sidebar to see them on every page |

The app shows only the model's predictions; actual outcomes are kept out of the UI.

App tiers (set in `risk_ui/model.py`, not in `config.yaml`):

| Tier | Predicted risk | Action |
| --- | --- | --- |
| 1 | below 3% | Normal COD + COD Reminder |
| 2 | 3% to 10% | COD Confirmation + Order Hold |
| 3 | 10% to 77% | Refundable Deposit 10% |
| 4 | 77% and above | COD → Prepaid Requirement |

The simulator's default catch / friction values per tier are assumptions (`SIM_DEFAULTS` in `risk_ui/model.py`).
The pipeline reports (`reports/decision_report.md`) still use the older 3-tier policy in `config.yaml` (`decision:`).

Filters: the Dashboard has its own row (date, tier, area, split). The Orders page has the full panel
(order / buyer ID, date, tier, plus "More filters" tabs for order, buyer, area and score); the Policy simulator
can run on the orders that match it. The two sets are independent and are kept when you switch pages.

The app does not load the joblib models (both projects have a package called `src`); it uses the exported
numbers, and `src/export_app.py` proves they give the same scores.

## Data steps in detail

| Step | File | Output |
| --- | --- | --- |
| 1. Simulate | `src/generate_data.py` | 30,000 buyers with hidden traits, 60 areas with hidden courier failure rates, and 18 months of COD orders (Feb 2025 to Jul 2026). Each outcome is drawn at random from a hidden log-odds rule. One intercept per failure type is tuned so Feb to Jul 2026 hits the targets. Writes `data/raw_events.csv` and `docs/hidden_rules.md`. |
| 2. Features | `src/build_features.py` | For every order from Feb to Jul 2026, computes features using only what was known at checkout. Adds labels and the time split. Writes `data/model_table.csv`, `train.csv`, `test.csv` and `docs/data_dictionary.md`. |
| 3. Validate | `src/validate_data.py` | Runs the checks, counts the edge cases, fits a small logistic-regression sanity model and writes `reports/validation_report.md`. |

Shared helpers are in `src/common.py`. Every setting is in `config.yaml`.

## Files

```
config.yaml                  every number: sizes, mixtures, effects, targets, seed
data/raw_events.csv          ALL orders incl. hidden values and failure_type (audit only)
data/model_table.csv         features + labels + split (no leakage columns)
data/train.csv               Feb to Jun 2026
data/test.csv                Jul 2026
docs/data_dictionary.md      every column: meaning, type, range, feature or not
docs/hidden_rules.md         the generator's true effects, to compare with what the model learns
reports/validation_report.md PASS/FAIL checks, edge cases, sanity model importance
```

## Ideas the demo relies on

- **Buyers are simulated over time, not row by row.** History features are
  real counts of earlier simulated orders, so they behave like real history.
- **History is evidence, not the cause.** The hidden rule uses the buyer's
  hidden `latent_wont`, not their past refusals. That is why
  `refusal_rate_smoothed` shrinks small samples toward the population rate:
  "1 order, 1 refusal" is weaker evidence than "40 orders, 15 refusals".
- **Courier failures are not blamed on buyers.** Logistics failures never enter
  refusal features, and the area feature only counts logistics failures.
- **No leakage.** History only counts past orders whose outcome was already
  known at checkout. `buyer_id` is kept for audit and splitting, never as a feature.
- **No protected traits.** No gender, age, religion or ethnicity data exists.
