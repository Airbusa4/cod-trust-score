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

The Streamlit app at the repo root (`app.py`) has a **System** switch in the sidebar:
the original "COD Trust Score (Scorecard + LLM)" and this project's **"COD Risk Score (ML model)"**
(code in `risk_ui/`). From the repo root:

```bash
python -m venv .venv
.venv/Scripts/activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt   # the root requirements (streamlit, plotly, ...)
streamlit run app.py
```

Open http://localhost:8501/?system=ml to go straight to the ML pages
(add `&page=orders`, `&page=score` or `&page=simulator`). Pages:

| Page | What it does |
| --- | --- |
| Dashboard | KPIs, risk distribution with cut-offs, tier table, failure types per tier, capture curve, calibration, breakdown by 14 segments, areas + fairness check, weekly trend |
| Orders | All 76,232 orders (sortable, downloadable). Click a row: score, tier, top 3 reasons, every factor's push, phone mockup, and the simulation truth |
| Score an order | Type raw checkout data (history counts, value, time, area ...); the app builds the 20 features and scores live. "Load into form" copies any existing order |
| Policy simulator | Move tier cut-offs, catch rates, friction and message cost; see tier shape, the three policies, net value vs cut-off and a sensitivity table |

Every filter in the sidebar (date, split, month, value, hour, history, refusals, account age, area,
risk, tier, actual outcome, order / buyer ID ...) applies to Dashboard, Orders and (optionally) the simulator,
and is kept when you switch pages. Feb-Jun orders were used for training, so filter **Split = test** for honest numbers.

The app does not load the joblib models (both projects have a package called `src`); it uses the exported
numbers, and `src/export_app.py` proves they give the same scores.
Tier cut-offs and money assumptions are under `decision:` in `config.yaml`.

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
