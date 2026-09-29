"""Step 6: put test orders into decision tiers and simulate the money.

SYNTHETIC DATA - for illustration only. This is NOT real Shopee data.

Tier 0: p <  tier1_cutoff               -> nothing extra
Tier 1: tier1_cutoff <= p < tier2_cutoff -> one-tap "Confirm to Ship"
Tier 2: p >= tier2_cutoff                -> Confirm + prepaid / deposit / call

Money model (expected values, per asked order, all numbers in config.yaml):
  caught            = `wont` failures in the tier x catch rate  (these failures are avoided)
  shipping saved    = caught x cost of a failed delivery
  commission back   = caught x share that end up delivered x commission
  friction loss     = good orders in the tier x friction rate x commission
  message cost      = asked orders x message cost
  net value         = shipping saved + commission back - friction loss - message cost

Writes reports/decision_report.md and reports/decision_summary.json.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.common import DATA_DIR, REPORTS_DIR, SYNTHETIC_NOTE, load_config, read_csv, write_text
from src.modeling import TARGET, load_audit, read_json, write_json, MODELS_DIR

TIER_ACTIONS = {0: "Nothing extra", 1: "One-tap \"Confirm to Ship\"", 2: "Confirm + choose prepaid, deposit or call"}


def assign_tiers(p, dc):
    return np.select([p >= dc["tier2_cutoff"], p >= dc["tier1_cutoff"]], [2, 1], 0)


def simulate(tier, ftype, dc, catch, friction):
    """Expected money effect of asking the orders in Tier 1 and Tier 2.
    `catch` and `friction` are dicts {1: rate, 2: rate}."""
    caught = lost_good = asked = 0.0
    for k in (1, 2):
        in_tier = tier == k
        caught += ((ftype == "wont") & in_tier).sum() * catch[k]
        lost_good += ((ftype == "none") & in_tier).sum() * friction[k]
        asked += in_tier.sum()
    shipping = caught * dc["cost_failed_delivery_thb"]
    commission_back = caught * dc["caught_become_delivered"] * dc["commission_per_order_thb"]
    friction_loss = lost_good * dc["commission_per_order_thb"]
    messages = asked * dc["message_cost_thb"]
    return {"asked_share": asked / len(tier), "failures_avoided": caught, "shipping_saved": shipping,
            "commission_won_back": commission_back, "commission_lost_friction": friction_loss,
            "message_cost": messages, "net": shipping + commission_back - friction_loss - messages}


def base_rates(dc):
    return {1: dc["catch_rate"]["tier1"], 2: dc["catch_rate"]["tier2"]}, \
           {1: dc["friction"]["tier1"], 2: dc["friction"]["tier2"]}


def main():
    cfg = load_config()
    dc = cfg["decision"]
    rec = read_json(MODELS_DIR / "recommended.json")
    df = read_csv(DATA_DIR / "test_scored.csv").merge(load_audit()[["order_id", "failure_type"]], on="order_id")
    p, y, ftype = df["p"].to_numpy(), df[TARGET].to_numpy(), df["failure_type"].to_numpy()
    n = len(df)
    yearly = dc["yearly_orders_total"] * dc["cod_share"]
    per_k, per_year = 1000 / n, yearly / n

    # ---- Tiers ----
    tier = assign_tiers(p, dc)
    tiers = []
    for k in (0, 1, 2):
        m = tier == k
        tiers.append({"tier": k, "orders": int(m.sum()), "order_share": m.mean(), "failure_share": y[m].sum() / y.sum(),
                      "failure_rate": y[m].mean() if m.any() else float("nan"),
                      "mean_predicted": p[m].mean() if m.any() else float("nan"),
                      "wont_share_of_failures": (ftype[m & (y == 1)] == "wont").mean() if (m & (y == 1)).any() else 0})

    # ---- Three policies ----
    catch, friction = base_rates(dc)
    policies = {
        "Do nothing": simulate(np.zeros(n, int), ftype, dc, catch, friction),
        "Ask everyone": simulate(np.ones(n, int), ftype, dc, catch, friction),
        "Our tiers": simulate(tier, ftype, dc, catch, friction),
    }

    # ---- Sensitivity: Tier 1 catch rate and friction (Tier 2 = catch + 15 pts, friction x2) ----
    gap = dc["catch_rate"]["tier2"] - dc["catch_rate"]["tier1"]
    ratio = dc["friction"]["tier2"] / dc["friction"]["tier1"]
    sens = []
    for c in dc["sensitivity_catch_rates"]:
        for f in dc["sensitivity_friction"]:
            cr, fr = {1: c, 2: min(c + gap, 1.0)}, {1: f, 2: f * ratio}
            sens.append({"catch": c, "friction": f,
                         "ours": simulate(tier, ftype, dc, cr, fr)["net"] * per_year,
                         "everyone": simulate(np.ones(n, int), ftype, dc, cr, fr)["net"] * per_year})

    # ---- Report ----
    thb = lambda x: f"฿{x:,.0f}"
    mthb = lambda x: f"{'-' if x < 0 else ''}฿{abs(x) / 1e6:,.1f}M"
    cmthb = lambda x: f"THB {x / 1e6:,.1f}M"  # console version (Windows console has no ฿)
    out = [
        "# Decision tiers and money simulation", "", f"> {SYNTHETIC_NOTE}", "",
        "**These results show how the system works on synthetic data. "
        "Real accuracy will be measured in the pilot with Shopee data.**", "",
        f"Score used: calibrated probability from the recommended model (**{rec['name']}**) on the July 2026 test set "
        f"({n:,} COD orders).", "",
        "## Tiers", "",
        "| Tier | Rule | Action | Share of orders | Share of failures | Actual failure rate | Mean predicted | `wont` share of its failures |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    rules = {0: f"p < {dc['tier1_cutoff']:.1%}", 1: f"{dc['tier1_cutoff']:.1%} <= p < {dc['tier2_cutoff']:.0%}",
             2: f"p >= {dc['tier2_cutoff']:.0%}"}
    out += [f"| Tier {t['tier']} | {rules[t['tier']]} | {TIER_ACTIONS[t['tier']]} | {t['order_share']:.1%} | "
            f"{t['failure_share']:.1%} | {t['failure_rate']:.2%} | {t['mean_predicted']:.2%} | {t['wont_share_of_failures']:.0%} |"
            for t in tiers]
    # Reference: cut-offs that would force the 70 / 25 / 5 shape (chosen by rank, not by money).
    q_lo, q_hi = np.quantile(p, 0.70), np.quantile(p, 0.95)
    shape_tier = np.select([p >= q_hi, p >= q_lo], [2, 1], 0)
    shape_net = simulate(shape_tier, ftype, dc, *base_rates(dc))["net"]
    shape_fail = [y[shape_tier == k].sum() / y.sum() for k in (0, 1, 2)]
    target_note = (
        "The target shape was about 70% / 25% / 5% of orders. We got "
        f"{' / '.join(f'{t['order_share']:.0%}' for t in tiers)}. The model's scores are fairly compressed "
        f"(the signal in the data is limited, test AUC {roc_auc_score(y, p):.3f}), so more orders sit just above "
        f"{dc['tier1_cutoff']:.1%} and fewer reach {dc['tier2_cutoff']:.0%}. Tier 0 still has a lower failure rate "
        f"than average, but it holds {tiers[0]['failure_share']:.0%} of failures, which is not small.\n\n"
        f"The {dc['tier1_cutoff']:.1%} cut-off is close to the money break-even point: asking a Tier 1 buyer pays off "
        "when the expected refusal saving beats the friction and message cost. "
        f"For reference, cut-offs chosen by rank to force the 70/25/5 shape (p >= {q_lo:.2%} and p >= {q_hi:.2%}) "
        f"would put {' / '.join(f'{s:.0%}' for s in shape_fail)} of failures in Tiers 0 / 1 / 2 and give a net value of "
        f"{mthb(shape_net * per_year)} a year (vs {mthb(simulate(tier, ftype, dc, *base_rates(dc))['net'] * per_year)} "
        "with the money-based cut-offs). That is slightly more, mainly because more orders get the stronger Tier 2 "
        "step. With these assumptions, Tier 2 (+15 pts catch, +0.5 pt friction) already pays for itself from roughly "
        "p = 3.5%, so the 10% Tier 2 cut-off is conservative. We keep it because the Tier 2 step (deposit, call) is "
        "heavier for the buyer than the model counts; it is worth revisiting in the pilot.")
    out += ["", target_note, ""]

    out += ["## Three policies", "",
            f"Assumptions (config.yaml): failed delivery costs {thb(dc['cost_failed_delivery_thb'])}, commission "
            f"{thb(dc['commission_per_order_thb'])} per order; Confirm catches {dc['catch_rate']['tier1']:.0%} (Tier 1) / "
            f"{dc['catch_rate']['tier2']:.0%} (Tier 2) of `wont` failures; {dc['caught_become_delivered']:.0%} of those "
            f"caught end up delivered (the rest are cancelled before shipping at no cost); friction loses "
            f"{dc['friction']['tier1']:.1%} / {dc['friction']['tier2']:.1%} of good orders; each message costs "
            f"฿{dc['message_cost_thb']:.2f}. \"Ask everyone\" treats every order as Tier 1. Yearly scale: "
            f"{dc['yearly_orders_total'] / 1e6:,.1f}M orders x {dc['cod_share']:.0%} COD = {yearly / 1e6:,.1f}M COD orders.", "",
            "### Per 1,000 COD orders", "",
            "| Policy | Orders asked | Failures avoided | Shipping saved | Commission won back | Commission lost to friction | Message cost | **Net value** |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in policies.items():
        out.append(f"| {name} | {r['asked_share']:.0%} | {r['failures_avoided'] * per_k:.2f} | {r['shipping_saved'] * per_k:,.1f} | "
                   f"{r['commission_won_back'] * per_k:,.1f} | -{r['commission_lost_friction'] * per_k:,.1f} | "
                   f"-{r['message_cost'] * per_k:,.1f} | **฿{r['net'] * per_k:,.1f}** |")
    out += ["", "### Scaled to one year", "",
            "| Policy | Failures avoided | Shipping saved | Commission won back | Commission lost to friction | Message cost | **Net value** |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in policies.items():
        out.append(f"| {name} | {r['failures_avoided'] * per_year / 1e6:,.2f}M | {mthb(r['shipping_saved'] * per_year)} | "
                   f"{mthb(r['commission_won_back'] * per_year)} | -{mthb(r['commission_lost_friction'] * per_year)} | "
                   f"-{mthb(r['message_cost'] * per_year)} | **{mthb(r['net'] * per_year)}** |")

    ours, every = policies["Our tiers"]["net"], policies["Ask everyone"]["net"]
    if ours > every:
        verdict = (f"**\"Our tiers\" beats \"Ask everyone\"** by {mthb((ours - every) * per_year)} a year, while asking only "
                   f"{policies['Our tiers']['asked_share']:.0%} of buyers instead of 100%. Asking everyone catches more "
                   "refusals, but most of the extra buyers it bothers would never have failed, so the friction and "
                   "message costs eat the gain.")
    else:
        verdict = ("**\"Ask everyone\" is NOT worse than \"Our tiers\" with these assumptions.** "
                   f"Net per year: ask everyone {mthb(every * per_year)}, our tiers {mthb(ours * per_year)}.")
    out += ["", verdict, ""]

    out += ["## Sensitivity (net value per year)", "",
            f"Tier 1 catch rate and friction change; Tier 2 keeps a +{gap * 100:.0f} point higher catch rate and "
            f"{ratio:g}x the friction.", "",
            "| Tier 1 catch rate | Tier 1 friction | Our tiers | Ask everyone | Our tiers better? |",
            "| --- | --- | --- | --- | --- |"]
    out += [f"| {s['catch']:.0%} | {s['friction']:.1%} | {mthb(s['ours'])} | {mthb(s['everyone'])} | "
            f"{'yes' if s['ours'] > s['everyone'] else 'NO'} |" for s in sens]
    wins = sum(s["ours"] > s["everyone"] for s in sens)
    out += ["", f"Our tiers beat asking everyone in {wins} of {len(sens)} scenarios.", ""]
    write_text(REPORTS_DIR / "decision_report.md", "\n".join(out))

    write_json(REPORTS_DIR / "decision_summary.json", {
        "note": "Synthetic data, for illustration only. Not Shopee data.",
        "model": rec["name"],
        "tiers": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()} for t in tiers],
        "net_per_year_thb": {k: round(v["net"] * per_year) for k, v in policies.items()},
        "net_per_1000_thb": {k: round(v["net"] * per_k, 2) for k, v in policies.items()},
        "sensitivity_wins": f"{wins}/{len(sens)}",
    })

    print("Tiers (orders / failures / failure rate):")
    for t in tiers:
        print(f"  Tier {t['tier']}: {t['order_share']:.1%} / {t['failure_share']:.1%} / {t['failure_rate']:.2%}")
    for name, r in policies.items():
        print(f"  {name:<13} net per 1,000 orders THB {r['net'] * per_k:8,.1f}   per year {cmthb(r["net"] * per_year)}")
    print(f"  Sensitivity: our tiers better in {wins}/{len(sens)} scenarios")
    print("Wrote reports/decision_report.md, reports/decision_summary.json")


if __name__ == "__main__":
    main()
