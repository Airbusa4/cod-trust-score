"""COD Risk Score (ML model) pages for the Streamlit app.

SYNTHETIC DATA - for illustration only. This is NOT real Shopee data.

Pages:
  Dashboard         - KPIs, risk distribution, tiers, capture curve, segments,
                      areas (incl. fairness), calibration, weekly trend
  Orders            - every order (filterable, sortable) + full detail of one order
  Score an order    - type raw order / buyer data, get the risk score, tier and reasons
  Policy simulator  - move the cut-offs and assumptions, see the money change

Data comes from cod_risk_demo/app_data/ (built by `python -m src.export_app`).
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from risk_ui.model import (NICE_NAMES, OUTCOME_NAMES, TIER_ACTIONS, TIER_COLORS, TIER_NAMES, Model,
                           assign_tiers, features_from_inputs, simulate, top_reasons)

APP_DATA = Path(__file__).resolve().parent.parent / "cod_risk_demo" / "app_data"
NOTE = "Synthetic data, for illustration only. Not Shopee data."
PAGES = ["Dashboard", "Orders", "Score an order", "Policy simulator"]
TIER_COLOR_BY_NAME = {TIER_NAMES[k]: v for k, v in TIER_COLORS.items()}
OUTCOME_COLORS = {"Delivered": "#9aa0a6", "Refused (won't)": "#d93025",
                  "Not home / no cash (can't)": "#c77700", "Courier problem (logistics)": "#1a73e8"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading orders ...")
def load():
    params = json.loads((APP_DATA / "model_params.json").read_text(encoding="utf-8"))
    df = pd.read_csv(APP_DATA / "orders.csv", comment="#")
    df["order_datetime"] = pd.to_datetime(df["order_datetime"])
    df["outcome"] = df["failure_type"].map(OUTCOME_NAMES)
    df["tier_name"] = df["tier"].map(TIER_NAMES)
    df["risk_pct"] = df["risk"] * 100
    return params, df


@st.cache_resource
def get_model(params_json):
    return Model(json.loads(params_json))


def show(fig, height=380):
    """Plot with the synthetic-data note in the corner."""
    title = fig.layout.title.text or ""
    fig.update_layout(title_text=f"{title}<br><sup style='color:#888'>{NOTE}</sup>", height=height,
                      margin=dict(t=70, b=50, l=10, r=10))
    st.plotly_chart(fig, width="stretch")


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


# ---------------------------------------------------------------------------
# Filters (sidebar). Every widget key starts with "f_" so it can be reset.
# ---------------------------------------------------------------------------
FILTERS = {
    "Order": [
        ("order_datetime", "date", "Order date"),
        ("split", "multi", "Split (train = Feb-Jun, test = Jul)"),
        ("order_month", "multi", "Order month"),
        ("order_value", "range", "Order value (THB)"),
        ("value_vs_aov", "range", "Value vs buyer's usual (x)"),
        ("order_hour", "range", "Order hour"),
        ("is_late_night", "bool", "Late night (01:00-04:59)"),
        ("is_campaign_day", "bool", "Campaign day"),
        ("same_item_other_shops_48h", "range", "Same item at other shops (48h)"),
        ("expected_days_to_delivery", "range", "Expected delivery days"),
        ("freq_change_ratio", "range", "Order frequency change (30d)"),
    ],
    "Buyer": [
        ("is_new_cod_buyer", "bool", "First COD order (no history)"),
        ("hist_cod_orders", "range", "Past COD orders"),
        ("hist_refusals", "range", "Past refusals"),
        ("has_ever_refused", "bool", "Has ever refused"),
        ("refusal_rate_smoothed", "range", "Refusal rate (smoothed)"),
        ("recent_refusal_rate_smoothed_90d", "range", "Refusal rate, last 90 days"),
        ("days_since_last_refusal", "range", "Days since last refusal (-1 = never)"),
        ("hist_buyer_caused_misses", "range", "Past 'not home / no cash' misses"),
        ("hist_courier_caused_failures", "range", "Past courier failures"),
        ("account_age_days", "range", "Account age (days)"),
        ("address_is_condo_with_office", "bool", "Condo with front office"),
    ],
    "Area": [
        ("area_id", "multi", "Area"),
        ("area_logistics_failure_rate", "range", "Area courier failure rate (90d)"),
    ],
    "Score & outcome": [
        ("risk_pct", "range", "Risk score (%)"),
        ("tier", "multi", "Tier"),
        ("outcome", "multi", "Actual outcome (synthetic truth)"),
    ],
}


def reset_filters():
    for k in [k for k in st.session_state if k.startswith("f_")]:
        del st.session_state[k]


def filter_panel(df):
    """Draw every filter in the sidebar and return the filtered orders."""
    sb = st.sidebar
    sb.markdown("### Filters")
    sb.button("Reset all filters", on_click=reset_filters, width="stretch")
    mask = pd.Series(True, index=df.index)
    active = 0

    with sb.expander("Search by ID", expanded=False):
        ids = st.text_input("Order IDs (comma separated)", key="f_order_ids")
        buyer = st.text_input("Buyer ID", key="f_buyer_id")
    if ids.strip():
        wanted = [int(x) for x in ids.replace(" ", "").split(",") if x.isdigit()]
        mask &= df["order_id"].isin(wanted)
        active += 1
    if buyer.strip().isdigit():
        mask &= df["buyer_id"] == int(buyer)
        active += 1

    for group, specs in FILTERS.items():
        with sb.expander(group, expanded=(group == "Score & outcome")):
            for col, kind, label in specs:
                key = f"f_{col}"
                s = df[col]
                if kind == "date":
                    lo, hi = s.min().date(), s.max().date()
                    init = {} if key in st.session_state else {"value": (lo, hi)}
                    val = st.date_input(label, min_value=lo, max_value=hi, key=key, **init)
                    if isinstance(val, (tuple, list)) and len(val) == 2 and (val[0] > lo or val[1] < hi):
                        mask &= s.dt.date.between(val[0], val[1])
                        active += 1
                elif kind == "multi":
                    opts = sorted(s.dropna().unique().tolist())
                    fmt = (lambda t: TIER_NAMES[t]) if col == "tier" else str
                    val = st.multiselect(label, opts, key=key, format_func=fmt, placeholder="All")
                    if val:
                        mask &= s.isin(val)
                        active += 1
                elif kind == "bool":
                    val = st.selectbox(label, ["Any", "Yes", "No"], key=key)
                    if val != "Any":
                        mask &= s == (1 if val == "Yes" else 0)
                        active += 1
                else:  # range
                    is_int = pd.api.types.is_integer_dtype(s)
                    lo, hi = (int(s.min()), int(s.max())) if is_int else (float(s.min()), float(s.max()))
                    if lo == hi:
                        continue
                    step = 1 if is_int else float(f"{(hi - lo) / 200:.2g}")
                    init = {} if key in st.session_state else {"value": (lo, hi)}
                    val = st.slider(label, lo, hi, step=step, key=key, **init)
                    if val[0] > lo or val[1] < hi:
                        mask &= s.between(val[0] - 1e-9, val[1] + 1e-9)
                        active += 1

    out = df[mask]
    sb.caption(f"**{len(out):,}** of {len(df):,} orders · {active} filter(s) active")
    return out


def in_sample_warning(d):
    if (d["split"] == "train").any():
        st.info("Feb-Jun orders were used to train the model, so their scores look better than they "
                "really are. Filter **Split = test** (July) for honest numbers.", icon="ℹ️")


# ---------------------------------------------------------------------------
# One order: facts, score, reasons, phone
# ---------------------------------------------------------------------------
PHONE_CSS = """
<style>
.phone{width:300px;height:560px;border-radius:40px;background:#111;padding:12px;margin:0 auto}
.screen{background:#fff;border-radius:30px;height:100%;overflow:hidden;display:flex;flex-direction:column;color:#1d2330}
.appbar{background:#ee4d2d;color:#fff;font-weight:700;padding:28px 16px 12px;font-size:17px}
.pbody{padding:16px;display:flex;flex-direction:column;gap:10px}
.ptitle{font-size:21px;font-weight:750;line-height:1.2}.psub{font-size:14px;color:#5d6677}
.pbtn{font-size:15px;font-weight:650;padding:11px;border-radius:10px;text-align:center;border:2px solid #e3e6ec}
.main{background:#ee4d2d;color:#fff;border-color:#ee4d2d}.offer{background:#fff4e0;border-color:#f3d19c;color:#7a4a00;text-align:left}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.check{width:64px;height:64px;border-radius:50%;background:#e6f4ea;color:#1e8e3e;font-size:36px;display:flex;
       align-items:center;justify-content:center;margin:24px auto 4px}
.badge{display:inline-block;font-weight:750;font-size:20px;padding:6px 18px;border-radius:999px}
</style>"""


def phone_html(tier, value):
    v = f"฿{value:,.0f}"
    if tier == 0:
        body = (f'<div class="check">✓</div><div class="ptitle" style="text-align:center">Order placed</div>'
                f'<div class="psub" style="text-align:center">{v} · Cash on delivery.<br>Pay the courier when your parcel arrives.</div>')
    else:
        body = (f'<div class="ptitle">Ready to ship {v} COD</div>'
                '<div class="psub">Please confirm you will be home to receive and pay for this parcel.</div>'
                '<div class="pbtn main">Confirm</div><div class="row2"><div class="pbtn">Cancel</div><div class="pbtn">Pay now</div></div>')
        if tier == 2:
            body += ('<div class="pbtn offer">Pay now and get ฿10 coins</div>'
                     '<div class="pbtn offer">Keep COD with ฿20 deposit</div>')
    return f'{PHONE_CSS}<div class="phone"><div class="screen"><div class="appbar">Checkout</div><div class="pbody">{body}</div></div></div>'


def order_view(feats, order_hour, model, params, extra=None):
    """Show one order. `feats` = Series of the 20 features. `extra` = dict of audit info (optional)."""
    X = feats.to_frame().T.astype(float)
    p = float(model.predict(X)[0])
    tier = int(assign_tiers(np.array([p]), params["decision"])[0])
    reasons, phi = top_reasons(model, X, order_hour)
    avg = params["population_failure_rate"]

    left, mid, right = st.columns([1.1, 1.3, 0.9])
    with left:
        st.markdown("##### Risk of failed delivery")
        st.markdown(f"<div style='font-size:56px;font-weight:800;color:{TIER_COLORS[tier]};line-height:1'>{p:.1%}</div>"
                    f"<div style='color:#5d6677'>vs about {avg:.1%} on average ({p / avg:.1f}x)</div>",
                    unsafe_allow_html=True)
        bg = {0: "#e6f4ea", 1: "#fff4e0", 2: "#fce8e6"}[tier]
        st.markdown(f"{PHONE_CSS}<div class='badge' style='background:{bg};color:{TIER_COLORS[tier]};margin:12px 0'>"
                    f"{TIER_NAMES[tier]}</div>", unsafe_allow_html=True)
        st.markdown(f"**Action:** {TIER_ACTIONS[tier]}")
        st.markdown("##### Top 3 reasons")
        for r in reasons:
            arrow, color, word = ("▲", "#d93025", "raises") if r["direction"] == "up" else ("▼", "#1e8e3e", "lowers")
            st.markdown(f"<div style='font-size:17px;margin:6px 0'><span style='color:{color};font-weight:800'>{arrow}</span> "
                        f"<b>{r['text']}</b><br><span style='color:#5d6677;font-size:13px'>{word} risk · {r['name']} "
                        f"({r['shap']:+.2f} log-odds)</span></div>", unsafe_allow_html=True)
    with mid:
        st.markdown("##### What pushes this order's risk (all factors)")
        contrib = pd.DataFrame({"factor": [NICE_NAMES[f] for f in model.features], "push": phi})
        contrib = contrib.reindex(contrib["push"].abs().sort_values().index)
        fig = go.Figure(go.Bar(x=contrib["push"], y=contrib["factor"], orientation="h",
                               marker_color=np.where(contrib["push"] > 0, "#d93025", "#1e8e3e")))
        fig.update_layout(xaxis_title="push on risk (log-odds, vs an average order)", yaxis_title=None)
        show(fig, height=520)
    with right:
        st.markdown("##### What the buyer sees")
        st.markdown(phone_html(tier, float(feats["order_value"])), unsafe_allow_html=True)

    with st.expander("Model inputs (the 20 features)"):
        st.dataframe(pd.DataFrame({"feature": [NICE_NAMES[f] for f in model.features], "column": model.features,
                                   "value": [feats[f] for f in model.features],
                                   "push (log-odds)": np.round(phi, 3)}), hide_index=True, width="stretch")
    if extra:
        with st.expander("Simulation truth (audit only, never a model input)"):
            st.markdown(f"- Actual outcome: **{extra['outcome']}**\n"
                        f"- True failure chance used by the generator: **{extra['p_true']:.2%}** "
                        f"(model said {p:.2%})")
    return p, tier


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------
def seg_hour(d):
    return pd.Series(np.select([d.order_hour.between(1, 4), d.order_hour.between(5, 11), d.order_hour.between(12, 17)],
                               ["01-04 (late night)", "05-11", "12-17"], "18-00"), index=d.index)


SEGMENTS = {
    "Past COD orders": lambda d: pd.cut(d.hist_cod_orders, [-1, 0, 4, 19, 1e9], labels=["0 (new)", "1-4", "5-19", "20+"]),
    "Has ever refused": lambda d: d.has_ever_refused.map({0: "No", 1: "Yes"}),
    "Account age": lambda d: pd.cut(d.account_age_days, [-1, 29, 89, 364, 1e9], labels=["<30 days", "30-89", "90-364", "1 year+"]),
    "Order hour": seg_hour,
    "Value vs usual": lambda d: pd.cut(d.value_vs_aov, [0, 0.9999, 1.0001, 2, 3, 1e9], labels=["<1x", "1x (or new)", "1-2x", "2-3x", "3x+"]),
    "Same item at other shops (48h)": lambda d: d.same_item_other_shops_48h.astype(str),
    "Frequency change (30d)": lambda d: pd.cut(d.freq_change_ratio, [-1, 1, 3, 1e9], labels=["<1x", "1-3x", "3x+"]),
    "Campaign day": lambda d: d.is_campaign_day.map({0: "No", 1: "Yes"}),
    "Expected delivery days": lambda d: d.expected_days_to_delivery.astype(str),
    "Condo with front office": lambda d: d.address_is_condo_with_office.map({0: "No", 1: "Yes"}),
    "Order month": lambda d: d.order_month.map(lambda m: dt.date(2026, m, 1).strftime("%b")),
    "Area courier failure rate": lambda d: pd.qcut(d.area_logistics_failure_rate.rank(method="first"), 4,
                                                   labels=["lowest 25%", "2nd", "3rd", "highest 25%"]),
    "Tier": lambda d: d.tier_name,
    "Split": lambda d: d.split,
}


def page_dashboard(d, params):
    st.title("COD Risk Score · Dashboard")
    st.caption(NOTE + " Numbers below follow the sidebar filters.")
    if d.empty:
        st.warning("No orders match the filters.")
        return
    in_sample_warning(d)
    dc = params["decision"]
    y = d["label_failed"]
    asked = d["tier"] >= 1
    money = simulate(d["tier"], d["failure_type"], dc, {1: dc["catch_rate"]["tier1"], 2: dc["catch_rate"]["tier2"]},
                     {1: dc["friction"]["tier1"], 2: dc["friction"]["tier2"]})

    c = st.columns(6)
    c[0].metric("Orders", f"{len(d):,}")
    c[1].metric("Actual failure rate", pct(y.mean(), 2))
    c[2].metric("Mean predicted risk", pct(d["risk"].mean(), 2), f"{(d['risk'].mean() - y.mean()) * 100:+.2f} pts vs actual",
                delta_color="off")
    c[3].metric("Orders asked (Tier 1+2)", pct(asked.mean()))
    c[4].metric("Failures in Tier 1+2", pct(y[asked].sum() / max(y.sum(), 1)))
    c[5].metric("Net value / 1,000 orders", f"฿{money['net'] / len(d) * 1000:,.0f}")

    left, right = st.columns([1.3, 1])
    with left:
        h = d.assign(risk_shown=d["risk_pct"].clip(upper=30))
        fig = px.histogram(h, x="risk_shown", color="tier_name", nbins=120, color_discrete_map=TIER_COLOR_BY_NAME,
                           category_orders={"tier_name": list(TIER_NAMES.values())},
                           labels={"risk_shown": "Risk score (%) (30+ grouped at 30)", "tier_name": "Tier"})
        for cut in (dc["tier1_cutoff"], dc["tier2_cutoff"]):
            fig.add_vline(x=cut * 100, line_dash="dash", line_color="#333",
                          annotation_text=f"{cut:.1%}", annotation_position="top")
        fig.update_layout(title="Risk score distribution", bargap=0.02, yaxis_title="Orders")
        show(fig)
    with right:
        st.markdown("##### Tiers")
        t = d.groupby("tier").agg(orders=("order_id", "size"), failures=("label_failed", "sum"),
                                  actual=("label_failed", "mean"), predicted=("risk", "mean")).reindex([0, 1, 2]).fillna(0)
        st.dataframe(pd.DataFrame({
            "Tier": [TIER_NAMES[k] for k in t.index],
            "Share of orders": (t["orders"] / t["orders"].sum()).map(pct),
            "Share of failures": (t["failures"] / max(t["failures"].sum(), 1)).map(pct),
            "Actual rate": t["actual"].map(lambda x: pct(x, 2)), "Mean predicted": t["predicted"].map(lambda x: pct(x, 2)),
        }), hide_index=True, width="stretch")
        mix = d[d["label_failed"] == 1].groupby(["tier_name", "outcome"]).size().reset_index(name="n")
        mix["share"] = mix["n"] / mix.groupby("tier_name")["n"].transform("sum") * 100
        fig = px.bar(mix, x="tier_name", y="share", color="outcome", color_discrete_map=OUTCOME_COLORS,
                     category_orders={"tier_name": list(TIER_NAMES.values())}, hover_data={"n": True},
                     labels={"tier_name": "", "share": "% of the tier's failures", "outcome": "Failure type"})
        fig.update_layout(title="What kind of failures each tier holds")
        show(fig, height=330)

    left, right = st.columns(2)
    with left:
        s = d.sort_values("risk", ascending=False)
        share_orders = np.arange(1, len(s) + 1) / len(s)
        share_fail = s["label_failed"].cumsum().to_numpy() / max(s["label_failed"].sum(), 1)
        idx = np.unique(np.linspace(0, len(s) - 1, 300).astype(int))
        fig = go.Figure([go.Scatter(x=share_orders[idx] * 100, y=share_fail[idx] * 100, name="Model", line=dict(width=3)),
                         go.Scatter(x=[0, 100], y=[0, 100], name="Random", line=dict(dash="dash", color="#999"))])
        at30 = share_fail[max(int(len(s) * 0.3) - 1, 0)]
        fig.add_annotation(x=30, y=at30 * 100, text=f"ask top 30% → find {at30:.0%} of failures", showarrow=True)
        fig.update_layout(title="Capture curve: ask X% of orders, find Y% of failures",
                          xaxis_title="% of orders asked (highest risk first)", yaxis_title="% of failures found")
        show(fig)
    with right:
        g = d.assign(group=pd.qcut(d["risk"].rank(method="first"), min(10, len(d)), labels=False) + 1)
        cal = g.groupby("group").agg(predicted=("risk", "mean"), actual=("label_failed", "mean"), n=("order_id", "size"))
        top = max(cal["predicted"].max(), cal["actual"].max()) * 110
        fig = go.Figure([go.Scatter(x=cal["predicted"] * 100, y=cal["actual"] * 100, mode="lines+markers",
                                    name="Model", text=cal["n"], hovertemplate="pred %{x:.2f}%<br>actual %{y:.2f}%<br>%{text} orders"),
                         go.Scatter(x=[0, top], y=[0, top], name="Perfect", line=dict(dash="dash", color="#999"))])
        fig.update_layout(title="Calibration: predicted vs actual (10 groups)",
                          xaxis_title="Predicted failure chance (%)", yaxis_title="Actual failure rate (%)")
        show(fig)

    st.subheader("Break down by segment")
    seg_name = st.selectbox("Segment", list(SEGMENTS), key="seg_choice")
    try:
        seg_raw = SEGMENTS[seg_name](d)
    except ValueError:
        st.info("Not enough orders for this breakdown.")
        return
    if isinstance(seg_raw.dtype, pd.CategoricalDtype):
        order = [str(x) for x in seg_raw.cat.categories]
    elif seg_name == "Order hour":
        order = ["01-04 (late night)", "05-11", "12-17", "18-00"]
    elif seg_name == "Order month":
        order = [dt.date(2026, m, 1).strftime("%b") for m in range(1, 13)]
    else:
        order = sorted(seg_raw.astype(str).unique(), key=lambda x: (len(x), x))
    sg = d.assign(segment=seg_raw.astype(str)).groupby("segment").agg(
        orders=("order_id", "size"), actual=("label_failed", "mean"), predicted=("risk", "mean"),
        asked=("tier", lambda x: (x >= 1).mean()))
    sg = sg.reindex([o for o in order if o in sg.index])
    left, right = st.columns([1.4, 1])
    with left:
        long = sg.reset_index().melt(id_vars="segment", value_vars=["actual", "predicted"], var_name="rate", value_name="v")
        fig = px.bar(long, x="segment", y=long["v"] * 100, color="rate", barmode="group",
                     color_discrete_map={"actual": "#1d2330", "predicted": "#ee4d2d"},
                     labels={"y": "Failure rate (%)", "segment": seg_name})
        fig.update_layout(title=f"Failure rate by segment ({seg_name}): actual vs predicted")
        show(fig)
    with right:
        st.dataframe(pd.DataFrame({"Segment": sg.index, "Orders": sg["orders"].map("{:,}".format),
                                   "Actual rate": sg["actual"].map(lambda x: pct(x, 2)),
                                   "Predicted": sg["predicted"].map(lambda x: pct(x, 2)),
                                   "Asked (Tier 1+2)": sg["asked"].map(pct)}), hide_index=True, width="stretch")

    st.subheader("Areas: buyer problems vs courier problems")
    st.caption("A good score should flag buyer behaviour, not a poor courier network. The last column shows how often "
               "buyers with a CLEAN record (5+ past orders, no refusals or misses) still get asked to confirm.")
    clean = (d["hist_cod_orders"] >= 5) & (d["hist_refusals"] == 0) & (d["hist_buyer_caused_misses"] == 0)
    a = d.assign(wont=d.failure_type.eq("wont"), cant=d.failure_type.eq("cant"), logi=d.failure_type.eq("logistics"),
                 clean_asked=np.where(clean, d["tier"] >= 1, np.nan))
    ar = a.groupby("area_id").agg(orders=("order_id", "size"), actual=("label_failed", "mean"), wont=("wont", "mean"),
                                  cant=("cant", "mean"), logistics=("logi", "mean"), predicted=("risk", "mean"),
                                  area_rate=("area_logistics_failure_rate", "mean"), clean_asked=("clean_asked", "mean"))
    left, right = st.columns([1, 1.2])
    with left:
        fig = px.scatter(ar.reset_index(), x=ar["area_rate"] * 100, y=ar["clean_asked"] * 100, size="orders",
                         hover_name="area_id", labels={"x": "Area courier failure rate (%)", "y": "% of clean buyers asked"})
        fig.update_layout(title="Fairness check: are clean buyers in poor courier areas asked more?")
        show(fig)
    with right:
        st.dataframe(pd.DataFrame({
            "Area": ar.index, "Orders": ar["orders"], "Failure rate": ar["actual"].map(lambda x: pct(x, 2)),
            "Won't": ar["wont"].map(lambda x: pct(x, 2)), "Can't": ar["cant"].map(lambda x: pct(x, 2)),
            "Logistics": ar["logistics"].map(lambda x: pct(x, 2)), "Predicted": ar["predicted"].map(lambda x: pct(x, 2)),
            "Clean buyers asked": ar["clean_asked"].map(lambda x: "-" if pd.isna(x) else pct(x))}),
            hide_index=True, width="stretch", height=380)

    st.subheader("Weekly trend")
    w = d.set_index("order_datetime").resample("W").agg({"label_failed": "mean", "risk": "mean", "order_id": "size"})
    w = w[w["order_id"] > 0]
    fig = go.Figure([go.Scatter(x=w.index, y=w["label_failed"] * 100, name="Actual", mode="lines+markers",
                                line=dict(color="#9aa0a6")),
                     go.Scatter(x=w.index, y=w["risk"] * 100, name="Predicted", mode="lines+markers",
                                line=dict(color="#ee4d2d", width=3))])
    fig.update_layout(title="Failure rate per week: actual vs predicted (a growing gap = the model needs a refresh)",
                      yaxis_title="Failure rate (%)")
    show(fig, height=340)


# ---------------------------------------------------------------------------
# Page: Orders
# ---------------------------------------------------------------------------
def page_orders(d, all_orders, params, model):
    st.title("COD Risk Score · Orders")
    st.caption(NOTE + " Click a row to see that order in full. Click a column header to sort.")
    in_sample_warning(d)
    if d.empty:
        st.warning("No orders match the filters.")
        return
    view = d.sort_values("risk", ascending=False)[[
        "order_id", "order_datetime", "buyer_id", "area_id", "risk_pct", "tier_name", "outcome", "order_value",
        "value_vs_aov", "hist_cod_orders", "hist_refusals", "account_age_days", "order_hour", "same_item_other_shops_48h",
        "is_campaign_day", "expected_days_to_delivery", "split"]]
    event = st.dataframe(
        view, hide_index=True, width="stretch", height=420, on_select="rerun", selection_mode="single-row",
        key="orders_table",
        column_config={
            "order_id": st.column_config.NumberColumn("Order", format="%d"),
            "order_datetime": st.column_config.DatetimeColumn("Placed", format="YYYY-MM-DD HH:mm"),
            "buyer_id": st.column_config.NumberColumn("Buyer", format="%d"),
            "area_id": st.column_config.NumberColumn("Area", format="%d"),
            "risk_pct": st.column_config.ProgressColumn("Risk", format="%.1f%%", min_value=0, max_value=40),
            "tier_name": "Tier", "outcome": "Actual outcome",
            "order_value": st.column_config.NumberColumn("Value (฿)", format="%d"),
            "value_vs_aov": st.column_config.NumberColumn("vs usual", format="%.1fx"),
            "hist_cod_orders": "Past orders", "hist_refusals": "Refusals", "account_age_days": "Account age (d)",
            "order_hour": "Hour", "same_item_other_shops_48h": "Same item elsewhere",
            "is_campaign_day": st.column_config.CheckboxColumn("Campaign"),
            "expected_days_to_delivery": "Delivery days", "split": "Split"})
    st.download_button("Download these orders (CSV)", view.to_csv(index=False).encode("utf-8"),
                       file_name="cod_orders_filtered.csv", mime="text/csv")

    rows = event.selection.rows if event and event.selection else []
    default_id = int(view.iloc[rows[0]]["order_id"]) if rows else int(view.iloc[0]["order_id"])
    st.divider()
    pick = st.number_input("Order ID to show", value=default_id, step=1, key="order_pick_" + str(default_id))
    row = all_orders[all_orders["order_id"] == pick]
    if row.empty:
        st.warning("Order not found.")
        return
    r = row.iloc[0]
    st.subheader(f"Order {int(r.order_id)} · buyer {int(r.buyer_id)} · {r.order_datetime:%Y-%m-%d %H:%M} · area {int(r.area_id)}")
    facts = {"Order value": f"฿{r.order_value:,.0f}", "Value vs usual": f"{r.value_vs_aov:.1f}x",
             "Account age": f"{int(r.account_age_days)} days", "Past COD orders": int(r.hist_cod_orders),
             "Past refusals": int(r.hist_refusals), "Same item at other shops (48h)": int(r.same_item_other_shops_48h),
             "Expected delivery": f"{int(r.expected_days_to_delivery)} days", "Split": r.split}
    st.markdown(" · ".join(f"**{k}:** {v}" for k, v in facts.items()))
    order_view(r[model.features], int(r.order_hour), model, params,
               extra={"outcome": r.outcome, "p_true": r.p_true})


# ---------------------------------------------------------------------------
# Page: Score an order
# ---------------------------------------------------------------------------
INPUT_DEFAULTS = {
    "in_value": 450, "in_date": dt.date(2026, 7, 15), "in_hour": 14, "in_same_item": 0, "in_days": 2,
    "in_account_age": 400, "in_condo": False, "in_area": 1, "in_area_rate": 0.0015,
    "in_past": 12, "in_refusals": 0, "in_last_refusal": 0, "in_not_home": 0, "in_courier": 0,
    "in_orders_90": 4, "in_refusals_90": 0, "in_usual": 450.0, "in_transit": 0, "in_last30": 1,
    "in_days_first": 300.0,
}


def load_order_into_form(all_orders, params):
    oid = st.session_state.get("in_load_id")
    row = all_orders[all_orders["order_id"] == oid]
    if row.empty:
        st.session_state["in_load_msg"] = f"Order {oid} not found."
        return
    r = row.iloc[0]
    st.session_state.update({
        "in_value": int(r.order_value), "in_date": r.order_datetime.date(), "in_hour": int(r.order_hour),
        "in_same_item": int(r.same_item_other_shops_48h), "in_days": int(r.expected_days_to_delivery),
        "in_account_age": int(r.account_age_days), "in_condo": bool(r.address_is_condo_with_office),
        "in_area": int(r.area_id), "in_area_rate": float(r.area_logistics_failure_rate),
        "in_past": int(r.hist_cod_orders), "in_refusals": int(r.hist_refusals),
        "in_last_refusal": max(int(r.days_since_last_refusal), 0), "in_not_home": int(r.hist_buyer_caused_misses),
        "in_courier": int(r.hist_courier_caused_failures), "in_orders_90": int(r.orders_last_90d),
        "in_refusals_90": int(r.refusals_last_90d), "in_usual": float(round(r.usual_order_value, 2)),
        "in_transit": int(r.orders_in_transit), "in_last30": int(r.orders_last_30d),
        "in_days_first": float(round(r.days_since_first_cod, 2)),
    })
    st.session_state["in_load_msg"] = (f"Loaded order {oid} (model score in the table: {r.risk:.2%}, "
                                       f"actual outcome: {r.outcome}).")


def reset_form():
    st.session_state.update(INPUT_DEFAULTS)
    st.session_state["in_load_msg"] = "Form reset to a typical buyer."


def set_area_rate(params):
    st.session_state["in_area_rate"] = params["area_latest_rate"][str(st.session_state["in_area"])]


def page_score(all_orders, params, model):
    st.title("COD Risk Score · Score an order")
    st.caption(NOTE + " Type what the platform knows at checkout. The app builds the 20 model features "
               "the same way as the training pipeline and scores the order live.")
    for k, v in INPUT_DEFAULTS.items():
        st.session_state.setdefault(k, v)

    c1, c2, c3 = st.columns([1, 1, 2])
    st.session_state.setdefault("in_load_id", int(all_orders["order_id"].iloc[0]))
    c1.number_input("Load an existing order ID", min_value=1, step=1, key="in_load_id")
    c2.button("Load into form", on_click=load_order_into_form, args=(all_orders, params), width="stretch")
    c2.button("Reset to a typical buyer", on_click=reset_form, width="stretch")
    if st.session_state.get("in_load_msg"):
        c3.info(st.session_state["in_load_msg"])

    st.markdown("#### 1. This order")
    a = st.columns(5)
    a[0].number_input("Order value (฿)", min_value=1, max_value=200_000, step=50, key="in_value")
    a[1].date_input("Order date", min_value=dt.date(2025, 1, 1), max_value=dt.date(2027, 12, 31), key="in_date")
    a[2].number_input("Order hour (0-23)", min_value=0, max_value=23, step=1, key="in_hour")
    a[3].number_input("Same item ordered with COD at other shops (48h)", min_value=0, max_value=10, step=1, key="in_same_item")
    a[4].number_input("Expected delivery days", min_value=1, max_value=6, step=1, key="in_days")

    st.markdown("#### 2. Buyer and address")
    b = st.columns(5)
    b[0].number_input("Account age (days)", min_value=0, max_value=10_000, step=1, key="in_account_age")
    b[1].selectbox("Area", list(range(1, 61)), key="in_area", on_change=set_area_rate, args=(params,))
    b[2].number_input("Area courier failure rate (last 90 days)", min_value=0.0, max_value=0.2, step=0.0005,
                      format="%.4f", key="in_area_rate", help="Filled in from the area's latest rate; you can override it.")
    b[3].checkbox("Condo with a front office", key="in_condo")

    st.markdown("#### 3. Buyer's COD history (orders whose outcome is known)")
    h = st.columns(5)
    h[0].number_input("Past COD orders", min_value=0, max_value=1000, step=1, key="in_past")
    h[1].number_input("Of which refused at the door", min_value=0, max_value=1000, step=1, key="in_refusals")
    h[2].number_input("Days since last refusal", min_value=0, max_value=5000, step=1, key="in_last_refusal",
                      disabled=st.session_state["in_refusals"] == 0)
    h[3].number_input("Of which 'not home / no cash'", min_value=0, max_value=1000, step=1, key="in_not_home")
    h[4].number_input("Of which courier failures", min_value=0, max_value=1000, step=1, key="in_courier")
    h2 = st.columns(5)
    h2[0].number_input("Past orders known in the last 90 days", min_value=0, max_value=1000, step=1, key="in_orders_90")
    h2[1].number_input("Refusals in the last 90 days", min_value=0, max_value=1000, step=1, key="in_refusals_90")
    h2[2].number_input("Buyer's usual order value (฿, 0 = none)", min_value=0.0, max_value=200_000.0, step=10.0,
                       key="in_usual")
    h2[3].number_input("COD orders placed in the last 30 days", min_value=0, max_value=500, step=1, key="in_last30")
    h2[4].number_input("Days since first COD order", min_value=0.0, max_value=5000.0, step=1.0, key="in_days_first")
    with st.expander("Advanced"):
        st.number_input("Earlier orders still in transit (outcome not known yet)", min_value=0, max_value=100, step=1,
                        key="in_transit", help="They count for 'value vs usual' and 'frequency', not for refusal history.")

    s = st.session_state
    errors = []
    if s.in_refusals + s.in_not_home + s.in_courier > s.in_past:
        errors.append("Refusals + 'not home' misses + courier failures cannot be more than past COD orders.")
    if s.in_orders_90 > s.in_past:
        errors.append("Orders in the last 90 days cannot be more than all past orders.")
    if s.in_refusals_90 > min(s.in_refusals, s.in_orders_90):
        errors.append("Refusals in the last 90 days cannot be more than all refusals or than orders in the last 90 days.")
    if s.in_refusals_90 > 0 and s.in_last_refusal > 90:
        errors.append("There is a refusal in the last 90 days, so 'days since last refusal' must be 90 or less.")
    if s.in_refusals > 0 and s.in_refusals_90 == 0 and s.in_last_refusal < 90:
        errors.append("The last refusal was under 90 days ago, so 'refusals in the last 90 days' must be at least 1.")
    if s.in_last30 > s.in_past + s.in_transit:
        errors.append("Orders in the last 30 days cannot be more than all earlier orders (past + in transit).")
    if s.in_past + s.in_transit == 0 and (s.in_usual > 0 or s.in_days_first > 0):
        errors.append("A buyer with no earlier orders has no usual value and no first order yet: set both to 0.")
    if s.in_past + s.in_transit > 0 and s.in_usual == 0:
        errors.append("A buyer with earlier orders needs a usual order value.")
    if s.in_days_first > s.in_account_age + 1:
        errors.append("The first COD order cannot be before the account was created.")
    if errors:
        for e in errors:
            st.error(e)
        return

    raw = pd.DataFrame([{
        "past_orders": s.in_past, "past_refusals": s.in_refusals, "orders_last_90d": s.in_orders_90,
        "refusals_last_90d": s.in_refusals_90, "days_since_last_refusal": s.in_last_refusal,
        "past_not_home": s.in_not_home, "past_courier_failures": s.in_courier, "order_value": s.in_value,
        "usual_order_value": s.in_usual, "orders_in_transit": s.in_transit, "orders_last_30d": s.in_last30,
        "days_since_first_cod": s.in_days_first, "account_age_days": s.in_account_age,
        "order_datetime": dt.datetime.combine(s.in_date, dt.time(s.in_hour)), "order_hour": s.in_hour,
        "same_item_other_shops": s.in_same_item, "delivery_days": s.in_days,
        "condo_with_office": int(s.in_condo), "area_logistics_failure_rate": s.in_area_rate}])
    feats = features_from_inputs(raw, params["feature_settings"]).iloc[0]
    notes = []
    if feats["is_campaign_day"]:
        notes.append(f"{s.in_date.day}.{s.in_date.month} is a campaign day")
    if feats["same_item_other_shops_48h"] > 3:
        notes.append("the training data had at most 3 other shops, so scores this far out are extrapolated")
    if s.in_date.year != 2026 or not 2 <= s.in_date.month <= 7:
        notes.append("the model only saw Feb-Jul 2026 orders; other months are extrapolated")
    st.divider()
    if notes:
        st.caption("Note: " + "; ".join(notes) + ".")
    order_view(feats, s.in_hour, model, params)


# ---------------------------------------------------------------------------
# Page: Policy simulator
# ---------------------------------------------------------------------------
SIM_KEYS = ["sim_t1", "sim_t2", "sim_c1", "sim_c2", "sim_deliv", "sim_f1", "sim_f2", "sim_msg"]


def sim_defaults(dc):
    return {"sim_t1": dc["tier1_cutoff"] * 100, "sim_t2": dc["tier2_cutoff"] * 100,
            "sim_c1": dc["catch_rate"]["tier1"] * 100, "sim_c2": dc["catch_rate"]["tier2"] * 100,
            "sim_deliv": dc["caught_become_delivered"] * 100, "sim_f1": dc["friction"]["tier1"] * 100,
            "sim_f2": dc["friction"]["tier2"] * 100, "sim_msg": dc["message_cost_thb"]}


def page_simulator(d_filtered, all_orders, params):
    st.title("COD Risk Score · Policy simulator")
    st.caption(NOTE + " Move the cut-offs and assumptions; every number is recomputed. "
               "Money uses expected values, the same model as reports/decision_report.md.")
    dc = params["decision"]
    for k, v in sim_defaults(dc).items():
        st.session_state.setdefault(k, v)

    scope = st.radio("Orders to simulate on", ["July test set (matches the report)", "Orders matching the sidebar filters"],
                     horizontal=True, key="sim_scope")
    d = all_orders[all_orders["split"] == "test"] if scope.startswith("July") else d_filtered
    if d.empty:
        st.warning("No orders to simulate.")
        return

    c = st.columns(4)
    c[0].slider("Tier 1 cut-off (risk %)", 0.5, 10.0, step=0.1, key="sim_t1")
    c[1].slider("Tier 2 cut-off (risk %)", 2.0, 40.0, step=0.5, key="sim_t2")
    c[2].slider("Tier 1 catch rate (% of refusals stopped)", 0.0, 100.0, step=5.0, key="sim_c1")
    c[3].slider("Tier 2 catch rate", 0.0, 100.0, step=5.0, key="sim_c2")
    c = st.columns(4)
    c[0].slider("Caught orders that end up delivered (%)", 0.0, 100.0, step=5.0, key="sim_deliv")
    c[1].slider("Tier 1 friction (% good orders lost)", 0.0, 3.0, step=0.1, key="sim_f1")
    c[2].slider("Tier 2 friction", 0.0, 5.0, step=0.1, key="sim_f2")
    c[3].slider("Message cost per asked order (฿)", 0.0, 1.0, step=0.05, key="sim_msg")
    if st.button("Back to the report's assumptions"):
        st.session_state.update(sim_defaults(dc))
        st.rerun()
    s = st.session_state
    if s.sim_t2 <= s.sim_t1:
        st.error("The Tier 2 cut-off must be above the Tier 1 cut-off.")
        return

    dcx = {**dc, "tier1_cutoff": s.sim_t1 / 100, "tier2_cutoff": s.sim_t2 / 100,
           "caught_become_delivered": s.sim_deliv / 100, "message_cost_thb": s.sim_msg}
    catch, fric = {1: s.sim_c1 / 100, 2: s.sim_c2 / 100}, {1: s.sim_f1 / 100, 2: s.sim_f2 / 100}
    tier = assign_tiers(d["risk"].to_numpy(), dcx)
    ft, y = d["failure_type"].to_numpy(), d["label_failed"].to_numpy()
    yearly = dc["yearly_orders_total"] * dc["cod_share"]
    n = len(d)
    pol = {"Do nothing": simulate(np.zeros(n, int), ft, dcx, catch, fric),
           "Ask everyone": simulate(np.ones(n, int), ft, dcx, catch, fric),
           "Our tiers": simulate(tier, ft, dcx, catch, fric)}

    k = st.columns(4)
    k[0].metric("Orders asked (our tiers)", pct(pol["Our tiers"]["asked_share"]))
    k[1].metric("Net per year: our tiers", f"฿{pol['Our tiers']['net'] / n * yearly / 1e6:,.1f}M")
    k[2].metric("Net per year: ask everyone", f"฿{pol['Ask everyone']['net'] / n * yearly / 1e6:,.1f}M")
    diff = (pol["Our tiers"]["net"] - pol["Ask everyone"]["net"]) / n * yearly
    k[3].metric("Our tiers vs ask everyone", f"{'+' if diff >= 0 else '-'}฿{abs(diff) / 1e6:,.1f}M / year")

    left, right = st.columns([1, 1.3])
    with left:
        st.markdown("##### Tier shape")
        st.dataframe(pd.DataFrame({
            "Tier": [TIER_NAMES[t] for t in (0, 1, 2)],
            "Share of orders": [pct((tier == t).mean()) for t in (0, 1, 2)],
            "Share of failures": [pct(y[tier == t].sum() / max(y.sum(), 1)) for t in (0, 1, 2)],
            "Actual rate": [pct(y[tier == t].mean(), 2) if (tier == t).any() else "-" for t in (0, 1, 2)]}),
            hide_index=True, width="stretch")
        st.markdown("##### Three policies (per 1,000 orders, ฿)")
        st.dataframe(pd.DataFrame([{
            "Policy": name, "Net per year (฿M)": round(r["net"] / n * yearly / 1e6, 1), "Net": round(r["net"] / n * 1000, 1),
            "Asked": pct(r["asked_share"], 0), "Failures avoided": round(r["failures_avoided"] / n * 1000, 2),
            "Shipping saved": round(r["shipping_saved"] / n * 1000, 1),
            "Commission won back": round(r["commission_won_back"] / n * 1000, 1),
            "Lost to friction": -round(r["commission_lost_friction"] / n * 1000, 1),
            "Messages": -round(r["message_cost"] / n * 1000, 1)} for name, r in pol.items()]),
            hide_index=True, width="stretch")
    with right:
        grid = np.round(np.arange(0.5, min(s.sim_t2, 10.0) + 0.001, 0.25), 2)
        nets = [simulate(assign_tiers(d["risk"].to_numpy(), {**dcx, "tier1_cutoff": g / 100}), ft, dcx, catch, fric)["net"]
                / n * yearly / 1e6 for g in grid]
        fig = go.Figure(go.Scatter(x=grid, y=nets, mode="lines", line=dict(width=3), name="Our tiers"))
        fig.add_hline(y=pol["Ask everyone"]["net"] / n * yearly / 1e6, line_dash="dash", line_color="#999",
                      annotation_text="Ask everyone")
        fig.add_vline(x=s.sim_t1, line_dash="dot", annotation_text=f"current {s.sim_t1:.1f}%")
        best = grid[int(np.argmax(nets))]
        fig.update_layout(title=f"Net value per year vs Tier 1 cut-off (best here: {best:.2f}%)",
                          xaxis_title="Tier 1 cut-off (risk %)", yaxis_title="฿ million per year")
        show(fig)

    st.markdown("##### Sensitivity: net per year (฿M) for our tiers, by Tier 1 catch rate and friction "
                "(Tier 2 keeps its gap: catch +{:.0f} pts, friction x{:.1f})".format(
                    s.sim_c2 - s.sim_c1, (s.sim_f2 / s.sim_f1) if s.sim_f1 else 1))
    rows = []
    for cr in (25, 40, 50):
        row = {"Tier 1 catch rate": f"{cr}%"}
        for fr in (0.3, 0.5, 1.0):
            ratio = (s.sim_f2 / s.sim_f1) if s.sim_f1 else 1
            r = simulate(tier, ft, dcx, {1: cr / 100, 2: min(cr + s.sim_c2 - s.sim_c1, 100) / 100},
                         {1: fr / 100, 2: fr * ratio / 100})
            e = simulate(np.ones(n, int), ft, dcx, {1: cr / 100, 2: cr / 100}, {1: fr / 100, 2: fr / 100})
            row[f"friction {fr}%"] = f"{r['net'] / n * yearly / 1e6:,.1f} (vs {e['net'] / n * yearly / 1e6:,.1f})"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Each cell: our tiers (vs ask everyone). Yearly scale: "
               f"{dc['yearly_orders_total'] / 1e6:,.1f}M orders x {dc['cod_share']:.0%} COD = {yearly / 1e6:,.1f}M COD orders.")


# ---------------------------------------------------------------------------
# Entry point (called from app.py)
# ---------------------------------------------------------------------------
def render():
    if not (APP_DATA / "orders.csv").exists():
        st.error("App data not found. Run `python -m src.export_app` inside cod_risk_demo/ "
                 "(or run_all.bat) first.")
        return
    # Keep filter / form values when switching pages (Streamlit drops state of hidden widgets).
    for k in list(st.session_state.keys()):
        if k.startswith(("f_", "in_", "sim_")) and not k.startswith("in_load_msg"):
            st.session_state[k] = st.session_state[k]

    params, df = load()
    model = get_model(json.dumps(params))

    st.sidebar.title("COD Risk Score")
    st.sidebar.caption("ML model (calibrated Logistic Regression) · " + NOTE)
    # Direct links: ?system=ml&page=dashboard|orders|score|simulator
    wanted = {"dashboard": 0, "orders": 1, "score": 2, "simulator": 3}.get(st.query_params.get("page", ""), 0)
    page = st.sidebar.radio("Page", PAGES, index=wanted, key="risk_page")

    if page == "Score an order":
        page_score(df, params, model)
        return
    d = filter_panel(df)
    if page == "Dashboard":
        page_dashboard(d, params)
    elif page == "Orders":
        page_orders(d, df, params, model)
    else:
        page_simulator(d, df, params)
