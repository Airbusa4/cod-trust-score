"""COD Risk Score (ML model) pages for the Streamlit app.

SYNTHETIC DATA - for illustration only. This is NOT real Shopee data.

Pages:
  Dashboard         - KPIs, risk distribution, tiers, risk trend (daily / weekly / monthly)
  Orders            - every order (filterable, sortable) + full detail of one order
  Score an order    - type raw order / buyer data, get the risk score, tier and reasons
  Import data       - upload a CSV / Excel file of orders, check it in a pop-up, score it;
                      every page can then show the imported orders instead of the demo data

Every number shown comes from the model's predicted risk; actual outcomes are not shown.
Data comes from cod_risk_demo/app_data/ (built by `python -m src.export_app`).
"""
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

if __name__ == "__main__":  # started as `streamlit run risk_ui/app.py`: make the repo root importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from risk_ui import importer  # noqa: E402
from risk_ui.model import (DEPOSIT_SHARE, NICE_NAMES, TIER_ACTIONS, TIER_COLORS, TIER_CUTOFFS,
                           TIER_NAMES, TIER_RANGES, TIER_TEXT_ON, TIERS, COST_FORMULA, Model, assign_tiers, cost_of_failure,
                           features_from_inputs, top_reasons)

APP_DATA = Path(__file__).resolve().parent.parent / "cod_risk_demo" / "app_data"
DEMO_NOTE = "Synthetic data, for illustration only. Not Shopee data."


def note():
    """The data note shown on every page and chart: says which data is on screen."""
    imp = st.session_state.get("imported")
    if st.session_state.get("active_source") == IMPORTED and imp:
        return f"Imported data ({imp['name']}), scored by a model trained on synthetic data."
    return DEMO_NOTE
PAGES = ["Dashboard", "Orders", "Score an order", "Import data"]
DEMO, IMPORTED = "Demo data", "Imported file"
TIER_COLOR_BY_NAME = {TIER_NAMES[k]: v for k, v in TIER_COLORS.items()}
# Theme colours (black / orange / white; the rest of the theme is in .streamlit/config.toml).
ACCENT = "#f47b20"        # the one series a chart is about
CONTEXT = "#9a9aa0"       # the series it is compared with
LOWERS = "#c9c9cc"        # factor pushes risk down (vs ACCENT = pushes it up)


def tier_badge(tier, size=20):
    return (f"<span style='display:inline-block;font-weight:750;font-size:{size}px;padding:4px 14px;border-radius:999px;"
            f"background:{TIER_COLORS[tier]};color:{TIER_TEXT_ON[tier]}'>{TIER_NAMES[tier]}</span>")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading orders ...")
def load():
    params = json.loads((APP_DATA / "model_params.json").read_text(encoding="utf-8"))
    df = pd.read_csv(APP_DATA / "orders.csv", comment="#")
    df["order_datetime"] = pd.to_datetime(df["order_datetime"])
    df["tier"] = assign_tiers(df["risk"].to_numpy())
    df["tier_name"] = df["tier"].map(TIER_NAMES)
    df["risk_pct"] = df["risk"] * 100
    return params, df


@st.cache_resource
def get_model(params_json):
    return Model(json.loads(params_json))


def show(fig, height=380, value_axis="y", bottom=50, subtitle=None):
    """Plot with the synthetic-data note in the corner and recessive chrome
    (hairline solid grid on the value axis only, rounded bar ends)."""
    title = fig.layout.title.text or ""
    fig.update_layout(title_text=f"{title}<br><sup style='color:#8a8a90'>{subtitle or note()}</sup>", height=height,
                      margin=dict(t=72, b=bottom, l=10, r=10), barcornerradius=4,
                      font=dict(size=14), title_font=dict(size=17), legend_font=dict(size=13))
    value, category = (fig.update_yaxes, fig.update_xaxes) if value_axis == "y" else (fig.update_xaxes, fig.update_yaxes)
    value(gridwidth=1, griddash="solid", zeroline=False)
    category(showgrid=False, zeroline=False)
    st.plotly_chart(fig, width="stretch")


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def baht(x):
    return f"฿{x / 1e6:,.2f}M" if abs(x) >= 1e6 else f"฿{x:,.0f}"


# ---------------------------------------------------------------------------
# Filters. Two independent sets, each with its own key prefix so it can be reset:
#   "fd_" = the Dashboard's row of filters, "f_" = the Orders page panel.
# A filter spec is (column, kind, label); kinds: date, multi, bool, range, ids, id.
# ---------------------------------------------------------------------------
DASHBOARD_FILTERS = [("order_datetime", "date", "Order date"), ("tier", "multi", "Tier"),
                     ("area_id", "multi", "Area"), ("split", "multi", "Split")]
ORDERS_MAIN_FILTERS = [("order_id", "ids", "Order IDs"), ("buyer_id", "id", "Buyer ID"),
                       ("order_datetime", "date", "Order date"), ("tier", "multi", "Tier")]
ORDERS_MORE_FILTERS = {
    "Order": [
        ("order_value", "range", "Order value (THB)"),
        ("value_vs_aov", "range", "Value vs buyer's usual (x)"),
        ("order_hour", "range", "Order hour"),
        ("order_month", "multi", "Order month"),
        ("is_late_night", "bool", "Late night (01:00-04:59)"),
        ("is_campaign_day", "bool", "Campaign day"),
        ("same_item_other_shops_48h", "range", "Same item at other shops (48h)"),
        ("expected_days_to_delivery", "range", "Expected delivery days"),
        ("freq_change_ratio", "range", "Order frequency change (30d)"),
        ("split", "multi", "Split (train = Feb-Jun, test = Jul)"),
    ],
    "Buyer": [
        ("is_new_cod_buyer", "bool", "First COD order (no history)"),
        ("has_ever_refused", "bool", "Has ever refused"),
        ("address_is_condo_with_office", "bool", "Condo with front office"),
        ("hist_cod_orders", "range", "Past COD orders"),
        ("hist_refusals", "range", "Past refusals"),
        ("account_age_days", "range", "Account age (days)"),
        ("refusal_rate_smoothed", "range", "Refusal rate (smoothed)"),
        ("recent_refusal_rate_smoothed_90d", "range", "Refusal rate, last 90 days"),
        ("days_since_last_refusal", "range", "Days since last refusal (-1 = never)"),
        ("hist_buyer_caused_misses", "range", "Past 'not home / no cash' misses"),
        ("hist_courier_caused_failures", "range", "Past courier failures"),
    ],
    "Area & score": [
        ("area_id", "multi", "Area"),
        ("area_logistics_failure_rate", "range", "Area courier failure rate (90d)"),
        ("risk_pct", "range", "Risk score (%)"),
    ],
}
ORDERS_ALL_FILTERS = ORDERS_MAIN_FILTERS + [x for specs in ORDERS_MORE_FILTERS.values() for x in specs]


def reset_filters(prefix):
    for k in [k for k in st.session_state if k.startswith(prefix)]:
        del st.session_state[k]


def bounds(s):
    is_int = pd.api.types.is_integer_dtype(s)
    return ((int(s.min()), int(s.max())) if is_int else (float(s.min()), float(s.max()))), is_int


def filter_widget(df, spec, prefix):
    """Draw one filter widget (its value lives in st.session_state[prefix + column])."""
    col, kind, label = spec
    key, s = prefix + col, df[col]
    if kind == "ids":
        st.text_input(label, key=key, placeholder="e.g. 3, 17, 2041")
    elif kind == "id":
        st.text_input(label, key=key, placeholder="e.g. 1250")
    elif kind == "date":
        lo, hi = s.min().date(), s.max().date()
        init = {} if key in st.session_state else {"value": (lo, hi)}
        st.date_input(label, min_value=lo, max_value=hi, key=key, **init)
    elif kind == "multi":
        fmt = (lambda t: TIER_NAMES[t]) if col == "tier" else str
        st.multiselect(label, sorted(s.dropna().unique().tolist()), key=key, format_func=fmt, placeholder="All")
    elif kind == "bool":
        st.selectbox(label, ["Any", "Yes", "No"], key=key)
    else:  # range
        (lo, hi), is_int = bounds(s)
        if lo == hi:
            return
        step = 1 if is_int else float(f"{(hi - lo) / 200:.2g}")
        init = {} if key in st.session_state else {"value": (lo, hi)}
        st.slider(label, lo, hi, step=step, key=key, **init)


def filter_state(df, spec, prefix):
    """(mask, short description) for one filter from its saved value, or None if it filters nothing."""
    col, kind, label = spec
    val, s = st.session_state.get(prefix + col), df[col]
    if val is None:
        return None
    if kind == "ids":
        wanted = [int(x) for x in str(val).replace(" ", "").split(",") if x.isdigit()]
        return (s.isin(wanted), f"{label}: {', '.join(map(str, wanted))}") if wanted else None
    if kind == "id":
        return (s == int(val), f"{label}: {int(val)}") if str(val).strip().isdigit() else None
    if kind == "date":
        lo, hi = s.min().date(), s.max().date()
        if isinstance(val, (tuple, list)) and len(val) == 2 and (val[0] > lo or val[1] < hi):
            return s.dt.date.between(val[0], val[1]), f"{label}: {val[0]:%d %b} – {val[1]:%d %b %Y}"
        return None
    if kind == "multi":
        if not val:
            return None
        shown = [TIER_NAMES[v].split(" · ")[0] if col == "tier" else str(v) for v in val]
        return s.isin(val), f"{label}: {', '.join(shown)}"
    if kind == "bool":
        return (s == (1 if val == "Yes" else 0), f"{label}: {val}") if val != "Any" else None
    (lo, hi), _ = bounds(s)
    if lo < hi and (val[0] > lo or val[1] < hi):
        return s.between(val[0] - 1e-9, val[1] + 1e-9), f"{label}: {fmt_num(val[0])} – {fmt_num(val[1])}"
    return None


def fmt_num(v):
    """Readable number for filter chips: 26,890 / 12 / 0.0015 / 2.5."""
    if abs(v) >= 1000 or float(v).is_integer():
        return f"{v:,.0f}"
    return f"{v:.3g}"


def apply_filters(df, specs, prefix):
    """Filtered orders + descriptions of the active filters, from the saved filter values."""
    active = [x for x in (filter_state(df, spec, prefix) for spec in specs) if x is not None]
    out = df[np.logical_and.reduce([m for m, _ in active])] if active else df
    return out, [text for _, text in active]


def filter_chips(texts, n_shown, n_all):
    chips = "".join(f"<span style='display:inline-block;padding:2px 10px;margin:2px 4px 2px 0;border-radius:999px;"
                    f"background:rgba(128,128,128,.15);font-size:13px'>{t}</span>" for t in texts)
    st.markdown(f"<div style='font-size:14px'><b>{n_shown:,}</b> of {n_all:,} orders"
                f"{' · ' + str(len(texts)) + ' filter(s):' if texts else ' · no filters'}</div>"
                f"<div>{chips}</div>", unsafe_allow_html=True)


def dashboard_filters(df):
    """The Dashboard's single row of filters; returns the filtered orders."""
    with st.container(border=True):
        cols = st.columns([1.4, 1.4, 1, 1, 0.8], vertical_alignment="bottom")
        for c, spec in zip(cols, DASHBOARD_FILTERS):
            with c:
                filter_widget(df, spec, "fd_")
        cols[-1].button("Reset filters", on_click=reset_filters, args=("fd_",), width="stretch", key="reset_dashboard_filters")
        out, texts = apply_filters(df, DASHBOARD_FILTERS, "fd_")
        if texts:
            filter_chips(texts, len(out), len(df))
    return out


def orders_filter_panel(df):
    """The Orders page filter panel: main filters in one row, the rest in tabs. Returns the filtered orders."""
    with st.container(border=True):
        cols = st.columns([1.2, 0.8, 1.4, 1.4, 0.8], vertical_alignment="bottom")
        for c, spec in zip(cols, ORDERS_MAIN_FILTERS):
            with c:
                filter_widget(df, spec, "f_")
        cols[-1].button("Reset filters", on_click=reset_filters, args=("f_",), width="stretch", key="reset_orders_filters")
        with st.expander("More filters: order, buyer, area and score"):
            for tab, specs in zip(st.tabs(list(ORDERS_MORE_FILTERS)), ORDERS_MORE_FILTERS.values()):
                with tab:
                    grid = st.columns(3)
                    for i, spec in enumerate(specs):
                        with grid[i % 3]:
                            filter_widget(df, spec, "f_")
        out, texts = apply_filters(df, ORDERS_ALL_FILTERS, "f_")
        filter_chips(texts, len(out), len(df))
    return out


# ---------------------------------------------------------------------------
# One order: score, reasons, factor chart, phone
# ---------------------------------------------------------------------------
PHONE_CSS = """
<style>
.phone{width:300px;height:560px;border-radius:40px;background:#000;border:2px solid #3a3a40;padding:12px;margin:0 auto}
.screen{background:#fff;border-radius:30px;height:100%;overflow:hidden;display:flex;flex-direction:column;color:#1d2330}
.appbar{background:#ee4d2d;color:#fff;font-weight:700;padding:28px 16px 12px;font-size:17px}
.pbody{padding:16px;display:flex;flex-direction:column;gap:10px}
.ptitle{font-size:21px;font-weight:750;line-height:1.2}.psub{font-size:14px;color:#5d6677}
.pbtn{font-size:15px;font-weight:650;padding:11px;border-radius:10px;text-align:center;border:2px solid #e3e6ec}
.main{background:#ee4d2d;color:#fff;border-color:#ee4d2d}
.pnote{font-size:13px;padding:10px;border-radius:10px;background:#fff4e0;color:#7a4a00}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.check{width:64px;height:64px;border-radius:50%;background:#e6f4ea;color:#1e8e3e;font-size:36px;display:flex;
       align-items:center;justify-content:center;margin:24px auto 4px}
.badge{display:inline-block;font-weight:750;font-size:20px;padding:6px 18px;border-radius:999px}
</style>"""


def phone_html(tier, value):
    v = f"฿{value:,.0f}"
    if tier == 1:
        body = (f'<div class="check">✓</div><div class="ptitle" style="text-align:center">Order placed</div>'
                f'<div class="psub" style="text-align:center">{v} · Cash on delivery.<br>Pay the courier when your parcel arrives.</div>'
                '<div class="pnote">🔔 We will remind you before delivery so you can have the cash ready.</div>')
    elif tier == 2:
        body = (f'<div class="ptitle">Confirm your {v} COD order</div>'
                '<div class="psub">Your order is on hold. Please confirm you will be home to receive and pay for it.</div>'
                '<div class="pbtn main">Confirm</div><div class="row2"><div class="pbtn">Cancel</div><div class="pbtn">Pay now</div></div>'
                '<div class="pnote">We ship as soon as you confirm.</div>')
    elif tier == 3:
        dep = f"฿{value * DEPOSIT_SHARE:,.0f}"
        body = (f'<div class="ptitle">Pay a {dep} deposit to keep COD</div>'
                f'<div class="psub">A refundable deposit (10% of {v}) is needed for this cash-on-delivery order.</div>'
                f'<div class="pbtn main">Pay {dep} deposit</div><div class="row2"><div class="pbtn">Cancel</div><div class="pbtn">Pay in full</div></div>'
                '<div class="pnote">The deposit is refunded when you receive and pay for the parcel.</div>')
    else:
        body = ('<div class="ptitle">COD is not available for this order</div>'
                f'<div class="psub">Please pay {v} now to place this order.</div>'
                f'<div class="pbtn main">Pay now {v}</div><div class="pbtn">Cancel</div>')
    return f'{PHONE_CSS}<div class="phone"><div class="screen"><div class="appbar">Checkout</div><div class="pbody">{body}</div></div></div>'


def score_order(feats, order_hour, model):
    """Risk, tier, top-3 reasons and every factor's push for ONE order (`feats` = Series of the 20 features)."""
    X = feats.to_frame().T.astype(float)
    p = float(model.predict(X)[0])
    reasons, phi = top_reasons(model, X, order_hour)
    return p, int(assign_tiers([p])[0]), reasons, phi


def score_block(p, tier, reasons, params):
    avg = params["population_failure_rate"]
    st.markdown("##### Risk of failed delivery")
    st.markdown(f"<div style='font-size:56px;font-weight:800;line-height:1'>{p:.1%}</div>"
                f"<div style='opacity:.7'>vs about {avg:.1%} on average ({p / avg:.1f}x)</div>",
                unsafe_allow_html=True)
    st.markdown(f"<div style='margin:12px 0'>{tier_badge(tier)}</div>", unsafe_allow_html=True)
    st.markdown(f"**Action:** {TIER_ACTIONS[tier]}")
    st.markdown("##### Top 3 reasons")
    for r in reasons:
        arrow, color, word = ("▲", ACCENT, "raises") if r["direction"] == "up" else ("▼", LOWERS, "lowers")
        st.markdown(f"<div style='font-size:17px;margin:6px 0'><span style='color:{color};font-weight:800'>{arrow}</span> "
                    f"<b>{r['text']}</b><br><span style='opacity:.65;font-size:13px'>{word} risk vs an average order · "
                    f"{r['name']} ({r['shap']:+.2f} log-odds)</span></div>", unsafe_allow_html=True)


def factor_chart(phi, model, height=520):
    contrib = pd.DataFrame({"factor": [NICE_NAMES[f] for f in model.features], "push": phi})
    contrib = contrib.reindex(contrib["push"].abs().sort_values().index)
    fig = go.Figure(go.Bar(x=contrib["push"], y=contrib["factor"], orientation="h",
                           marker_color=np.where(contrib["push"] > 0, ACCENT, LOWERS)))
    fig.update_layout(title="What pushes this order's risk (all 20 factors)",
                      xaxis_title="push on risk (log-odds, vs an average order)", yaxis_title=None)
    show(fig, height=height, value_axis="x")


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------
TREND_FREQ = {"Daily": ("D", "day"), "Weekly": ("W", "week"), "Monthly": ("MS", "month")}
TREND_MEASURES = ["Mean predicted risk", "Predicted cost lost", "Orders"]


def share_label(v):
    """A percentage label that never shows a non-empty share as 0.0%."""
    return "<0.1%" if 0 < v < 0.05 else f"{v:.1f}%"


def tier_card(k, g, total_orders):
    """One tier's summary card: colour key, range, action, orders, share bar, risk and cost."""
    n = int(g.loc[k, "orders"])
    share = n / max(total_orders, 1)
    risk = "-" if n == 0 else pct(g.loc[k, "predicted"], 2)
    with st.container(border=True):
        # Inner padding + clear gaps between the card's three parts (who / how many / what it costs)
        st.markdown(
            f"<div style='padding:6px 6px 4px'>"
            f"<div style='height:6px;border-radius:3px;background:{TIER_COLORS[k]};margin-bottom:16px'></div>"
            f"<div style='font-weight:700;font-size:17px;margin-bottom:4px'>{TIER_NAMES[k]}</div>"
            f"<div style='opacity:.7;font-size:13px;line-height:1.55;min-height:42px'>"
            f"Risk {TIER_RANGES[k]}<br>{TIER_ACTIONS[k]}</div>"
            f"<div style='font-size:32px;font-weight:750;line-height:1.1;margin-top:18px'>{n:,}</div>"
            f"<div style='opacity:.7;font-size:13px;margin-top:4px'>"
            f"orders · {'<0.1%' if 0 < share < 0.001 else pct(share)} of all</div>"
            f"<div style='height:6px;border-radius:3px;background:rgba(128,128,128,.2);margin:14px 0 18px'>"
            f"<div style='height:6px;border-radius:3px;width:{max(share * 100, 0.5 if n else 0):.2f}%;"
            f"background:{TIER_COLORS[k]}'></div></div>"
            f"<div style='display:flex;justify-content:space-between;font-size:13px;line-height:1.9'>"
            f"<span style='opacity:.7'>Mean predicted risk</span><b>{risk}</b></div>"
            f"<div style='display:flex;justify-content:space-between;font-size:13px;line-height:1.9'>"
            f"<span style='opacity:.7'>Predicted cost lost</span><b>{baht(g.loc[k, 'cost'])}</b></div>"
            f"</div>",
            unsafe_allow_html=True)


def page_dashboard(df, params):
    st.title("COD Risk Score · Dashboard")
    st.caption(note() + " Every number is the model's prediction and follows the filters below.")
    d = dashboard_filters(df)
    if d.empty:
        st.warning("No orders match the filters.")
        return
    dc = params["decision"]
    d = d.assign(cost=d["risk"] * cost_of_failure(d["order_value"]))

    # ---- headline numbers ----
    card = dict(border=True, height="stretch")
    c = st.columns(3)
    c[0].metric("Orders", f"{len(d):,}", **card, help="Number of COD orders matching the filters.")
    c[1].metric("Mean predicted risk", pct(d["risk"].mean(), 2), **card,
                help="Average of the model's predicted chance that an order fails.")
    c[2].metric("Predicted cost lost", baht(d["cost"].sum()), **card,
                help=f"Sum over orders of {COST_FORMULA}; about ฿{d['cost'].mean():,.2f} per order.")

    # ---- one card per tier ----
    st.markdown("#### Tiers")
    g = d.groupby("tier").agg(orders=("order_id", "size"), predicted=("risk", "mean"), cost=("cost", "sum"))
    g = g.reindex(TIERS).fillna({"orders": 0, "cost": 0})
    for col, k in zip(st.columns(4, gap="medium"), TIERS):
        with col:
            tier_card(k, g, len(d))

    # ---- where the orders vs the cost sit; how risk is spread ----
    left, right = st.columns(2)
    with left:
        names = [TIER_NAMES[k] for k in TIERS]
        so = g["orders"] / max(g["orders"].sum(), 1) * 100
        sc = g["cost"] / max(g["cost"].sum(), 1e-9) * 100
        fig = go.Figure([
            go.Bar(y=names, x=so, name="Share of orders", orientation="h", marker_color=CONTEXT,
                   text=[share_label(v) for v in so], textposition="outside",
                   hovertemplate="%{y}<br>Share of orders: %{x:.1f}%<extra></extra>"),
            go.Bar(y=names, x=sc, name="Share of predicted cost lost", orientation="h", marker_color=ACCENT,
                   text=[share_label(v) for v in sc], textposition="outside",
                   hovertemplate="%{y}<br>Share of predicted cost lost: %{x:.1f}%<extra></extra>")])
        fig.update_layout(title="Share of orders vs share of predicted cost lost", barmode="group", bargroupgap=0.08,
                          xaxis=dict(title="% of total", range=[0, 110], ticksuffix="%"),
                          yaxis=dict(autorange="reversed", title=None),
                          legend=dict(orientation="h", yanchor="top", y=-0.22, x=0))
        show(fig, value_axis="x", bottom=110)
    with right:
        h = d.assign(risk_shown=d["risk_pct"].clip(upper=30))
        fig = px.histogram(h, x="risk_shown", color="tier_name", nbins=120, color_discrete_map=TIER_COLOR_BY_NAME,
                           category_orders={"tier_name": list(TIER_NAMES.values())},
                           labels={"risk_shown": "Predicted risk (%), 30+ grouped at 30", "tier_name": "Tier"})
        for cut in (TIER_CUTOFFS[2], TIER_CUTOFFS[3]):
            fig.add_vline(x=cut * 100, line_width=1, line_color="rgba(128,128,128,.8)",
                          annotation_text=f"{cut:.0%}", annotation_position="top")
        fig.update_traces(hovertemplate="Risk %{x}%<br>%{y:,} orders<extra></extra>")
        fig.update_layout(title="How predicted risk is spread", bargap=0.08, yaxis_title="Orders",
                          legend=dict(orientation="h", yanchor="top", y=-0.22, x=0, title=None))
        show(fig, bottom=110)
        n4 = int(g.loc[4, "orders"])
        st.caption(f"The Tier 4 cut-off ({TIER_CUTOFFS[4]:.0%}) is off this chart: {n4:,} order(s) are at or above it.")

    # ---- trend ----
    st.markdown("#### Trend")
    c = st.columns([1, 1.6, 2])
    freq = c[0].segmented_control("Group by", list(TREND_FREQ), default="Weekly", key="trend_freq") or "Weekly"
    measure = c[1].segmented_control("Show", TREND_MEASURES, default=TREND_MEASURES[0], key="trend_measure") \
        or TREND_MEASURES[0]
    rule, unit = TREND_FREQ[freq]
    t = d.set_index("order_datetime")
    if measure == "Mean predicted risk":
        w = t.resample(rule).agg({"risk": "mean", "order_id": "size"})
        w = w[w["order_id"] > 0]
        fig = go.Figure(go.Scatter(
            x=w.index, y=w["risk"] * 100, mode="lines" if freq == "Daily" else "lines+markers",
            line=dict(color=ACCENT, width=2), marker=dict(size=8), customdata=w["order_id"],
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<br>%{customdata:,} orders<extra></extra>"))
        fig.update_layout(title=f"Mean predicted risk per {unit}", yaxis_title="Predicted risk (%)", hovermode="x")
        table = w.rename(columns={"risk": "Mean predicted risk", "order_id": "Orders"})
    else:
        col, agg = ("cost", "sum") if measure == "Predicted cost lost" else ("order_id", "size")
        w = t.groupby([pd.Grouper(freq=rule), "tier"])[col].agg(agg).unstack("tier").reindex(columns=TIERS).fillna(0)
        w = w[w.sum(axis=1) > 0]
        fig = go.Figure([go.Bar(x=w.index, y=w[k], name=TIER_NAMES[k], marker_color=TIER_COLORS[k],
                                hovertemplate=f"%{{x|%Y-%m-%d}}<br>{TIER_NAMES[k]}: "
                                              + ("฿%{y:,.0f}" if col == "cost" else "%{y:,}") + "<extra></extra>")
                         for k in TIERS])
        fig.update_layout(title=f"{measure} per {unit}, by tier", barmode="stack", bargap=0.15,
                          yaxis_title="฿" if col == "cost" else "Orders",
                          legend=dict(orientation="h", yanchor="top", y=-0.15, x=0))
        table = w.rename(columns=TIER_NAMES)
    show(fig, height=400, bottom=90)
    with st.expander("Trend data as a table"):
        st.dataframe(table.reset_index().rename(columns={"order_datetime": unit.title()}), hide_index=True,
                     width="stretch")


# ---------------------------------------------------------------------------
# Page: Orders
# ---------------------------------------------------------------------------
def page_orders(all_orders, params, model):
    st.title("COD Risk Score · Orders")
    st.caption(note() + " Filter the orders, then click a row to see that order in full. Click a column header to sort.")
    d = orders_filter_panel(all_orders)
    if d.empty:
        st.warning("No orders match the filters.")
        return
    view = d.sort_values("risk", ascending=False)[[
        "order_id", "order_datetime", "buyer_id", "area_id", "risk_pct", "tier_name", "order_value",
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
            "tier_name": "Tier",
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
    p, tier, reasons, phi = score_order(r[model.features], int(r.order_hour), model)
    left, mid, right = st.columns([1.1, 1.3, 0.9])
    with left:
        score_block(p, tier, reasons, params)
    with mid:
        factor_chart(phi, model)
    with right:
        st.markdown("##### What the buyer sees")
        st.markdown(phone_html(tier, float(r.order_value)), unsafe_allow_html=True)


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
PRESETS = {
    "Typical buyer": {},
    "New buyer": {"in_value": 1200, "in_account_age": 20, "in_past": 0, "in_orders_90": 0, "in_usual": 0.0,
                  "in_last30": 0, "in_days_first": 0.0},
    "Past refuser": {"in_value": 1800, "in_past": 10, "in_refusals": 4, "in_last_refusal": 10, "in_orders_90": 4,
                     "in_refusals_90": 2},
}


def apply_preset(name):
    st.session_state.update({**INPUT_DEFAULTS, **PRESETS[name]})
    st.session_state["in_load_msg"] = f"Form set to: {name}."


def load_order_into_form(all_orders):
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
    st.session_state["in_load_msg"] = f"Loaded order {oid} (score in the Orders table: {r.risk:.2%})."


def set_area_rate(params):
    st.session_state["in_area_rate"] = params["area_latest_rate"][str(st.session_state["in_area"])]


def field(widget, label, desc, **kw):
    """An input with a short description underneath."""
    widget(label, **kw)
    st.caption(desc)


def form_errors(s):
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
    return errors


def page_score(all_orders, params, model):
    st.title("COD Risk Score · Score an order")
    st.caption(note() + " Fill in what the platform knows at checkout. The score on the right updates as you type.")
    for k, v in INPUT_DEFAULTS.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("in_load_id", int(all_orders["order_id"].iloc[0]))

    with st.container(border=True):
        st.markdown("**Start from an example**, then change any field below.")
        c = st.columns([1, 1, 1, 0.3, 1.2, 1])
        for col, name in zip(c[:3], PRESETS):
            col.button(name, on_click=apply_preset, args=(name,), width="stretch")
        c[4].number_input("Order ID", min_value=1, step=1, key="in_load_id", label_visibility="collapsed")
        c[5].button("Load this order", on_click=load_order_into_form, args=(all_orders,), width="stretch")
        if st.session_state.get("in_load_msg"):
            st.caption(st.session_state["in_load_msg"])

    form, result = st.columns([1.5, 1], gap="large")
    with form:
        t1, t2, t3 = st.tabs(["1 · This order", "2 · Buyer & address", "3 · COD history"])
        with t1:
            a, b = st.columns(2)
            with a:
                field(st.number_input, "Order value (฿)", "Total price of this order, paid in cash at the door.",
                      min_value=1, max_value=200_000, step=50, key="in_value")
                field(st.number_input, "Order hour (0-23)", "Hour of checkout. 01:00-04:59 counts as late night.",
                      min_value=0, max_value=23, step=1, key="in_hour")
                field(st.number_input, "Expected delivery days", "Delivery time promised at checkout (1-6 days).",
                      min_value=1, max_value=6, step=1, key="in_days")
            with b:
                field(st.date_input, "Order date",
                      "Day of the order. Days like 7.7 or 11.11 count as campaign days. The model saw Feb-Jul 2026.",
                      min_value=dt.date(2025, 1, 1), max_value=dt.date(2027, 12, 31), key="in_date")
                field(st.number_input, "Same item at other shops (48h)",
                      "How many OTHER shops this buyer ordered the same item from with COD in the last 48 hours.",
                      min_value=0, max_value=10, step=1, key="in_same_item")
        with t2:
            a, b = st.columns(2)
            with a:
                field(st.number_input, "Account age (days)", "Days since the buyer's account was created.",
                      min_value=0, max_value=10_000, step=1, key="in_account_age")
                field(st.checkbox, "Condo with a front office",
                      "Tick if building staff can receive the parcel when the buyer is out.", key="in_condo")
            with b:
                field(st.selectbox, "Area", "Delivery area 1-60 (synthetic). Picking one fills in its courier failure rate.",
                      options=list(range(1, 61)), key="in_area", on_change=set_area_rate, args=(params,))
                field(st.number_input, "Area courier failure rate (last 90 days)",
                      "Share of this area's orders that failed because of the courier. 0.0015 = 0.15%.",
                      min_value=0.0, max_value=0.2, step=0.0005, format="%.4f", key="in_area_rate")
        with t3:
            st.caption("Count only earlier COD orders whose result is already known (delivered or failed).")
            a, b = st.columns(2)
            with a:
                field(st.number_input, "Past COD orders", "All earlier COD orders with a known result. 0 = first COD order.",
                      min_value=0, max_value=1000, step=1, key="in_past")
                field(st.number_input, "Refused at the door", "Of those, how many the buyer refused to accept.",
                      min_value=0, max_value=1000, step=1, key="in_refusals")
                field(st.number_input, "Not home / no cash", "Of those, how many failed because the buyer was out or had no cash.",
                      min_value=0, max_value=1000, step=1, key="in_not_home")
                field(st.number_input, "Courier failures", "Of those, how many failed because of the courier. Not held against the buyer.",
                      min_value=0, max_value=1000, step=1, key="in_courier")
                field(st.number_input, "Buyer's usual order value (฿)", "The buyer's typical order size. 0 if they have never ordered.",
                      min_value=0.0, max_value=200_000.0, step=10.0, key="in_usual")
            with b:
                field(st.number_input, "Days since last refusal", "Only used if the buyer has refused before.",
                      min_value=0, max_value=5000, step=1, key="in_last_refusal", disabled=st.session_state["in_refusals"] == 0)
                field(st.number_input, "Past orders in the last 90 days", "Earlier COD orders (known result) from the last 90 days.",
                      min_value=0, max_value=1000, step=1, key="in_orders_90")
                field(st.number_input, "Refusals in the last 90 days", "Refusals among those last-90-day orders.",
                      min_value=0, max_value=1000, step=1, key="in_refusals_90")
                field(st.number_input, "COD orders in the last 30 days", "Orders placed in the last 30 days, including ones still on the way.",
                      min_value=0, max_value=500, step=1, key="in_last30")
                field(st.number_input, "Days since first COD order", "How long the buyer has used COD. 0 if this is the first.",
                      min_value=0.0, max_value=5000.0, step=1.0, key="in_days_first")
            with st.expander("Advanced"):
                field(st.number_input, "Earlier orders still on the way",
                      "Outcome not known yet. They count for 'value vs usual' and 'frequency', not for refusals.",
                      min_value=0, max_value=100, step=1, key="in_transit")

    s = st.session_state
    with result:
        errors = form_errors(s)
        if errors:
            st.markdown("##### Please fix the form")
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
        p, tier, reasons, phi = score_order(feats, s.in_hour, model)
        score_block(p, tier, reasons, params)
        notes = []
        if feats["is_campaign_day"]:
            notes.append(f"{s.in_date.day}.{s.in_date.month} is a campaign day")
        if feats["same_item_other_shops_48h"] > 3:
            notes.append("the training data had at most 3 other shops, so this score is extrapolated")
        if s.in_date.year != 2026 or not 2 <= s.in_date.month <= 7:
            notes.append("the model only saw Feb-Jul 2026 orders; other months are extrapolated")
        if notes:
            st.caption("Note: " + "; ".join(notes) + ".")

    st.divider()
    left, right = st.columns([1.6, 1])
    with left:
        factor_chart(phi, model)
    with right:
        st.markdown("##### What the buyer sees")
        st.markdown(phone_html(tier, float(s.in_value)), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page: Import data (+ the check pop-up)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Reading and checking the file ...", max_entries=3)
def analyse_file(name, data, area_rates_json):
    t0 = time.perf_counter()
    raw = importer.read_file(name, data)
    t1 = time.perf_counter()
    report = importer.check(raw, json.loads(area_rates_json))
    report["timing"] = {"read": t1 - t0, "check": time.perf_counter() - t1}
    return report


def secs(s):
    return f"{s * 1000:.0f} ms" if s < 1 else f"{s:.2f} s"


@st.cache_data(max_entries=1)
def template_files(examples):
    return importer.template_csv(examples), importer.template_excel(examples)


def completeness_table(report):
    """Per template column: present in the file? how many rows are filled / blank / unreadable."""
    raw, t, rows = report["raw"], report["table"], []
    for col, kind, required, _, _ in importer.TEMPLATE:
        if col not in raw.columns:
            rows.append({"Column": col, "Required": "yes" if required else "no", "In file": "missing",
                         "Filled": 0, "Blank": report["rows"], "Unreadable": 0})
            continue
        blank = int(importer._blank(raw[col]).sum())
        filled = int(t[col].notna().sum())
        rows.append({"Column": col, "Required": "yes" if required else "no", "In file": "yes",
                     "Filled": filled, "Blank": blank, "Unreadable": report["rows"] - filled - blank})
    out = pd.DataFrame(rows)
    share = out["Filled"] / max(report["rows"], 1) * 100
    out["Complete"] = [f"{v:.0f}%" if v == 100 else f"{min(v, 99.99):.2f}%" for v in share]  # never round up to 100%
    return out


@st.dialog("Check the imported file", width="large")
def import_check_dialog(name, report, params, model):
    n, bad = report["rows"], int(report["bad"].sum())
    noted_rows = pd.Series(False, index=report["bad"].index)
    for m in report["warnings"].values():
        noted_rows |= m
    noted = int((noted_rows & ~report["bad"]).sum())
    st.markdown(f"**{name}** · {n:,} rows · {len(report['raw'].columns)} columns")

    if report["missing_cols"]:
        st.error("The file is missing required column(s): **" + ", ".join(report["missing_cols"]) +
                 "**. Add them (see the template) and upload the file again.")
    c = st.columns(4)
    c[0].metric("Rows in file", f"{n:,}", border=True)
    c[1].metric("Ready to import", f"{n - bad:,}", border=True)
    c[2].metric("Rows with problems", f"{bad:,}", border=True, help="These rows are not imported.")
    c[3].metric("Rows with notes", f"{noted:,}", border=True, help="Imported and scored, but worth a look.")
    if report["extra_cols"]:
        st.caption("Columns not in the template are ignored: " + ", ".join(report["extra_cols"]))
    tm = report.get("timing")
    if tm:
        st.caption(f"Read the file in {secs(tm['read'])}, checked {n:,} rows in {secs(tm['check'])}.")

    t1, t2, t3 = st.tabs(["Column completeness", "Checks", f"Problem rows ({bad:,})"])
    with t1:
        st.dataframe(completeness_table(report), hide_index=True, width="stretch", height=300)
    with t2:
        summary = importer.summary_table(report)
        if summary.empty:
            st.success("Every row passed every check.")
        else:
            st.dataframe(summary, hide_index=True, width="stretch")
    with t3:
        if bad == 0:
            st.success("No problem rows.")
        else:
            rows = importer.problem_rows(report)
            st.dataframe(rows.head(importer.MAX_ERRORS_SHOWN), hide_index=True, width="stretch", height=260)
            if bad > importer.MAX_ERRORS_SHOWN:
                st.caption(f"Showing the first {importer.MAX_ERRORS_SHOWN:,}; download the file for all {bad:,}.")
            st.download_button("Download problem rows (CSV)", rows.to_csv(index=False).encode("utf-8-sig"),
                               file_name="import_problems.csv", mime="text/csv")

    a, b = st.columns(2)
    ready = n - bad
    label = f"Import {ready:,} row(s)" + (f", skip {bad:,}" if bad and ready else "")
    if a.button(label, type="primary", disabled=ready == 0 or bool(report["missing_cols"]), width="stretch"):
        with st.spinner("Scoring the orders ..."):
            t0 = time.perf_counter()
            scored = importer.score(report, params, model)
            score_s = time.perf_counter() - t0
        st.session_state["imported"] = {"df": scored, "name": name, "rows": n, "skipped": bad,
                                        "at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                                        "timing": {**report.get("timing", {}), "score": score_s}}
        st.session_state["switch_source"] = IMPORTED
        st.rerun()
    if b.button("Cancel", width="stretch"):
        st.rerun()


def page_import(params, model, demo):
    st.title("COD Risk Score · Import data")
    st.caption(note() + " Upload your own orders (CSV or Excel). The app checks the file, builds the 20 model "
               "features and scores every order; all pages can then show them.")

    st.markdown("#### 1 · Get the template")
    csv_bytes, xlsx_bytes = template_files(importer.demo_as_template(demo, 100))
    c = st.columns([1, 1, 3])
    c[0].download_button("Template (Excel)", xlsx_bytes, file_name="cod_orders_template.xlsx", width="stretch",
                         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    c[1].download_button("Template (CSV)", csv_bytes, file_name="cod_orders_template.csv", mime="text/csv",
                         width="stretch")
    c[2].caption("One row per COD order, with what the platform knows at checkout. Both files hold 100 example "
                 "rows from the demo data; replace them with yours. The Excel file also has a 'columns' sheet.")
    with st.expander("What each column means"):
        st.dataframe(importer.column_guide(), hide_index=True, width="stretch")
        st.caption("History counts only earlier orders whose result was already known at checkout. "
                   "The courier failure rate counts only courier-caused failures in the area over the last 90 days.")

    st.markdown("#### 2 · Upload your file")
    up = st.file_uploader("CSV or Excel (.xlsx)", type=["csv", "xlsx", "xlsm"], key="import_file")
    if up is not None:
        try:
            report = analyse_file(up.name, up.getvalue(), json.dumps(params["area_latest_rate"]))
        except ValueError as e:
            st.error(str(e))
            report = None
        except Exception as e:  # unreadable file
            st.error(f"Could not read the file: {e}")
            report = None
        if report is not None:
            st.session_state["import_report"] = (up.name, report)
            if st.session_state.get("import_checked") != up.file_id:  # open the pop-up once per new file
                st.session_state["import_checked"] = up.file_id
                import_check_dialog(up.name, report, params, model)
    if st.session_state.get("import_report") and up is not None:
        name, report = st.session_state["import_report"]
        if st.button("Review the check again"):
            import_check_dialog(name, report, params, model)

    st.markdown("#### 3 · Imported data")
    imp = st.session_state.get("imported")
    if not imp:
        st.info("Nothing imported yet. The pages show the demo data.")
        return
    d = imp["df"]
    st.success(f"**{imp['name']}** imported at {imp['at']}: {len(d):,} orders scored"
               + (f", {imp['skipped']:,} row(s) skipped." if imp["skipped"] else "."))
    c = st.columns(4)
    c[0].metric("Orders", f"{len(d):,}", border=True)
    c[1].metric("Mean predicted risk", pct(d["risk"].mean(), 2), border=True)
    for col, k in zip(c[2:], (3, 4)):
        col.metric(f"{TIER_NAMES[k]}", f"{int((d['tier'] == k).sum()):,}", border=True)

    tm = imp.get("timing", {})
    if "score" in tm:
        st.markdown("##### Processing time")
        total = tm.get("read", 0) + tm.get("check", 0) + tm["score"]
        c = st.columns(5)
        c[0].metric("Read file", secs(tm.get("read", 0)), border=True)
        c[1].metric("Check rows", secs(tm.get("check", 0)), border=True,
                    help=f"Checked all {imp['rows']:,} rows in the file.")
        c[2].metric("Model scoring", secs(tm["score"]), border=True,
                    help="Builds the 20 features and computes risk, tier and reasons for every imported order.")
        c[3].metric("Total", secs(total), border=True)
        c[4].metric("Speed (orders / second)", f"{len(d) / max(total, 1e-9):,.0f}", border=True,
                    help=f"End to end. Model scoring alone: {len(d) / max(tm['score'], 1e-9):,.0f} orders/s.")
        st.caption("Measured on this computer when the file was imported. If the same file was checked "
                   "before, the read and check times shown are from that first check.")
    out = d[["order_id", "buyer_id", "order_datetime", "area_id", "order_value", "risk", "tier"]].assign(
        tier_name=d["tier_name"], action=d["tier"].map(TIER_ACTIONS))
    c = st.columns([1, 1, 3])
    c[0].download_button("Download scores (CSV)", out.to_csv(index=False).encode("utf-8-sig"),
                         file_name="cod_orders_scored.csv", mime="text/csv", width="stretch")
    if c[1].button("Remove imported data", width="stretch"):
        st.session_state.pop("imported", None)
        st.session_state["switch_source"] = DEMO
        st.rerun()
    c[2].caption("Pick **Imported file** under *Data* in the sidebar to see these orders on every page.")


# ---------------------------------------------------------------------------
# Entry point (called from app.py)
# ---------------------------------------------------------------------------
LAYOUT_CSS = """
<style>
/* Content: centred, never wider than 1440px, less empty space above the title */
[data-testid="stMainBlockContainer"], .block-container {max-width: 1440px; margin: 0 auto;
    padding-top: 2.2rem; padding-bottom: 3rem; padding-left: 2.5rem; padding-right: 2.5rem}
[data-testid="stSidebarContent"] [data-testid="stHeading"] h1 {color: #f47b20}
/* Orange rule under each page title */
[data-testid="stMainBlockContainer"] h1 {padding-bottom: .35rem; border-bottom: 3px solid #f47b20; margin-bottom: .4rem}
[data-testid="stMetricLabel"] p {font-size: 0.95rem; color: #b8b8be}
</style>"""


def render():
    st.markdown(LAYOUT_CSS, unsafe_allow_html=True)
    if not (APP_DATA / "orders.csv").exists():
        st.error("App data not found. Run `python -m src.export_app` inside cod_risk_demo/ "
                 "(or run_all.bat) first.")
        return
    # Keep filter / form values when switching pages (Streamlit drops state of hidden widgets).
    for k in list(st.session_state.keys()):
        if k.startswith(("f_", "fd_", "in_")) and not k.startswith("in_load_msg"):
            st.session_state[k] = st.session_state[k]

    params, demo = load()
    model = get_model(json.dumps(params))

    st.sidebar.title("COD Risk Score")
    st.sidebar.caption("ML model (calibrated Logistic Regression), trained on synthetic data. Not Shopee data.")
    # Direct links: ?page=dashboard|orders|score|import
    wanted = {"dashboard": 0, "orders": 1, "score": 2, "import": 3}.get(st.query_params.get("page", ""), 0)
    page = st.sidebar.radio("Page", PAGES, index=wanted, key="risk_page")
    df = data_source_picker(demo)

    if page == "Score an order":
        page_score(df, params, model)
    elif page == "Dashboard":  # draws its own filter row
        page_dashboard(df, params)
    elif page == "Orders":
        page_orders(df, params, model)
    else:
        page_import(params, model, demo)


def data_source_picker(demo):
    """Sidebar choice between the demo orders and an imported file. Returns the orders to show."""
    imp = st.session_state.get("imported")
    if "switch_source" in st.session_state:  # set by the import pop-up / remove button before the rerun
        st.session_state["data_source"] = st.session_state.pop("switch_source")
    if not imp:
        st.session_state["data_source"] = DEMO
    st.sidebar.radio("Data", [DEMO, IMPORTED] if imp else [DEMO], key="data_source",
                     captions=[f"{len(demo):,} synthetic orders"]
                     + ([f"{imp['name']} · {len(imp['df']):,} orders"] if imp else []))
    source = st.session_state["data_source"]
    if st.session_state.get("active_source") != source:  # filter ranges and choices come from the data: start fresh
        if "active_source" in st.session_state:
            for prefix in ("f_", "fd_"):
                reset_filters(prefix)
            st.session_state.pop("orders_table", None)
            st.session_state.pop("in_load_id", None)
        st.session_state["active_source"] = source
    return imp["df"] if source == IMPORTED and imp else demo


if __name__ == "__main__":  # same as the root app.py, so either file works as the start file
    st.set_page_config(page_title="COD Risk Score", page_icon="🛡️", layout="wide")
    render()
