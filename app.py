"""COD Trust Score — Streamlit demo (Scorecard + LLM, Phase 1).

Run:  streamlit run app.py
"""
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.generate_data import generate
from src.llm_review import review
from src.scorecard import TIER_ORDER, needs_llm_review, score_user, tier_policy

DATA = Path(__file__).parent / "data" / "users.csv"
TIER_COLORS = {"A": "#2E7D32", "B": "#1565C0", "C": "#EF6C00", "D": "#C62828"}

st.set_page_config(page_title="COD Trust Score", page_icon="🛡️", layout="wide")

# Streamlit Cloud secrets -> env var for the Anthropic SDK
try:
    if "ANTHROPIC_API_KEY" in st.secrets and not os.getenv("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass


@st.cache_data
def load_scored() -> pd.DataFrame:
    df = pd.read_csv(DATA) if DATA.exists() else generate()
    df["last_refusal_reason"] = df["last_refusal_reason"].fillna("")
    rows = []
    for u in df.to_dict("records"):
        r = score_user(u)
        rows.append({"score": r.score, "tier": r.tier, "borderline": r.borderline,
                     "llm_needed": needs_llm_review(u, r)})
    return pd.concat([df, pd.DataFrame(rows)], axis=1)


def fmt_limit(limit) -> str:
    if limit is None:
        return "ไม่จำกัด"
    return "งด COD" if limit == 0 else f"฿{limit:,}"


df = load_scored()
has_key = bool(os.getenv("ANTHROPIC_API_KEY"))

st.sidebar.title("🛡️ COD Trust Score")
st.sidebar.caption("ระบบประเมินสิทธิ์ COD ด้วย Scorecard + LLM")
page = st.sidebar.radio("หน้า", ["ภาพรวม", "ค้นหาผู้ใช้", "จำลองการสั่งซื้อ COD", "นโยบาย Tier"])
use_llm = st.sidebar.toggle("ใช้ Claude จริง", value=has_key, disabled=not has_key,
                            help="ตั้งค่า ANTHROPIC_API_KEY เพื่อเปิดใช้ ถ้าไม่มีจะใช้โหมดจำลอง (offline)")
st.sidebar.info("LLM: " + ("Claude API" if use_llm else "โหมดจำลอง (offline)"))
st.sidebar.caption("ข้อมูลทั้งหมดเป็นข้อมูลจำลอง (synthetic)")

# ---------------------------------------------------------------- Overview
if page == "ภาพรวม":
    st.title("ภาพรวมระบบ COD Trust Score")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ผู้ใช้ทั้งหมด", f"{len(df):,}")
    c2.metric("งด COD (Tier D)", f"{(df.tier == 'D').sum():,}")
    c3.metric("ส่งให้ LLM ตรวจ", f"{df.llm_needed.mean():.0%}")
    base_rate = df.refused_count.sum() / max(df.cod_orders.sum(), 1)
    kept = df[df.tier != "D"]
    new_rate = kept.refused_count.sum() / max(kept.cod_orders.sum(), 1)
    c4.metric("อัตราตีกลับ (ถ้าตัด Tier D)", f"{new_rate:.1%}", f"{(new_rate - base_rate):.1%}",
              delta_color="inverse")

    left, right = st.columns(2)
    counts = df.tier.value_counts().reindex(TIER_ORDER, fill_value=0).reset_index()
    counts.columns = ["tier", "users"]
    left.plotly_chart(px.bar(counts, x="tier", y="users", color="tier",
                             color_discrete_map=TIER_COLORS, title="จำนวนผู้ใช้แต่ละ Tier"),
                      width="stretch")
    right.plotly_chart(px.histogram(df, x="score", color="tier", nbins=30,
                                    color_discrete_map=TIER_COLORS,
                                    category_orders={"tier": TIER_ORDER},
                                    title="การกระจายของคะแนน"),
                       width="stretch")

    st.subheader("ระบบจับกลุ่ม abuse ได้แม่นแค่ไหน (เทียบกับ ground truth ของข้อมูลจำลอง)")
    ct = pd.crosstab(df.segment_truth, df.tier).reindex(columns=TIER_ORDER, fill_value=0)
    st.dataframe(ct, width="stretch")
    caught = (df[df.segment_truth == "abuser"].tier.isin(["C", "D"])).mean()
    fp = (df[df.segment_truth == "loyal"].tier.isin(["C", "D"])).mean()
    st.caption(f"ผู้ใช้สาย abuse ถูกจัดเป็น Tier C/D: **{caught:.0%}** · "
               f"ลูกค้าประจำถูกจัดผิดเป็น C/D (false positive): **{fp:.0%}**")

# ---------------------------------------------------------------- User lookup
elif page == "ค้นหาผู้ใช้":
    st.title("ค้นหาผู้ใช้")
    quick = st.selectbox("ตัวอย่างเคส", ["—", "ผู้ใช้สาย abuse", "ลูกค้าประจำ", "ผู้ใช้ใหม่", "ตีกลับเพราะร้านผิด"])
    samples = {
        "ผู้ใช้สาย abuse": df[df.segment_truth == "abuser"],
        "ลูกค้าประจำ": df[df.segment_truth == "loyal"],
        "ผู้ใช้ใหม่": df[df.segment_truth == "new"],
        "ตีกลับเพราะร้านผิด": df[df.last_refusal_reason.str.contains("ไม่ตรงปก|ส่งช้า|เสียหาย|ผิดรุ่น")],
    }
    default_id = samples[quick].iloc[0].user_id if quick in samples else df.iloc[0].user_id
    uid = st.text_input("User ID", value=default_id)
    row = df[df.user_id == uid.strip()]
    if row.empty:
        st.warning("ไม่พบผู้ใช้")
        st.stop()
    u = row.iloc[0].to_dict()
    r = score_user(u)

    c1, c2, c3 = st.columns(3)
    c1.metric("คะแนน Scorecard", r.score)
    c2.metric("Tier (Scorecard)", r.tier)
    pol = tier_policy(r.tier)
    c3.metric("วงเงิน COD", fmt_limit(pol["cod_limit_thb"]))

    left, right = st.columns([3, 2])
    with left:
        st.subheader("เหตุผลของคะแนน")
        rs = pd.DataFrame(r.reasons, columns=["คะแนน", "เหตุผล"])
        fig = px.bar(rs, x="คะแนน", y="เหตุผล", orientation="h",
                     color=rs["คะแนน"] > 0, color_discrete_map={True: "#2E7D32", False: "#C62828"})
        fig.update_layout(showlegend=False, yaxis_title=None, height=320)
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("ข้อมูลผู้ใช้")
        show = {k: u[k] for k in ["account_age_days", "kyc_verified", "cod_orders", "success_count",
                                  "refused_count", "shared_device_accounts", "address_changes_90d",
                                  "last_refusal_reason"]}
        st.table(pd.Series(show, name="ค่า").astype(str))

    st.subheader("🤖 การตรวจโดย LLM")
    if needs_llm_review(u, r):
        st.caption("เคสนี้ถูกส่งให้ LLM เพราะอยู่ใกล้เส้นแบ่ง Tier, จะถูกตัดสิทธิ์ หรือมีเหตุผลการตีกลับที่ต้องตีความ")
        if st.button("ให้ LLM วิเคราะห์"):
            with st.spinner("กำลังวิเคราะห์..."):
                d = review(u, r, use_real_llm=use_llm)
            final = tier_policy(d["tier"])
            a, b, c = st.columns(3)
            a.metric("Tier สุดท้าย", d["tier"], None if d["tier"] == r.tier else f"จาก {r.tier}")
            b.metric("ความผิดของ", d.get("fault", "-"))
            c.metric("ตีกลับได้ / 90 วัน", final["refusals_allowed_90d"])
            for reason in d["reasons"]:
                st.write("•", reason)
            for n in d["guardrail_notes"]:
                st.warning("Guardrail: " + n)
            with st.expander("ข้อมูลที่ส่งให้ LLM (ไม่มีข้อมูลระบุตัวตน)"):
                st.code(d["prompt"])
    else:
        st.success("เคสชัดเจน Scorecard ตัดสินได้เลย ไม่ต้องเรียก LLM (ประหยัดค่าใช้จ่าย)")

# ---------------------------------------------------------------- Checkout sim
elif page == "จำลองการสั่งซื้อ COD":
    st.title("จำลองหน้า Checkout")
    c1, c2 = st.columns(2)
    uid = c1.selectbox("ผู้ใช้", df.user_id.tolist())
    amount = c2.number_input("ยอดสั่งซื้อ (บาท)", 50, 50000, 1500, step=100)
    u = df[df.user_id == uid].iloc[0].to_dict()
    r = score_user(u)
    pol = tier_policy(r.tier)
    refusals_left = pol["refusals_allowed_90d"] - int(u["refused_90d"])

    if st.button("เลือกชำระเงินปลายทาง (COD)", type="primary"):
        limit = pol["cod_limit_thb"]
        if limit == 0:
            st.error("❌ บัญชีนี้ยังใช้ COD ไม่ได้ กรุณาชำระผ่าน ShopeePay / บัตร / โอนเงิน")
            st.caption("รับของสำเร็จต่อเนื่องเพื่อปลดล็อก COD อีกครั้ง หรือยื่นอุทธรณ์ได้")
        elif refusals_left <= 0:
            st.error("❌ ใช้สิทธิ์ตีกลับครบแล้วในรอบ 90 วัน ใช้ COD ได้อีกครั้งเมื่อครบรอบ")
        elif limit is not None and amount > limit:
            st.warning(f"⚠️ ยอดเกินวงเงิน COD ของคุณ ({fmt_limit(limit)}) กรุณาชำระล่วงหน้า")
        else:
            st.success(f"✅ อนุญาต COD · Tier {r.tier} · ตีกลับได้อีก {refusals_left} ครั้งใน 90 วัน")
    st.caption(f"Tier {r.tier} ({pol['label']}) · วงเงิน {fmt_limit(pol['cod_limit_thb'])} · คะแนน {r.score}")

# ---------------------------------------------------------------- Policy
else:
    st.title("นโยบาย Tier")
    st.table(pd.DataFrame([{**tier_policy(t), "cod_limit_thb": fmt_limit(tier_policy(t)["cod_limit_thb"])}
                           for t in TIER_ORDER]).rename(columns={
        "tier": "Tier", "cod_limit_thb": "วงเงิน COD", "refusals_allowed_90d": "ตีกลับได้/90 วัน",
        "label": "ความหมาย"}))
    st.markdown("""
**Flow การตัดสิน**
1. Scorecard ให้คะแนนจากพฤติกรรม (โค้ด ไม่ใช่ AI) → เคสชัดเจนได้ Tier ทันที
2. เคสก้ำกึ่ง / จะถูกตัดสิทธิ์ / มีเหตุผลเป็นข้อความ → ส่งให้ LLM วิเคราะห์
3. Guardrail: LLM ปรับได้ไม่เกิน ±1 ขั้น, ห้ามตัดสิทธิ์ COD เอง, JSON ผิด → ใช้ผล Scorecard
4. บันทึก log ทุกการตัดสิน · ผู้ใช้อุทธรณ์ได้ · รับของสำเร็จแล้วคะแนนฟื้น

**Phase 2:** เมื่อเก็บผลออเดอร์จริงได้พอ เทรนโมเดล XGBoost มาแทน Scorecard
""")
