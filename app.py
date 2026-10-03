"""COD Risk Score — Streamlit demo (ML model).

Run:  streamlit run app.py

Pages live in risk_ui/; the model and data come from the cod_risk_demo/ pipeline.
"""
import streamlit as st

from risk_ui.app import render

st.set_page_config(page_title="COD Risk Score", page_icon="🛡️", layout="wide")

render()
