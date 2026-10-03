# The pipeline package (python -m src.<step>). Importing it does nothing.
#
# Streamlit Community Cloud was set up with this file as its main file, so when it is
# started directly (`streamlit run cod_risk_demo/src/__init__.py`) it shows the app,
# exactly like the root app.py.
if __name__ == "__main__":
    import sys
    from pathlib import Path

    _root = str(Path(__file__).resolve().parents[2])
    if _root not in sys.path:
        sys.path.insert(0, _root)

    import streamlit as st

    from risk_ui.app import render

    st.set_page_config(page_title="COD Risk Score", page_icon="🛡️", layout="wide")
    render()
