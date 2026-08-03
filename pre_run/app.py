import streamlit as st
from src.utils.config import load_config

st.set_page_config(page_title="PRE-RUN", page_icon="\U0001F4C8", layout="wide")

cfg = load_config()

st.title("PRE-RUN")
st.caption("Find the move before the move.")

st.markdown(f"""
**Model version:** `{cfg['model_version']}` &nbsp;|&nbsp; **Universe mode:** `{cfg['universe']['mode']}`

This is a research and paper-trading engine. **It never places real trades.**
Scores are model outputs, not probabilities, and are not investment advice.

Use the sidebar to navigate:
- **Command Center** — ranked PRE-RUN candidates
- **Early Detection** — stocks that haven't already run
- **Imminent** — score >= 85
- **Backtest** — score-bucket hit rates, out-of-sample results
- **Paper Trades** — every live hypothetical trade
- **Performance** — overall system track record
- **Model Lab** — adjust weights/thresholds and re-run experiments

---
### Current status
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Components with live free data", "4 / 9", help="Momentum, Volume, Compression, Breakout. See README for phase plan.")
with col2:
    st.metric("Data cost", "$0/month")
with col3:
    st.metric("Real trades placed", "0", help="This system is paper-trading / research only, by design (SPEC #84).")

st.info(
    "Catalyst, options activity, short interest, insider activity, and news-attention "
    "components are marked **N/A** until Phase 7-9 data sources are connected — "
    "they are never assigned a fake score. See `MODEL_METHODOLOGY.md`."
)
