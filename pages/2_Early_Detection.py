import streamlit as st
from _shared import load_table, empty_state

st.title("Early Detection")
st.caption("Score >= 75 AND the stock has NOT already made a large move (SPEC #36). "
           "This is the core purpose of PRE-RUN.")

scores = load_table("scores")
if scores.empty:
    empty_state("No scores in the database yet.")
    st.stop()

latest_date = scores["date"].max()
today = scores[scores["date"] == latest_date]
early = today[(today["prerun_score"] >= 75) & (today["already_running"] == 0)] \
    .sort_values("prerun_score", ascending=False)

if early.empty:
    st.info(f"No qualifying setups as of {latest_date}. That's a valid, honest result — not every day has one.")
else:
    st.dataframe(
        early[["ticker", "prerun_score", "classification", "market_regime"]].reset_index(drop=True),
        use_container_width=True,
    )
