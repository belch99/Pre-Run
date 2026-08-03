import streamlit as st
from _shared import load_table, empty_state

st.title("Command Center")
st.caption("Top PRE-RUN candidates, most recent trading day.")

scores = load_table("scores")
if scores.empty:
    empty_state("No scores in the database yet.")
    st.stop()

latest_date = scores["date"].max()
today = scores[scores["date"] == latest_date].sort_values("prerun_score", ascending=False)

st.subheader(f"As of {latest_date}")
cols = ["ticker", "prerun_score", "classification", "already_running",
        "max_possible_pts", "market_regime", "universe_mode"]
cols = [c for c in cols if c in today.columns]
st.dataframe(today[cols].reset_index(drop=True), use_container_width=True)

sel = st.selectbox("Inspect a ticker", today["ticker"].tolist()) if not today.empty else None
if sel:
    row = today[today["ticker"] == sel].iloc[0]
    st.markdown(f"### {sel} — {row['prerun_score']} ({row['classification']})")
    st.text(row.get("explanation", "No explanation stored."))
