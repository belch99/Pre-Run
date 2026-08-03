import streamlit as st
from _shared import load_table, empty_state

st.title("Paper Trades")
st.caption("Every hypothetical trade PRE-RUN has ever generated. Nothing here is ever deleted (SPEC #24).")

trades = load_table("paper_trades")
if trades.empty:
    empty_state("No paper trades yet.")
    st.stop()

status_filter = st.multiselect("Status", trades["status"].unique().tolist(),
                                default=trades["status"].unique().tolist())
view = trades[trades["status"].isin(status_filter)]

cols = ["ticker", "signal_date", "entry_price", "status", "shares", "capital_allocated",
        "dollar_risk", "target1", "stop_price", "return_pct", "return_dollar", "score"]
cols = [c for c in cols if c in view.columns]
st.dataframe(view[cols].sort_values("signal_date", ascending=False), use_container_width=True)

closed = view[view["status"].isin(["STOPPED", "TARGET1", "TARGET2"])]
if not closed.empty:
    win_rate = (closed["return_pct"] > 0).mean() * 100
    st.metric("Win rate (closed trades)", f"{win_rate:.1f}%")
    st.metric("Total realized P/L", f"${closed['return_dollar'].sum():,.2f}")
