import streamlit as st
import pandas as pd
from _shared import load_table, empty_state

st.title("Backtest")
st.caption("Does a higher PRE-RUN score actually correspond to a higher probability "
           "of a future run? (SPEC #22)")

runs = load_table("backtest_runs")
if runs.empty:
    empty_state("No backtest runs stored yet.")
    st.stop()

run_id = st.selectbox("Backtest run", runs["run_id"].tolist(),
                       format_func=lambda i: f"Run {i} — "
                       f"{runs[runs.run_id==i]['model_version'].iloc[0]} "
                       f"({runs[runs.run_id==i]['dataset_split'].iloc[0]})")

row = runs[runs.run_id == run_id].iloc[0]
st.markdown(f"**Model:** `{row['model_version']}` &nbsp; | &nbsp; "
            f"**Split:** `{row['dataset_split']}` &nbsp; | &nbsp; "
            f"**Window:** {row['start_date']} → {row['end_date']} &nbsp; | &nbsp; "
            f"**Signals:** {row['n_signals']}")

import json
results = json.loads(row["results_json"]) if row["results_json"] else {}
if "bucket_table" in results:
    st.subheader("Hit rate by score bucket")
    st.dataframe(pd.DataFrame(results["bucket_table"]), use_container_width=True)
else:
    st.json(results)

st.warning(
    "Remember: this is one run/window/model_version. Check the "
    "OUT_OF_SAMPLE split specifically before trusting any pattern seen in "
    "TRAINING or VALIDATION (SPEC #27)."
)
