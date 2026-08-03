import pandas as pd
import streamlit as st
from src.utils.config import load_config, db_path
from src.data.db import connect, init_db


@st.cache_data(ttl=300)
def load_table(table: str, where: str = "", params: tuple = ()) -> pd.DataFrame:
    cfg = load_config()
    dbp = db_path(cfg)
    init_db(dbp)
    with connect(dbp) as conn:
        q = f"SELECT * FROM {table}"
        if where:
            q += f" WHERE {where}"
        try:
            return pd.read_sql_query(q, conn, params=params)
        except Exception:
            return pd.DataFrame()


def empty_state(msg: str):
    st.warning(msg + "\n\nRun `python scripts/run_daily_scan.py` (on a machine with internet access) "
                      "to populate real signals, or `python scripts/run_first_experiment.py` "
                      "to run the first historical backtest.")
