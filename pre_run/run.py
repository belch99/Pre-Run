"""Convenience launcher.

    python run.py init          # create database
    python run.py scan          # run today's daily scan
    python run.py experiment    # run the first historical backtest
    python run.py walkforward   # run walk-forward out-of-sample tests
    python run.py dashboard     # launch the Streamlit dashboard
"""
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == "init":
        from src.data.db import init_db
        from src.utils.config import load_config, db_path
        init_db(db_path(load_config()))
        print("Database initialized.")
    elif cmd == "scan":
        subprocess.run([sys.executable, "scripts/run_daily_scan.py"], cwd=ROOT)
    elif cmd == "experiment":
        subprocess.run([sys.executable, "scripts/run_first_experiment.py"], cwd=ROOT)
    elif cmd == "walkforward":
        subprocess.run([sys.executable, "scripts/walk_forward_test.py"], cwd=ROOT)
    elif cmd == "dashboard":
        subprocess.run(["streamlit", "run", "app.py"], cwd=ROOT)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
