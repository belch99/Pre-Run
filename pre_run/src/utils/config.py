from __future__ import annotations
import os
import logging
import logging.handlers
from pathlib import Path
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = None) -> dict:
    path = Path(path) if path else PROJECT_ROOT / "config" / "config.yaml"
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_env() -> None:
    env_path = PROJECT_ROOT / "config" / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # falls back to whatever is in the real environment; all keys optional
        load_dotenv()


_LOGGER_CACHE = {}


def get_logger(name: str) -> logging.Logger:
    """Console + rotating file logger, per SPEC #55."""
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        fh = logging.handlers.RotatingFileHandler(
            log_dir / "pre_run.log", maxBytes=5_000_000, backupCount=5
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    _LOGGER_CACHE[name] = logger
    return logger


def db_path(cfg: dict = None) -> Path:
    cfg = cfg or load_config()
    return PROJECT_ROOT / cfg["database"]["path"]
