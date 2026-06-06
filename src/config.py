""" project wide configuration for the cmapss rul project"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_DATASET = "FD001"
DEFAULT_RUL_CAP = 140
RANDOM_STATE = 42
SELECTED_SENSORS = ["s_7", "s_12", "s_14", "s_20", "s_21"]