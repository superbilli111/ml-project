"""
config.py
Shared configuration: file paths, random seed, and feature-engineering constants
used across preprocess.py, train.py, and compare.py.
"""
from pathlib import Path

# ---------- Paths ----------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

X_TRAIN_PATH = DATA_PROCESSED_DIR / "X_train.parquet"
X_VAL_PATH = DATA_PROCESSED_DIR / "X_val.parquet"
X_TEST_PATH = DATA_PROCESSED_DIR / "X_test.parquet"
Y_TRAIN_PATH = DATA_PROCESSED_DIR / "y_train.parquet"
Y_VAL_PATH = DATA_PROCESSED_DIR / "y_val.parquet"
Y_TEST_PATH = DATA_PROCESSED_DIR / "y_test.parquet"

PREPROCESSING_REPORT_PATH = RESULTS_DIR / "preprocessing_report.txt"
RESULTS_CSV_PATH = RESULTS_DIR / "results.csv"
COMPARISON_PNG_PATH = RESULTS_DIR / "comparison.png"

# ---------- Constants ----------
SEED = 42
TOP_CATS = 100          # keep the 100 most common categories
MIN_ATTR_FRAC = 0.05    # drop attribute columns present in <5% of businesses

# 60 / 10 / 30 train / val / test split
TRAIN_SIZE = 0.6
VAL_SIZE = 0.1
TEST_SIZE = 0.3

# Make sure output directories exist
for d in (DATA_PROCESSED_DIR, MODELS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
