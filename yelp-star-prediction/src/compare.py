"""
compare.py
Reads results/results.csv, adds a train-mean baseline row, prints a
sorted comparison table, and saves results/comparison.png containing a
bar chart of RMSE by model plus a predicted-vs-actual scatter for each
saved model.

Usage:
    python src/compare.py
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import (
    MODELS_DIR,
    RESULTS_CSV_PATH,
    COMPARISON_PNG_PATH,
    X_TRAIN_PATH,
    X_VAL_PATH,
    X_TEST_PATH,
    Y_TRAIN_PATH,
    Y_VAL_PATH,
    Y_TEST_PATH,
)

MODEL_NAMES = ["ridge", "rf", "hgb"]


def load_data():
    X_train = pd.read_parquet(X_TRAIN_PATH)
    X_val = pd.read_parquet(X_VAL_PATH)
    X_test = pd.read_parquet(X_TEST_PATH)
    y_train = pd.read_parquet(Y_TRAIN_PATH)["stars"].values
    y_val = pd.read_parquet(Y_VAL_PATH)["stars"].values
    y_test = pd.read_parquet(Y_TEST_PATH)["stars"].values
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_baseline_row(y_train, y_val, y_test):
    val_pred = np.full(len(y_val), y_train.mean())
    pred = np.full(len(y_test), y_train.mean())
    return {
        "model": "baseline (train mean)",
        "best_params": "{}",
        "val_RMSE": mean_squared_error(y_val, val_pred) ** 0.5,
        "val_MAE": mean_absolute_error(y_val, val_pred),
        "val_R2": r2_score(y_val, val_pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
        "MAE": mean_absolute_error(y_test, pred),
        "R2": r2_score(y_test, pred),
        "timestamp": "",
    }


def load_saved_models():
    models = {}
    for name in MODEL_NAMES:
        path = MODELS_DIR / f"{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)
    return models


def plot_comparison(results_df, models, X_test, y_test):
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(5 * (n_models + 1), 4.5))

    # Bar chart of RMSE by model
    ax0 = axes[0]
    sorted_df = results_df.sort_values("RMSE")
    ax0.bar(sorted_df["model"], sorted_df["RMSE"], color="steelblue")
    ax0.set_title("RMSE by model")
    ax0.set_ylabel("RMSE")
    ax0.tick_params(axis="x", rotation=45)

    # Predicted vs actual scatter for each saved model
    for ax, (name, model) in zip(axes[1:], models.items()):
        y_pred = model.predict(X_test)
        ax.scatter(y_test, y_pred, alpha=0.2, s=8)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_title(f"{name}: predicted vs actual")
        ax.set_xlabel("actual stars")
        ax.set_ylabel("predicted stars")

    fig.tight_layout()
    fig.savefig(COMPARISON_PNG_PATH, dpi=150)
    print(f"Saved comparison plot to {COMPARISON_PNG_PATH}")


def main():
    if not RESULTS_CSV_PATH.exists():
        raise SystemExit(f"{RESULTS_CSV_PATH} not found. Run src/train.py first.")

    results_df = pd.read_csv(RESULTS_CSV_PATH)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    baseline_row = build_baseline_row(y_train, y_val, y_test)
    results_df = pd.concat([results_df, pd.DataFrame([baseline_row])], ignore_index=True)

    sorted_df = results_df.sort_values("RMSE").reset_index(drop=True)
    print(sorted_df.to_string(index=False))

    models = load_saved_models()
    if models:
        plot_comparison(results_df, models, X_test, y_test)
    else:
        print("No saved models found in models/, skipping plot.")


if __name__ == "__main__":
    main()
