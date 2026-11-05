"""
train.py
Trains one of three regression models (ridge, rf, hgb) on the
preprocessed Yelp features, tunes hyperparameters with 5-fold KFold CV
(optionally on a random subset via --sample, then refits best params on
the full training set), evaluates on the held-out test set, and saves
the fitted model plus CV/importance results.

Usage:
    python src/train.py --model ridge
    python src/train.py --model rf --sample 30000 --n-iter 4
"""
import argparse
import csv
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import (
    SEED,
    MODELS_DIR,
    RESULTS_DIR,
    X_TRAIN_PATH,
    X_VAL_PATH,
    X_TEST_PATH,
    Y_TRAIN_PATH,
    Y_VAL_PATH,
    Y_TEST_PATH,
    RESULTS_CSV_PATH,
)


def load_data():
    X_train = pd.read_parquet(X_TRAIN_PATH)
    X_val = pd.read_parquet(X_VAL_PATH)
    X_test = pd.read_parquet(X_TEST_PATH)
    y_train = pd.read_parquet(Y_TRAIN_PATH)["stars"].values
    y_val = pd.read_parquet(Y_VAL_PATH)["stars"].values
    y_test = pd.read_parquet(Y_TEST_PATH)["stars"].values
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_search(model_name, cv, n_iter):
    """Build the (estimator, param_grid, search_cls) for each model, scaled where needed."""
    if model_name == "ridge":
        estimator = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
        param_grid = {"model__alpha": np.logspace(-3, 3, 13)}
        return GridSearchCV(estimator, param_grid, cv=cv,
                             scoring="neg_root_mean_squared_error", n_jobs=-1)

    if model_name == "rf":
        estimator = RandomForestRegressor(random_state=SEED, n_jobs=-1)
        param_grid = {
            "n_estimators": [200, 400],
            "max_depth": [None, 10, 20, 30],
            "min_samples_leaf": [1, 2, 5, 10],
            "max_features": ["sqrt", 0.3, 0.5],
        }
        return RandomizedSearchCV(estimator, param_grid, n_iter=n_iter, cv=cv,
                                   scoring="neg_root_mean_squared_error",
                                   random_state=SEED, n_jobs=-1)

    if model_name == "hgb":
        estimator = HistGradientBoostingRegressor(random_state=SEED)
        param_grid = {
            "max_iter": [100, 200, 400],
            "max_depth": [None, 5, 10, 20],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "l2_regularization": [0.0, 0.1, 1.0],
        }
        return RandomizedSearchCV(estimator, param_grid, n_iter=n_iter, cv=cv,
                                   scoring="neg_root_mean_squared_error",
                                   random_state=SEED, n_jobs=-1)

    raise ValueError(f"Unknown model: {model_name}")


def report(name, y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print({"model": name, "RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)})
    return rmse, mae, r2


def append_results_row(model_name, best_params, val_rmse, val_mae, val_r2, rmse, mae, r2):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "model": model_name,
        "best_params": best_params,
        "val_RMSE": val_rmse,
        "val_MAE": val_mae,
        "val_R2": val_r2,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    file_exists = RESULTS_CSV_PATH.exists()
    with open(RESULTS_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_extra_outputs(model_name, search, X_train):
    """Save CV results, and feature importances/coefficients where relevant."""
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.to_csv(RESULTS_DIR / f"{model_name}_cv_results.csv", index=False)

    best = search.best_estimator_
    if model_name == "rf":
        imp = pd.Series(best.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        imp.to_csv(RESULTS_DIR / "rf_importances.csv")
    elif model_name == "ridge":
        ridge_model = best.named_steps["model"]
        coef = pd.Series(ridge_model.coef_, index=X_train.columns).sort_values(key=abs, ascending=False)
        coef.to_csv(RESULTS_DIR / "ridge_coefficients.csv")


def main():
    parser = argparse.ArgumentParser(description="Train a model on the Yelp dataset.")
    parser.add_argument("--model", required=True, choices=["ridge", "rf", "hgb"])
    parser.add_argument("--sample", type=int, default=None,
                         help="Tune on a random subset of this many training rows.")
    parser.add_argument("--n-iter", type=int, default=12,
                         help="Number of iterations for RandomizedSearchCV (ignored for ridge).")
    args = parser.parse_args()

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    cv = KFold(5, shuffle=True, random_state=SEED)

    if args.sample is not None and args.sample < len(X_train):
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(X_train), size=args.sample, replace=False)
        X_tune, y_tune = X_train.iloc[idx], y_train[idx]
    else:
        X_tune, y_tune = X_train, y_train

    search = get_search(args.model, cv, args.n_iter)
    search.fit(X_tune, y_tune)
    print(f"best {args.model} params:", search.best_params_)

    # Refit best params on the full training set
    best_estimator = search.best_estimator_
    best_estimator.fit(X_train, y_train)

    y_val_pred = best_estimator.predict(X_val)
    val_rmse, val_mae, val_r2 = report(f"{args.model} (val)", y_val, y_val_pred)

    y_pred = best_estimator.predict(X_test)
    rmse, mae, r2 = report(f"{args.model} (test)", y_test, y_pred)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_estimator, MODELS_DIR / f"{args.model}.joblib", compress=3)

    append_results_row(args.model, search.best_params_, val_rmse, val_mae, val_r2, rmse, mae, r2)
    save_extra_outputs(args.model, search, X_train)

    print(f"Saved model to models/{args.model}.joblib")
    print(f"Appended results to {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()
