# Yelp Star Prediction

A 2-person ML class project predicting Yelp business `stars` rating from
business metadata (location, categories, attributes, hours, reviews).

## Project layout

```
data/raw/           raw Yelp JSON (never committed)
data/processed/     train/test parquet files (never committed)
models/             saved .joblib models (never committed)
results/            CSV results, reports, and comparison plots
src/                config.py, preprocess.py, train.py, compare.py
notebooks/          run_colab.ipynb — end-to-end Colab runner
```

## Setup

1. Clone the repo and create a virtual environment:
   ```
   git clone <repo-url>
   cd ml-project
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Get `yelp_academic_dataset_business.json` from the team's **shared Google
   Drive folder** (`MyDrive/yelp-project/data/raw/`). This file is large and
   is **never committed** to git — it's ignored via `.gitignore`.
   Place it at `data/raw/yelp_academic_dataset_business.json` locally, or
   work entirely in Colab (see below), where `data/` and `models/` are
   symlinked to the shared Drive folder.

## Where the data lives

- Raw data: shared Drive folder `MyDrive/yelp-project/data/raw/`
- Processed train/test parquet files: `MyDrive/yelp-project/data/processed/`
- Saved models: `MyDrive/yelp-project/models/`

None of these are committed to git — only `results/` (small CSVs and plots)
and code are tracked.

## Running the pipeline

```bash
# 1. Preprocess raw JSON into train/test parquet files
python src/preprocess.py --input data/raw/yelp_academic_dataset_business.json

# 2. Train a model (ridge, rf, knn, or hgb)
python src/train.py --model ridge
python src/train.py --model rf --sample 30000 --n-iter 4
python src/train.py --model knn
python src/train.py --model hgb

# 3. Compare all trained models
python src/compare.py
```

Each `train.py` run appends a row to `results/results.csv`, saves
`results/<model>_cv_results.csv`, and (for ridge/rf) saves coefficients or
feature importances. `compare.py` prints a sorted table and saves
`results/comparison.png`.

Data is split **60% train / 10% validation / 30% test** (`SEED=42`) in
`preprocess.py`. Hyperparameters are tuned with 5-fold CV on the training
set; `train.py` reports both validation and test metrics (`val_RMSE`,
`val_MAE`, `val_R2`, `RMSE`, `MAE`, `R2`).

## Running in Colab

Open `notebooks/run_colab.ipynb` in Google Colab and run the cells in
order: mount Drive, clone the repo, install requirements, symlink
`data/` and `models/` to the shared Drive folder, then run preprocess,
train ridge, train rf (on a 30k-row sample), and compare.

## Loading a saved model

```python
import joblib

model = joblib.load("models/ridge.joblib")
predictions = model.predict(X_test)
```

Note: `ridge` and `knn` models are saved as a `Pipeline(StandardScaler, model)`,
so scaling is applied automatically. `rf` and `hgb` models take unscaled
features directly.

#### Training Commands

<details>
<summary>View training commands</summary>

### Ridge Regression

```bash
python src/train.py --model ridge
```

### Random Forest

```bash
python src/train.py --model rf --sample 30000 --n-iter 4
```

### HistGradientBoosting

```bash
python src/train.py --model hgb
```

The training script evaluates models on validation and test datasets and saves trained models as `.joblib` files.

</details>

## Initial Experimental Results

<details>
<summary>View initial results and findings</summary>

### Test Performance

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Ridge Regression | 0.8431 | 0.6695 | 0.2590 |
| Random Forest | 0.7948 | 0.6258 | 0.3415 |
| HistGradientBoosting | 0.7950 | 0.6227 | 0.3411 |
| Baseline (Training Mean) | 0.9795 | 0.8036 | ≈ 0.0000 |

Lower RMSE and MAE indicate better prediction performance. Higher R² indicates better explanatory performance.

### Initial Findings

- Random Forest achieved the best test RMSE and R².
- HistGradientBoosting achieved the lowest test MAE.
- Both tree-based models outperformed Ridge Regression.
- All three machine learning models outperformed the baseline.
- Random Forest and HistGradientBoosting produced very similar results.

These are initial experimental results and may be updated as the project develops.

</details>

## Comparing Models

<details>
<summary>View model comparison instructions</summary>

Run:

```bash
python src/compare.py
```

This generates:

- A sorted model comparison table.
- `results/results.csv`
- `results/comparison.png`

</details>

## Future Work

<details>
<summary>View planned improvements and experiments</summary>

- Investigate K-Nearest Neighbors (KNN) as an additional regression model.
- Perform feature importance analysis.
- Investigate model errors and prediction patterns.
- Explore additional feature engineering techniques.
- Improve preprocessing and evaluation methodology.
- Compare additional model configurations.

</details>