# Yelp Star Prediction

A 2-person ML class project predicting Yelp business `stars` rating from
business metadata (location, categories, attributes, hours, reviews).

## Project layout

```
yelp-star-prediction/
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
   cd yelp-star-prediction
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

## Task breakdown

| Person   | Responsibilities                                              |
|----------|-----------------------------------------------------------------|
| Person A | Preprocessing write-up (cleaning, encoding, outliers) + Ridge model |
| Person B | Encoding/scaling write-up + Random Forest model                 |
| Both     | Model comparison (`compare.py`, `results/comparison.png`) + sentiment analysis stretch goal |
