import argparse
import ast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

/**
*{"business_id":"tUFrWirKiKi_TAnsVWINQQ","name":"Target", "city":"Tucson","state":"AZ",
 "latitude":32.22,"longitude":-110.88,"stars":3.5,"review_count":22,"is_open":0,
 "attributes":{"BikeParking":"True","RestaurantsPriceRange2":"2",
   "BusinessParking":"{'garage': False, 'street': False, ...}","WiFi":"u'no'", ...},
 "categories":"Department Stores, Shopping, Fashion, ...",
 "hours":{"Monday":"8:0-22:0", ...}}
*/

from config import (
    SEED,
    TOP_CATS,
    MIN_ATTR_FRAC,
    VAL_SIZE,
    TEST_SIZE,
    X_TRAIN_PATH,
    X_VAL_PATH,
    X_TEST_PATH,
    Y_TRAIN_PATH,
    Y_VAL_PATH,
    Y_TEST_PATH,
    PREPROCESSING_REPORT_PATH,
)

report_lines = []


def parse(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, str):
        v = v.strip()
        if v.startswith("u'") or v.startswith('u"'):
            v = v[1:]
        try:
            v = ast.literal_eval(v)
        except Exception:
            return v
    return v


def flatten_attrs(a):
    out = {}
    if not isinstance(a, dict):
        return out
    for k, v in a.items():
        v = parse(v)
        if isinstance(v, dict):
            for k2, v2 in v.items():
                out["attr_" + k + "_" + k2] = parse(v2)
        else:
            out["attr_" + k] = v
    return out


def hours_feats(h):
    if not isinstance(h, dict) or not h:
        return {"days_open": 0.0, "hours_per_week": 0.0, "hours_missing": 1.0}
    total = 0.0
    for span in h.values():
        try:
            start, end = span.split("-")
            start_h, start_m = map(int, start.split(":"))
            end_h, end_m = map(int, end.split(":"))
            length = (end_h + end_m / 60) - (start_h + start_m / 60)
            if length <= 0:
                length = length + 24
            total = total + length
        except Exception:
            pass
    return {"days_open": float(len(h)), "hours_per_week": total, "hours_missing": 0.0}


parser = argparse.ArgumentParser()
parser.add_argument("--input", default="data/raw/yelp_academic_dataset_business.json")
args = parser.parse_args()

df = pd.read_json(args.input, lines=True)
print("raw rows:", len(df))
report_lines.append("raw rows: " + str(len(df)))

df = df.drop_duplicates(subset="business_id")

df["city"] = df["city"].str.strip().str.title()
df["state"] = df["state"].str.strip().str.upper()
df["postal_code"] = df["postal_code"].astype(str).str.strip()

bad_loc = df["latitude"].isna() | df["longitude"].isna() | (df["state"].str.len() != 2)
print("invalid location rows dropped:", int(bad_loc.sum()))
report_lines.append("invalid location rows dropped: " + str(int(bad_loc.sum())))
df = df[~bad_loc]

dup = df.duplicated(subset=["name", "address", "postal_code"], keep="first")
print("name+address duplicates dropped:", int(dup.sum()))
report_lines.append("name+address duplicates dropped: " + str(int(dup.sum())))
df = df[~dup].reset_index(drop=True)

# base numeric columns
base = pd.DataFrame({
    "log_review_count": np.log1p(df["review_count"]),
    "latitude": df["latitude"],
    "longitude": df["longitude"],
    "is_open": df["is_open"].astype(float),
})

# one column per state, 1 or 0
states = pd.get_dummies(df["state"], prefix="state", dtype=float)
states = states.loc[:, states.sum() >= 50]

# hours columns
hours_list = []
for h in df["hours"]:
    hours_list.append(hours_feats(h))
hours = pd.DataFrame(hours_list)

# categories columns
cats = df["categories"].fillna("").str.split(r",\s*")
cats = cats.apply(lambda xs: [x for x in xs if x])
all_cats = []
for xs in cats:
    for c in xs:
        all_cats.append(c)
top_cats = pd.Series(all_cats).value_counts().head(TOP_CATS).index

cat_cols = {}
for c in top_cats:
    cat_cols["cat_" + c] = cats.apply(lambda xs, c=c: float(c in xs))
categories = pd.DataFrame(cat_cols)
categories["n_categories"] = cats.str.len().astype(float)

# attributes columns
attrs_list = []
for a in df["attributes"]:
    attrs_list.append(flatten_attrs(a))
attrs = pd.DataFrame(attrs_list)
attrs = attrs.loc[:, attrs.notna().mean() >= MIN_ATTR_FRAC]

numeric_cols = []
text_cols = []
for c in attrs.columns:
    values = set(attrs[c].dropna().unique())
    if values <= {True, False}:
        attrs[c] = attrs[c].map({True: 1.0, False: 0.0})
        numeric_cols.append(c)
    elif c == "attr_RestaurantsPriceRange2":
        attrs[c] = pd.to_numeric(attrs[c], errors="coerce")
        numeric_cols.append(c)
    else:
        attrs[c] = attrs[c].astype(str).str.strip("'").str.lower()
        text_cols.append(c)

for c in numeric_cols:
    attrs[c + "_missing"] = attrs[c].isna().astype(float)
    attrs[c] = attrs[c].fillna(0.0)

text_data = attrs[text_cols].replace("nan", "missing").fillna("missing")
text_dummies = pd.get_dummies(text_data, prefix=text_cols, dtype=float)
numeric_data = attrs[[c for c in attrs.columns if c not in text_cols]]
attrs = pd.concat([numeric_data, text_dummies], axis=1)

# put all the feature tables together
X = pd.concat([base, states, hours, categories, attrs], axis=1)
X.columns = [str(c) for c in X.columns]
y = df["stars"].astype(float).values
print("feature matrix:", X.shape)
report_lines.append("feature matrix: " + str(X.shape))

# print some stats about low star businesses, dont remove anything
low = df["stars"] <= 1.5
msg = "stars<=1.5: " + str(int(low.sum())) + ", of which review_count<10: " + str(int((low & (df["review_count"] < 10)).sum()))
print(msg)
report_lines.append(msg)
buckets = pd.cut(df["review_count"], [0, 10, 50, 200, 1e9])
stats = df.groupby(buckets)["stars"].agg(["mean", "std", "count"])
print("stars by review_count bucket:")
print(stats)
report_lines.append("stars by review_count bucket:")
report_lines.append(str(stats))

# split into 60% train, 10% val, 30% test
X_train, X_rest, y_train, y_rest = train_test_split(
    X, y, test_size=(VAL_SIZE + TEST_SIZE), random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_rest, y_rest, test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE), random_state=SEED
)
print("train rows:", len(X_train), "val rows:", len(X_val), "test rows:", len(X_test))
report_lines.append("train rows: " + str(len(X_train)) + " val rows: " + str(len(X_val)) + " test rows: " + str(len(X_test)))

PREPROCESSING_REPORT_PATH.write_text("\n".join(report_lines) + "\n")

X_train.reset_index(drop=True).to_parquet(X_TRAIN_PATH)
X_val.reset_index(drop=True).to_parquet(X_VAL_PATH)
X_test.reset_index(drop=True).to_parquet(X_TEST_PATH)
pd.DataFrame({"stars": y_train}).to_parquet(Y_TRAIN_PATH)
pd.DataFrame({"stars": y_val}).to_parquet(Y_VAL_PATH)
pd.DataFrame({"stars": y_test}).to_parquet(Y_TEST_PATH)

print("Saved train/val/test parquet files to data/processed/")
print("Saved preprocessing report to", PREPROCESSING_REPORT_PATH)


