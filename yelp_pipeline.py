"""
CSCE 478/878 Yelp star prediction: shared preprocessing + Ridge + Random Forest.
Usage:  python yelp_pipeline.py path/to/yelp_academic_dataset_business.json
"""
import ast, json, sys
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PATH = sys.argv[1] if len(sys.argv) > 1 else "yelp_academic_dataset_business.json"
TOP_CATS = 100      # keep the 100 most common categories
MIN_ATTR_FRAC = 0.05  # drop attribute columns present in <5% of businesses
SEED = 42

# ---------- 1. Load ----------
df = pd.read_json(PATH, lines=True)
print("raw rows:", len(df))

# ---------- 2. Duplicates and invalid records ----------
df = df.drop_duplicates(subset="business_id")
df["city"] = df["city"].str.strip().str.title()
df["state"] = df["state"].str.strip().str.upper()
df["postal_code"] = df["postal_code"].astype(str).str.strip()
bad_loc = df["latitude"].isna() | df["longitude"].isna() | (df["state"].str.len() != 2)
print("invalid location rows dropped:", int(bad_loc.sum()))
df = df[~bad_loc]
# same name + same address = likely duplicate listing
dup = df.duplicated(subset=["name", "address", "postal_code"], keep="first")
print("name+address duplicates dropped:", int(dup.sum()))
df = df[~dup].reset_index(drop=True)

# ---------- 3. Attributes (nested, stored as strings) ----------
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
        if isinstance(v, dict):                 # e.g. Ambience, BusinessParking
            for k2, v2 in v.items():
                out[f"attr_{k}_{k2}"] = parse(v2)
        else:
            out[f"attr_{k}"] = v
    return out

attrs = pd.DataFrame([flatten_attrs(a) for a in df["attributes"]])
attrs = attrs.loc[:, attrs.notna().mean() >= MIN_ATTR_FRAC]

num_attr, cat_attr = [], []
for c in attrs.columns:
    vals = set(attrs[c].dropna().unique())
    if vals <= {True, False}:
        attrs[c] = attrs[c].map({True: 1.0, False: 0.0})
        num_attr.append(c)
    elif c == "attr_RestaurantsPriceRange2":
        attrs[c] = pd.to_numeric(attrs[c], errors="coerce")
        num_attr.append(c)
    else:
        attrs[c] = attrs[c].astype(str).str.strip("'").str.lower()
        cat_attr.append(c)

# Missing attributes: flag + impute (missing usually means "not listed", not random)
for c in num_attr:
    attrs[c + "_missing"] = attrs[c].isna().astype(float)
    attrs[c] = attrs[c].fillna(0.0)           # median is fine too; 0 = "not listed"
attrs_cat = pd.get_dummies(attrs[cat_attr].replace("nan", "missing").fillna("missing"),
                           prefix=cat_attr, dtype=float)
attrs = pd.concat([attrs[[c for c in attrs.columns if c not in cat_attr]], attrs_cat], axis=1)

# ---------- 4. Categories (multi-hot, top N) ----------
cats = df["categories"].fillna("").str.split(r",\s*")
cats = cats.apply(lambda xs: [x for x in xs if x])
top = pd.Series([c for xs in cats for c in xs]).value_counts().head(TOP_CATS).index
cat_mh = pd.DataFrame({f"cat_{c}": cats.apply(lambda xs, c=c: float(c in xs)) for c in top})
cat_mh["n_categories"] = cats.str.len().astype(float)

# ---------- 5. Hours ----------
def hours_feats(h):
    if not isinstance(h, dict) or not h:
        return {"days_open": 0.0, "hours_per_week": 0.0, "hours_missing": 1.0}
    total = 0.0
    for span in h.values():
        try:
            a, b = span.split("-")
            ha, ma = map(int, a.split(":")); hb, mb = map(int, b.split(":"))
            d = (hb + mb / 60) - (ha + ma / 60)
            total += d if d > 0 else d + 24   # closes after midnight, or 0:0-0:0 = 24h
        except Exception:
            pass
    return {"days_open": float(len(h)), "hours_per_week": total, "hours_missing": 0.0}

hours = pd.DataFrame([hours_feats(h) for h in df["hours"]])

# ---------- 6. Base numeric + state ----------
base = pd.DataFrame({
    "log_review_count": np.log1p(df["review_count"]),
    "latitude": df["latitude"], "longitude": df["longitude"],
    "is_open": df["is_open"].astype(float),
})
states = pd.get_dummies(df["state"], prefix="state", dtype=float)
states = states.loc[:, states.sum() >= 50]       # rare states fold into "all zeros"

X = pd.concat([base, states, hours, cat_mh, attrs], axis=1)
y = df["stars"].astype(float).values
X.columns = [str(c) for c in X.columns]
print("feature matrix:", X.shape)

# ---------- 7. Outliers (flag, don't drop) ----------
low = (df["stars"] <= 1.5)
print(f"stars<=1.5: {int(low.sum())}, of which review_count<10: "
      f"{int((low & (df['review_count'] < 10)).sum())}")
print("stars by review_count bucket:\n",
      df.groupby(pd.cut(df["review_count"], [0, 10, 50, 200, 1e9]))["stars"].agg(["mean", "std", "count"]))

# ---------- 8. Split ----------
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
scaler = StandardScaler().fit(X_tr)              # fit on train only
X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)
cv = KFold(5, shuffle=True, random_state=SEED)

def report(name, p):
    r = {"model": name, "RMSE": mean_squared_error(y_te, p) ** 0.5,
         "MAE": mean_absolute_error(y_te, p), "R2": r2_score(y_te, p)}
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
    return r

results = [report("Baseline (train mean)", np.full(len(y_te), y_tr.mean()))]

# ---------- 9. Model 1: Ridge ----------
ridge = GridSearchCV(Ridge(), {"alpha": np.logspace(-3, 3, 13)},
                     cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
ridge.fit(X_tr_s, y_tr)
print("best ridge alpha:", ridge.best_params_)
results.append(report("Ridge", ridge.predict(X_te_s)))

# ---------- 10. Model 2: Random Forest (no scaling needed) ----------
rf = RandomizedSearchCV(
    RandomForestRegressor(random_state=SEED, n_jobs=-1),
    {"n_estimators": [200, 400],
     "max_depth": [None, 10, 20, 30],
     "min_samples_leaf": [1, 2, 5, 10],
     "max_features": ["sqrt", 0.3, 0.5]},
    n_iter=12, cv=cv, scoring="neg_root_mean_squared_error",
    random_state=SEED, n_jobs=-1)
rf.fit(X_tr, y_tr)
print("best RF params:", rf.best_params_)
results.append(report("Random Forest", rf.predict(X_te)))

# ---------- 11. Compare + interpret ----------
res = pd.DataFrame(results)
print("\n", res.round(4).to_string(index=False))
res.to_csv("results.csv", index=False)

imp = pd.Series(rf.best_estimator_.feature_importances_, index=X.columns).sort_values(ascending=False)
coef = pd.Series(ridge.best_estimator_.coef_, index=X.columns).sort_values(key=abs, ascending=False)
print("\nTop 15 RF importances:\n", imp.head(15).round(4))
print("\nTop 15 |Ridge coefficients|:\n", coef.head(15).round(4))
imp.to_csv("rf_importances.csv"); coef.to_csv("ridge_coefficients.csv")
