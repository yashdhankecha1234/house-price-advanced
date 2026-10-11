import json
import numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from features import load_data, add_features, RAW_COLUMNS

# ---------- 1. Data ----------
df = load_data()
X, y, groups = df[RAW_COLUMNS], df["price"], df["id"]

# Same ghar (id) train aur test dono me nahi aayega
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr, te = next(gss.split(X, y, groups=groups))
X_train, X_test, y_train, y_test = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]

# ---------- 2. Pipeline builder ----------
def make_pipeline(model):
    pipe = Pipeline([("features", FunctionTransformer(add_features)), ("model", model)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)

# ---------- 3. Models ----------
models = {
    "Linear Regression": Pipeline([("scale", StandardScaler()), ("lr", LinearRegression())]),
    "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42),
    "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, random_state=42),
}
try:                                    # optional: agar installed ho to
    from xgboost import XGBRegressor
    models["XGBoost"] = XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, random_state=42)
except ImportError:
    print("xgboost not installed, skipping")
try:
    from lightgbm import LGBMRegressor
    models["LightGBM"] = LGBMRegressor(n_estimators=600, learning_rate=0.05, random_state=42, verbose=-1)
except ImportError:
    print("lightgbm not installed, skipping")

# ---------- 4. Compare ----------
rows, fitted = [], {}
for name, m in models.items():
    est = make_pipeline(m).fit(X_train, y_train)
    p = est.predict(X_test)
    rows.append({"Model": name,
                 "R2": round(r2_score(y_test, p), 3),
                 "MAE": round(mean_absolute_error(y_test, p)),
                 "RMSE": round(float(np.sqrt(mean_squared_error(y_test, p))))})
    fitted[name] = est
res = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
print("\n", res.to_string(index=False))
res.to_csv("results.csv", index=False)

best_name = res.loc[0, "Model"]
best = fitted[best_name]
pred = best.predict(X_test)
print("\nBest model:", best_name)

# ---------- 5. Cross validation (best model) ----------
cv = cross_val_score(make_pipeline(models[best_name]), X_train, y_train,
                     cv=GroupKFold(n_splits=5), groups=groups.iloc[tr], scoring="r2")
print("CV R2:", cv.round(3), "| mean:", round(cv.mean(), 3))

# ---------- 6. Prediction interval (80%) ----------
def quantile_model(q):
    return make_pipeline(HistGradientBoostingRegressor(
        loss="quantile", quantile=q, max_iter=300, random_state=42)).fit(X_train, y_train)
low, high = quantile_model(0.1), quantile_model(0.9)
lo, hi = low.predict(X_test), high.predict(X_test)
coverage = float(((y_test >= lo) & (y_test <= hi)).mean())
print("Interval coverage (target 0.80):", round(coverage, 3))

# ---------- 7. Plots ----------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred, s=4, alpha=0.3, color="#2E6F5E")
lim = [0, 3_000_000]
plt.plot(lim, lim, "r--"); plt.xlim(lim); plt.ylim(lim)
plt.xlabel("Actual price"); plt.ylabel("Predicted price"); plt.title(f"Actual vs Predicted ({best_name})")
plt.tight_layout(); plt.savefig("images/actual_vs_predicted.png", dpi=130); plt.close()

idx = np.random.RandomState(0).choice(len(X_test), 1500, replace=False)
pi = permutation_importance(best, X_test.iloc[idx], y_test.iloc[idx],
                            scoring="r2", n_repeats=3, random_state=0, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=RAW_COLUMNS).sort_values().tail(12)
plt.figure(figsize=(7, 5)); imp.plot(kind="barh", color="#E8833A")
plt.title("Feature importance (permutation)"); plt.tight_layout()
plt.savefig("images/feature_importance.png", dpi=130); plt.close()
print("\nTop features:\n", imp.tail(5).round(3).iloc[::-1])

# ---------- 8. Save ----------
joblib.dump({"model": best, "low": low, "high": high, "columns": RAW_COLUMNS,
             "best_name": best_name}, "model.pkl", compress=3)
json.dump({"best_model": best_name, "r2": float(res.loc[0, "R2"]), "mae": int(res.loc[0, "MAE"]),
           "cv_r2_mean": round(float(cv.mean()), 3), "interval_coverage": round(coverage, 3)},
          open("metrics.json", "w"), indent=2)
print("\nSaved: model.pkl, metrics.json, results.csv, images/")