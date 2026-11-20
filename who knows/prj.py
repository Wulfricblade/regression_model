import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------- SETTINGS: change only these ----------
FILE   = "city_weather_pollution_2024.csv"
target = "pm2_5"               # set to "us_aqi" to predict AQI instead
label  = "PM2.5" if target == "pm2_5" else "AQI"
prefix = label.replace(".", "") # file names: PM25_... or AQI_...

# Weather features matching the generated dataset & project scope
features = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "precipitation"
]

# ---------- 1. Load ----------
df = pd.read_csv(FILE)
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").dropna().reset_index(drop=True)
print("Data Preview:\n", df.head())
print("\nShape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

year = df["time"].dt.year.iloc[0]
X, y = df[features], df[target]

# ---------- 2. Explore ----------
plt.figure(figsize=(8, 6))
sns.heatmap(df[features + [target]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title(f"Correlation: Weather vs {label}")
plt.tight_layout()
plt.savefig(f"{prefix}_1_correlation.png", dpi=150)
plt.show()

# Dynamic subplot grid for the 5 weather features
fig, axes = plt.subplots(1, len(features), figsize=(20, 4))
for ax, f in zip(axes, features):
    sns.regplot(x=df[f], y=y, ax=ax, scatter_kws={"alpha": 0.3, "s": 10}, line_kws={"color": "red"})
    ax.set_title(f"{f} vs {label}")
plt.tight_layout()
plt.savefig(f"{prefix}_2_scatter_relationships.png", dpi=150)
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(df["time"], y, linewidth=0.8)
plt.title(f"{label} over time ({year})")
plt.xlabel("Date")
plt.ylabel(label)
plt.tight_layout()
plt.savefig(f"{prefix}_3_timeline.png", dpi=150)
plt.show()

# ---------- 3. Split and scale ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------- 4. Train ----------
lr = LinearRegression().fit(X_train_s, y_train)
rf = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1).fit(X_train, y_train)

# ---------- 5. Evaluate ----------
pred_lr = lr.predict(X_test_s)
pred_rf = rf.predict(X_test)

def metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted)
    }

results = pd.DataFrame({
    "Linear Regression": metrics(y_test, pred_lr),
    "Random Forest": metrics(y_test, pred_rf)
}).T.round(3)

print(f"\n=== MODEL RESULTS ({label}) ===\n", results)

cv = cross_val_score(rf, X, y, cv=KFold(5, shuffle=True, random_state=42), scoring="r2", n_jobs=-1)
print("\nRandom Forest 5-fold CV R2:", cv.round(3), "mean =", cv.mean().round(3))

# ---------- 6. Result graphs ----------
# 6a. Actual vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred_rf, alpha=0.4, s=12)
lims = [min(y_test.min(), pred_rf.min()), max(y_test.max(), pred_rf.max())]
plt.plot(lims, lims, "r--", label="Ideal Fit")
plt.xlabel(f"Actual {label}")
plt.ylabel(f"Predicted {label}")
plt.title(f"Actual vs Predicted (Random Forest)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{prefix}_4_actual_vs_predicted.png", dpi=150)
plt.show()

# 6b. Prediction Trend on a test sample slice for clean readability
order = y_test.sort_index().index[:200]  # First 200 chronological points in test set
plt.figure(figsize=(12, 4))
plt.plot(df.loc[order, "time"], y_test.loc[order].values, label="Actual", marker="o", ms=2.5)
plt.plot(df.loc[order, "time"], pd.Series(pred_rf, index=y_test.index).loc[order].values,
         label="Predicted", marker="x", ms=2.5)
plt.legend()
plt.title("Prediction Trend on Sample Test Data")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(f"{prefix}_5_prediction_trend.png", dpi=150)
plt.show()

# 6c. Residual Distribution
plt.figure(figsize=(6, 4))
sns.histplot(y_test - pred_rf, kde=True, color="teal")
plt.xlabel(f"Error ({label})")
plt.title("Residual Distribution (Actual - Predicted)")
plt.tight_layout()
plt.savefig(f"{prefix}_6_residuals.png", dpi=150)