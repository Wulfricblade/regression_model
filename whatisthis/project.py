import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------- 1. Load ----------
df = pd.read_csv("Air_Pollution_Weather_Prediction_Dataset.xlsx")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values("Date").reset_index(drop=True)

features = ["Temp_Mean", "Temp_Max", "Temp_Min", "Rainfall_mm"]
target = "PM2_5_f"
X, y = df[features], df[target]

# ---------- 2. Explore ----------
plt.figure(figsize=(7, 5))
sns.heatmap(df[features + [target]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation: Weather vs PM2.5")
plt.tight_layout(); plt.savefig("1_correlation.png", dpi=150); plt.show()

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, f in zip(axes, features):
    sns.regplot(x=df[f], y=y, ax=ax, scatter_kws={"alpha": 0.5})
    ax.set_title(f"{f} vs PM2.5")
plt.tight_layout(); plt.savefig("2_scatter_relationships.png", dpi=150); plt.show()

plt.figure(figsize=(10, 4))
plt.plot(df["Date"], y)
plt.title("PM2.5 over time (2024)"); plt.xlabel("Date"); plt.ylabel("PM2.5")
plt.tight_layout(); plt.savefig("3_pm25_timeline.png", dpi=150); plt.show()

# ---------- 3. Split and scale ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------- 4. Train ----------
lr = LinearRegression().fit(X_train_s, y_train)
rf = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# ---------- 5. Evaluate ----------
pred_lr = lr.predict(X_test_s)
pred_rf = rf.predict(X_test)

def metrics(y_true, y_pred):
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred)}

results = pd.DataFrame({"Linear Regression": metrics(y_test, pred_lr),
                        "Random Forest": metrics(y_test, pred_rf)}).T.round(3)
print("\n=== MODEL RESULTS ===")
print(results)

cv = cross_val_score(rf, X, y, cv=5, scoring="r2")
print("\nRandom Forest 5-fold CV R2:", cv.round(3), "mean =", cv.mean().round(3))

# ---------- 6. Result graphs ----------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred_rf, alpha=0.6)
lims = [y.min(), y.max()]
plt.plot(lims, lims, "r--")
plt.xlabel("Actual PM2.5"); plt.ylabel("Predicted PM2.5")
plt.title("Actual vs Predicted (Random Forest)")
plt.tight_layout(); plt.savefig("4_actual_vs_predicted.png", dpi=150); plt.show()

order = y_test.sort_index().index
plt.figure(figsize=(10, 4))
plt.plot(df.loc[order, "Date"], y_test.loc[order].values, label="Actual", marker="o")
plt.plot(df.loc[order, "Date"], pd.Series(pred_rf, index=y_test.index).loc[order].values,
         label="Predicted", marker="x")
plt.legend(); plt.title("Prediction Trend on Test Data"); plt.xticks(rotation=30)
plt.tight_layout(); plt.savefig("5_prediction_trend.png", dpi=150); plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(y_test - pred_rf, kde=True)
plt.title("Error Distribution (Actual - Predicted)")
plt.tight_layout(); plt.savefig("6_residuals.png", dpi=150); plt.show()

plt.figure(figsize=(6, 4))
pd.Series(rf.feature_importances_, index=features).sort_values().plot(kind="barh")
plt.title("Which weather variable matters most?")
plt.tight_layout(); plt.savefig("7_feature_importance.png", dpi=150); plt.show()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, m in zip(axes, ["MAE", "RMSE", "R2"]):
    results[m].plot(kind="bar", ax=ax, color=["#4C72B0", "#55A868"])
    ax.set_title(m); ax.tick_params(axis="x", rotation=15)
plt.tight_layout(); plt.savefig("8_model_comparison.png", dpi=150); plt.show()

# ---------- 7. Predict for new weather ----------
new = pd.DataFrame([{"Temp_Mean": 27, "Temp_Max": 32, "Temp_Min": 22, "Rainfall_mm": 2.0}])
print("\nPredicted PM2.5 (Random Forest):", round(rf.predict(new[features])[0], 2))
print("Predicted PM2.5 (Linear):", round(lr.predict(scaler.transform(new[features]))[0], 2))