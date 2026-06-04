# ==========================================================
# Mining Process Flotation Plant - Machine Learning Pipeline
# ==========================================================

# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import xgboost as xgb


# =========================
# 2. EVALUATION FUNCTION
# =========================
def evaluate(y_true, y_pred):
    """
    Returns MAE, RMSE, and R² score.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2


# =========================
# 3. LOAD DATASET
# =========================
file_path = r"C:\Users\Angel\OneDrive\Desktop\comprog_lab\BRISENIO_ML\MiningProcess_Flotation_Plant_Database.csv"

df = pd.read_csv(
    file_path,
    sep=",",
    low_memory=False
)

print("Dataset Loaded Successfully")
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())


# =========================
# 4. DATA CLEANING
# =========================
# Convert all columns to numeric
df = df.apply(pd.to_numeric, errors="coerce")

# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

print("\nTotal Missing Values:", df.isna().sum().sum())


# =========================
# 5. FEATURE / TARGET SPLIT
# =========================
target_col = df.columns[-1]

# Remove rows with missing target values
df = df.dropna(subset=[target_col])

X = df.drop(columns=[target_col])
y = df[target_col]

print("\nTarget Variable:", target_col)
print("Feature Matrix Shape:", X.shape)


# =========================
# 6. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# =========================
# 7. HANDLE MISSING VALUES
# =========================
imputer = SimpleImputer(strategy="mean")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


# =========================
# 8. INITIALIZE MODELS
# =========================
lr = LinearRegression()

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)


# =========================
# 9. TRAIN MODELS
# =========================
print("\nTraining Models...")

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

print("Training Complete.")


# =========================
# 10. MAKE PREDICTIONS
# =========================
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)
xgb_pred = xgb_model.predict(X_test)


# =========================
# 11. MODEL EVALUATION
# =========================
results = pd.DataFrame([
    ["Linear Regression", *evaluate(y_test, lr_pred)],
    ["Random Forest", *evaluate(y_test, rf_pred)],
    ["XGBoost", *evaluate(y_test, xgb_pred)]
], columns=["Model", "MAE", "RMSE", "R2"])

print("\n==========================")
print("MODEL PERFORMANCE RESULTS")
print("==========================")
print(results)


# =========================
# 12. FEATURE IMPORTANCE
# =========================
feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf.feature_importances_
})

feat_df = feat_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n================")
print("TOP 10 FEATURES")
print("================")
print(feat_df.head(10))


# =========================
# 13. VISUALIZATION
# =========================
plt.figure(figsize=(6, 6))

plt.scatter(
    y_test,
    xgb_pred,
    alpha=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted (XGBoost)")

plt.tight_layout()
plt.show()