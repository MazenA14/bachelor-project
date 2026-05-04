import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import warnings

warnings.filterwarnings("ignore")

print("Initializing crops multi-stage evaluation...\n")

PROCESSED_DIR = "../data-crops/02_processed/"
FINAL_DIR = "../data-crops/03_final/"

df_master = pd.read_csv(
    os.path.join(PROCESSED_DIR, "01_master_crops_dataset.csv"),
    index_col="Date",
    parse_dates=["Date"],
)
df_a = pd.read_csv(
    os.path.join(FINAL_DIR, "01a_engineered_differencing_crops_dataset.csv"),
    index_col="Date",
    parse_dates=["Date"],
)
df_b = pd.read_csv(
    os.path.join(FINAL_DIR, "01b_engineered_detrending_crops_dataset.csv"),
    index_col="Date",
    parse_dates=["Date"],
)

TRAIN_END = "2023-12-31"
VAL_START = "2024-01-01"
VAL_END = "2024-12-31"
TEST_START = "2025-01-01"

train_a = df_a[:TRAIN_END]
val_a = df_a[VAL_START:VAL_END]
test_a = df_a[TEST_START:]

train_b = df_b[:TRAIN_END]
val_b = df_b[VAL_START:VAL_END]
test_b = df_b[TEST_START:]

drop_cols_a = [
    "Wheat_Close",
    "Corn_Close",
    "Sugar_Close",
    "Brent_Crude_Close",
    "DXY_Close",
    "EGP_USD_Close",
    "Wheat_Close_LogReturn",
    "Egypt_Inflation_YoY",
    "CBE_Interest_Rate",
]

drop_cols_b = [
    "Wheat_Close",
    "Corn_Close",
    "Sugar_Close",
    "Brent_Crude_Close",
    "DXY_Close",
    "EGP_USD_Close",
    "Wheat_Close_Trend",
    "Wheat_Close_Residual",
    "Corn_Close_Trend",
    "Sugar_Close_Trend",
    "Brent_Crude_Close_Trend",
    "DXY_Close_Trend",
    "EGP_USD_Close_Trend",
]


def evaluate_stage_arch_a(model, X):
    """Log-return model: reconstruct Wheat close from predicted log returns."""
    preds = model.predict(X)
    yesterday = df_master.loc[X.index, "Wheat_Close"].shift(1)
    first_idx = df_master.index.get_loc(X.index[0])
    yesterday.iloc[0] = df_master.iloc[first_idx - 1]["Wheat_Close"]

    price_pred = yesterday * np.exp(preds)
    actual = df_master.loc[X.index, "Wheat_Close"]

    rmse = root_mean_squared_error(actual, price_pred)
    mape = mean_absolute_percentage_error(actual, price_pred)
    mae = mean_absolute_error(actual, price_pred)
    mse = mean_squared_error(actual, price_pred)
    r2 = r2_score(actual, price_pred)
    return rmse, mape, mae, mse, r2


def evaluate_stage_arch_b(model, X, df_feat):
    """Detrending model: trend + predicted residual."""
    preds = model.predict(X)
    trend = df_feat.loc[X.index, "Wheat_Close_Trend"]
    price_pred = trend + preds
    actual = df_master.loc[X.index, "Wheat_Close"]

    rmse = root_mean_squared_error(actual, price_pred)
    mape = mean_absolute_percentage_error(actual, price_pred)
    mae = mean_absolute_error(actual, price_pred)
    mse = mean_squared_error(actual, price_pred)
    r2 = r2_score(actual, price_pred)
    return rmse, mape, mae, mse, r2


def print_metrics_arch_a(model, X_train, X_val, X_test, title):
    print(f"=== {title} ===")
    for name, X in [
        ("Training", X_train),
        ("Validation", X_val),
        ("Testing", X_test),
    ]:
        rmse, mape, mae, mse, r2 = evaluate_stage_arch_a(model, X)
        print(
            f"{name:10s} - RMSE: ${rmse:.2f} | MAPE: {mape:.4f} | "
            f"MAE: ${mae:.2f} | MSE: {mse:.2f} | R2: {r2:.4f}"
        )
    print()


def print_metrics_arch_b(model, X_train, X_val, X_test, title, df_feat):
    print(f"=== {title} ===")
    for name, X in [
        ("Training", X_train),
        ("Validation", X_val),
        ("Testing", X_test),
    ]:
        rmse, mape, mae, mse, r2 = evaluate_stage_arch_b(model, X, df_feat)
        print(
            f"{name:10s} - RMSE: ${rmse:.2f} | MAPE: {mape:.4f} | "
            f"MAE: ${mae:.2f} | MSE: {mse:.2f} | R2: {r2:.4f}"
        )
    print()


# --- Architecture A: differencing / log returns ---
X_train_a = train_a.drop(columns=drop_cols_a)
y_train_a = train_a["Wheat_Close_LogReturn"]
X_val_a = val_a.drop(columns=drop_cols_a)
y_val_a = val_a["Wheat_Close_LogReturn"]
X_test_a = test_a.drop(columns=drop_cols_a)

xgb_a = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=50,
)
xgb_a.fit(X_train_a, y_train_a, eval_set=[(X_val_a, y_val_a)], verbose=False)

print_metrics_arch_a(
    xgb_a,
    X_train_a,
    X_val_a,
    X_test_a,
    "XGBoost Architecture A (Log Returns / Differencing)",
)

# --- Architecture B: detrending / residuals ---
X_train_b = train_b.drop(columns=drop_cols_b)
y_train_b = train_b["Wheat_Close_Residual"]
X_val_b = val_b.drop(columns=drop_cols_b)
y_val_b = val_b["Wheat_Close_Residual"]
X_test_b = test_b.drop(columns=drop_cols_b)

xgb_b = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=50,
)
xgb_b.fit(X_train_b, y_train_b, eval_set=[(X_val_b, y_val_b)], verbose=False)

print_metrics_arch_b(
    xgb_b,
    X_train_b,
    X_val_b,
    X_test_b,
    "XGBoost Architecture B (Detrending / Residuals)",
    df_b,
)
