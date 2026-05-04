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

print("Initializing energy multi-stage evaluation...\n")

PROCESSED_DIR = "../data-energy/02_processed/"
FINAL_DIR = "../data-energy/03_final/"

df_master = pd.read_csv(
    os.path.join(PROCESSED_DIR, "01_master_energy_dataset.csv"),
    index_col="Date",
    parse_dates=["Date"],
)
df_a = pd.read_csv(
    os.path.join(FINAL_DIR, "01a_engineered_differencing_energy_dataset.csv"),
    index_col="Date",
    parse_dates=["Date"],
)
df_b = pd.read_csv(
    os.path.join(FINAL_DIR, "01b_engineered_detrending_energy_dataset.csv"),
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
    "Brent_Crude_Close",
    "Natural_Gas_Close",
    "DXY_Close",
    "VIX_Close",
    "SP500_Close",
    "EGP_USD_Close",
    "Brent_Crude_Close_LogReturn",
    "Natural_Gas_Close_LogReturn",
    "DXY_Close_LogReturn",
    "VIX_Close_LogReturn",
    "SP500_Close_LogReturn",
    "EGP_USD_Close_LogReturn",
    "US_10Yr_Yield_Diff",
    "Egypt_Inflation_YoY",
    "CBE_Interest_Rate",
]

drop_cols_b = [
    "Brent_Crude_Close",
    "Natural_Gas_Close",
    "DXY_Close",
    "VIX_Close",
    "SP500_Close",
    "EGP_USD_Close",
    "Brent_Crude_Close_Trend",
    "Brent_Crude_Close_Residual",
    "Natural_Gas_Close_Trend",
    "Natural_Gas_Close_Residual",
    "DXY_Close_Trend",
    "DXY_Close_Residual",
    "VIX_Close_Trend",
    "VIX_Close_Residual",
    "SP500_Close_Trend",
    "SP500_Close_Residual",
    "EGP_USD_Close_Trend",
    "EGP_USD_Close_Residual",
    "Egypt_Inflation_YoY",
    "CBE_Interest_Rate",
]

# Same feature matrices for both commodities; only the target y changes.
X_train_a = train_a.drop(columns=drop_cols_a)
X_val_a = val_a.drop(columns=drop_cols_a)
X_test_a = test_a.drop(columns=drop_cols_a)

X_train_b = train_b.drop(columns=drop_cols_b)
X_val_b = val_b.drop(columns=drop_cols_b)
X_test_b = test_b.drop(columns=drop_cols_b)

COMMODITIES = (
    {
        "label": "Brent crude",
        "close": "Brent_Crude_Close",
        "log_return": "Brent_Crude_Close_LogReturn",
        "trend": "Brent_Crude_Close_Trend",
        "residual": "Brent_Crude_Close_Residual",
    },
    {
        "label": "Natural gas",
        "close": "Natural_Gas_Close",
        "log_return": "Natural_Gas_Close_LogReturn",
        "trend": "Natural_Gas_Close_Trend",
        "residual": "Natural_Gas_Close_Residual",
    },
)


def evaluate_stage_arch_a(model, X, close_col):
    """Log-return model: reconstruct spot close from predicted log returns."""
    preds = model.predict(X)
    yesterday = df_master.loc[X.index, close_col].shift(1)
    first_idx = df_master.index.get_loc(X.index[0])
    yesterday.iloc[0] = df_master.iloc[first_idx - 1][close_col]

    price_pred = yesterday * np.exp(preds)
    actual = df_master.loc[X.index, close_col]

    rmse = root_mean_squared_error(actual, price_pred)
    mape = mean_absolute_percentage_error(actual, price_pred)
    mae = mean_absolute_error(actual, price_pred)
    mse = mean_squared_error(actual, price_pred)
    r2 = r2_score(actual, price_pred)
    return rmse, mape, mae, mse, r2


def evaluate_stage_arch_b(model, X, df_feat, close_col, trend_col):
    """Detrending model: trend + predicted residual."""
    preds = model.predict(X)
    trend = df_feat.loc[X.index, trend_col]
    price_pred = trend + preds
    actual = df_master.loc[X.index, close_col]

    rmse = root_mean_squared_error(actual, price_pred)
    mape = mean_absolute_percentage_error(actual, price_pred)
    mae = mean_absolute_error(actual, price_pred)
    mse = mean_squared_error(actual, price_pred)
    r2 = r2_score(actual, price_pred)
    return rmse, mape, mae, mse, r2


def print_metrics_arch_a(model, X_train, X_val, X_test, title, close_col):
    print(f"=== {title} ===")
    for name, X in [
        ("Training", X_train),
        ("Validation", X_val),
        ("Testing", X_test),
    ]:
        rmse, mape, mae, mse, r2 = evaluate_stage_arch_a(model, X, close_col)
        print(
            f"{name:10s} - RMSE: ${rmse:.2f} | MAPE: {mape:.4f} | "
            f"MAE: ${mae:.2f} | MSE: {mse:.2f} | R2: {r2:.4f}"
        )
    print()


def print_metrics_arch_b(model, X_train, X_val, X_test, title, df_feat, close_col, trend_col):
    print(f"=== {title} ===")
    for name, X in [
        ("Training", X_train),
        ("Validation", X_val),
        ("Testing", X_test),
    ]:
        rmse, mape, mae, mse, r2 = evaluate_stage_arch_b(
            model, X, df_feat, close_col, trend_col
        )
        print(
            f"{name:10s} - RMSE: ${rmse:.2f} | MAPE: {mape:.4f} | "
            f"MAE: ${mae:.2f} | MSE: {mse:.2f} | R2: {r2:.4f}"
        )
    print()


for c in COMMODITIES:
    print(f"{'=' * 60}\n{c['label']}\n{'=' * 60}\n")

    y_train_a = train_a[c["log_return"]]
    y_val_a = val_a[c["log_return"]]

    xgb_a = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        early_stopping_rounds=50,
    )
    xgb_a.fit(X_train_a, y_train_a, eval_set=[(X_val_a, y_val_a)], verbose=False)

    print_metrics_arch_a(
        xgb_a,
        X_train_a,
        X_val_a,
        X_test_a,
        f"XGBoost Architecture A (Log Returns / Differencing) — {c['label']}",
        c["close"],
    )

    y_train_b = train_b[c["residual"]]
    y_val_b = val_b[c["residual"]]

    xgb_b = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        early_stopping_rounds=50,
    )
    xgb_b.fit(X_train_b, y_train_b, eval_set=[(X_val_b, y_val_b)], verbose=False)

    print_metrics_arch_b(
        xgb_b,
        X_train_b,
        X_val_b,
        X_test_b,
        f"XGBoost Architecture B (Detrending / Residuals) — {c['label']}",
        df_b,
        c["close"],
        c["trend"],
    )
