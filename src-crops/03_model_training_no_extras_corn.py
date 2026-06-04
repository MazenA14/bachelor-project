# =============================================================================
# PIPELINE IDENTITY
#   Extra Features      : NO (Only raw prices + stationarity columns — no lags/rolling/cross)
#   Target Commodity    : CORN
#   Architecture A      : DIFFERENCING  — reads 01a_differencing_crops_dataset.csv
#                         Target: Corn_Close_LogReturn
#   Architecture B      : DETRENDING    — reads 01b_detrending_crops_dataset.csv
#                         Target: Corn_Close_Residual
#   Saves models to     : ../models-crops/  (xgboost_a_no_extras_corn.json, xgboost_b_no_extras_corn.json,
#                                             arima_baseline_no_extras_corn.pkl)
# =============================================================================
import os
import pandas as pd
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION & DIRECTORY SETUP ---
PROCESSED_DIR = '../data-crops/02_processed/'
FINAL_DIR = '../data-crops/03_final/'
MODELS_DIR = '../models-crops/'

os.makedirs(MODELS_DIR, exist_ok=True)

print("Initiating Phase 3: Model Training Protocol — Corn (No Extras)...\n")

# --- 2. LOAD ALL THREE DATASETS ---
df_master = pd.read_csv(os.path.join(PROCESSED_DIR, '01_master_crops_dataset.csv'), index_col='Date', parse_dates=['Date'])
df_a = pd.read_csv(os.path.join(FINAL_DIR, '01a_differencing_crops_dataset.csv'), index_col='Date', parse_dates=['Date'])
df_b = pd.read_csv(os.path.join(FINAL_DIR, '01b_detrending_crops_dataset.csv'), index_col='Date', parse_dates=['Date'])

# --- 3. THE CHRONOLOGICAL TIMELINE SPLIT ---
TRAIN_END = '2023-12-31'
VAL_START = '2024-01-01'
VAL_END = '2024-12-31'
TEST_START = '2025-01-01'

def split_data(df):
    train = df[:TRAIN_END]
    val = df[VAL_START:VAL_END]
    test = df[TEST_START:]
    return train, val, test

train_a, val_a, test_a = split_data(df_a)
train_b, val_b, test_b = split_data(df_b)

# --- 4. FEATURE SELECTION (Isolating X and Y) ---
drop_cols_a = [
    'Wheat_Close', 'Corn_Close', 'Sugar_Close', 'Brent_Crude_Close', 'DXY_Close', 'EGP_USD_Close',
    'Corn_Close_LogReturn',
    'Egypt_Inflation_YoY', 'CBE_Interest_Rate'
]

X_train_a = train_a.drop(columns=drop_cols_a)
y_train_a = train_a['Corn_Close_LogReturn']
X_val_a = val_a.drop(columns=drop_cols_a)
y_val_a = val_a['Corn_Close_LogReturn']

drop_cols_b = [
    'Wheat_Close', 'Corn_Close', 'Sugar_Close', 'Brent_Crude_Close', 'DXY_Close', 'EGP_USD_Close',
    'Wheat_Close_Trend',
    'Corn_Close_Trend', 'Corn_Close_Residual',
    'Sugar_Close_Trend',
    'Brent_Crude_Close_Trend',
    'DXY_Close_Trend',
    'EGP_USD_Close_Trend'
]

X_train_b = train_b.drop(columns=drop_cols_b)
y_train_b = train_b['Corn_Close_Residual']
X_val_b = val_b.drop(columns=drop_cols_b)
y_val_b = val_b['Corn_Close_Residual']

print(f"Features for XGBoost A: {list(X_train_a.columns)}")
print(f"Features for XGBoost B: {list(X_train_b.columns)}\n")

# --- 5. MODEL 1: ARIMA BASELINE ---
print("Training Model 1: ARIMA Baseline (Corn)...")
arima_train = df_master['Corn_Close'][:VAL_END]
arima_model = ARIMA(arima_train, order=(5, 1, 0))
arima_fitted = arima_model.fit()

arima_save_path = os.path.join(MODELS_DIR, 'arima_baseline_no_extras_corn.pkl')
arima_fitted.save(arima_save_path)
print(f"ARIMA Base saved to: {arima_save_path}")

# --- 6. MODEL 2: XGBOOST ARCHITECTURE A ---
print("\nTraining Model 2: XGBoost (Log Returns, No Extras, Corn)...")
xgb_a = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=50
)
xgb_a.fit(X_train_a, y_train_a, eval_set=[(X_val_a, y_val_a)], verbose=False)

xgb_a_save_path = os.path.join(MODELS_DIR, 'xgboost_a_no_extras_corn.json')
xgb_a.save_model(xgb_a_save_path)
print(f"XGBoost A saved to: {xgb_a_save_path}")

# --- 7. MODEL 3: XGBOOST ARCHITECTURE B ---
print("\nTraining Model 3: XGBoost (Detrended Residuals, No Extras, Corn)...")
xgb_b = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=50
)
xgb_b.fit(X_train_b, y_train_b, eval_set=[(X_val_b, y_val_b)], verbose=False)

xgb_b_save_path = os.path.join(MODELS_DIR, 'xgboost_b_no_extras_corn.json')
xgb_b.save_model(xgb_b_save_path)
print(f"XGBoost B saved to: {xgb_b_save_path}")

print("\n--- TRAINING PHASE (NO EXTRAS, CORN) COMPLETE ---")
