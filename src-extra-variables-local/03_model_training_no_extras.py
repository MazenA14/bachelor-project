import os
import pandas as pd
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION & DIRECTORY SETUP ---
PROCESSED_DIR = '../data-extra-variables-local/02_processed/'
FINAL_DIR = '../data-extra-variables-local/03_final/'
MODELS_DIR = '../models-extra-variables-local/'

# Ensure the models directory exists so the script doesn't crash when saving
os.makedirs(MODELS_DIR, exist_ok=True)

print("Initiating Phase 3: Model Training (No Extras)...\n")

# --- 2. LOAD ALL THREE DATASETS ---
df_master = pd.read_csv(os.path.join(PROCESSED_DIR, '01_master_metals_dataset.csv'), index_col='Date', parse_dates=['Date'])
df_a = pd.read_csv(os.path.join(FINAL_DIR, '01a_differencing_metals_dataset.csv'), index_col='Date', parse_dates=['Date'])
df_b = pd.read_csv(os.path.join(FINAL_DIR, '01b_detrending_metals_dataset.csv'), index_col='Date', parse_dates=['Date'])

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
# These datasets have NO lag, rolling, or cross-feature columns.
# We only drop raw prices + the stationarity-specific columns.
drop_cols_a = ['Gold_Close', 'Silver_Close', 'DXY_Close', 'SP500_Close', 'VIX_Close', 'EGP_USD_Close', 'Local_Gold_24k_EGP',
               'Gold_Close_LogReturn', 'Silver_Close_LogReturn', 'DXY_Close_LogReturn',
               'SP500_Close_LogReturn', 'VIX_Close_LogReturn', 'EGP_USD_Close_LogReturn', 'Local_Gold_24k_EGP_LogReturn']

X_train_a = train_a.drop(columns=drop_cols_a)
y_train_a = train_a['Local_Gold_24k_EGP_LogReturn']
X_val_a = val_a.drop(columns=drop_cols_a)
y_val_a = val_a['Local_Gold_24k_EGP_LogReturn']

drop_cols_b = ['Gold_Close', 'Silver_Close', 'DXY_Close', 'SP500_Close', 'VIX_Close', 'EGP_USD_Close', 'Local_Gold_24k_EGP',
               'Gold_Close_Trend', 'Gold_Close_Residual', 'Silver_Close_Trend', 'Silver_Close_Residual',
               'DXY_Close_Trend', 'DXY_Close_Residual', 'SP500_Close_Trend', 'SP500_Close_Residual',
               'VIX_Close_Trend', 'VIX_Close_Residual', 'EGP_USD_Close_Trend', 'EGP_USD_Close_Residual',
               'Local_Gold_24k_EGP_Trend', 'Local_Gold_24k_EGP_Residual']

X_train_b = train_b.drop(columns=drop_cols_b)
y_train_b = train_b['Local_Gold_24k_EGP_Residual']
X_val_b = val_b.drop(columns=drop_cols_b)
y_val_b = val_b['Local_Gold_24k_EGP_Residual']

print(f"Features for XGBoost A: {list(X_train_a.columns)}")
print(f"Features for XGBoost B: {list(X_train_b.columns)}\n")

# --- 5. MODEL 1: ARIMA BASELINE ---
print("Training Model 1: ARIMA Baseline...")
arima_train = df_master['Local_Gold_24k_EGP'][:VAL_END] 
arima_model = ARIMA(arima_train, order=(5, 1, 0))
arima_fitted = arima_model.fit()

# SAVE ARIMA MODEL TO DISK
arima_save_path = os.path.join(MODELS_DIR, 'arima_baseline_no_extras.pkl')
arima_fitted.save(arima_save_path)
print(f"ARIMA Base saved to: {arima_save_path}")

# --- 6. MODEL 2: XGBOOST ARCHITECTURE A ---
print("\nTraining Model 2: XGBoost (Log Returns, No Extras)...")
xgb_a = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=5, random_state=42, early_stopping_rounds=50)
xgb_a.fit(X_train_a, y_train_a, eval_set=[(X_val_a, y_val_a)], verbose=False)

# SAVE XGBOOST A TO DISK (Using native JSON format)
xgb_a_save_path = os.path.join(MODELS_DIR, 'xgboost_a_no_extras.json')
xgb_a.save_model(xgb_a_save_path)
print(f"XGBoost A saved to: {xgb_a_save_path}")

# --- 7. MODEL 3: XGBOOST ARCHITECTURE B ---
print("\nTraining Model 3: XGBoost (Detrended Residuals, No Extras)...")
xgb_b = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=5, random_state=42, early_stopping_rounds=50)
xgb_b.fit(X_train_b, y_train_b, eval_set=[(X_val_b, y_val_b)], verbose=False)

# SAVE XGBOOST B TO DISK
xgb_b_save_path = os.path.join(MODELS_DIR, 'xgboost_b_no_extras.json')
xgb_b.save_model(xgb_b_save_path)
print(f"XGBoost B saved to: {xgb_b_save_path}")

print("\n--- TRAINING PHASE (NO EXTRAS) COMPLETE ---")

