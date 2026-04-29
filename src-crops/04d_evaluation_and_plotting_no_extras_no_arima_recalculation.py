# =============================================================================
# PIPELINE IDENTITY
#   Extra Features      : NO (No Extras — only raw prices + stationarity columns)
#   Architecture A      : DIFFERENCING  — reads 01a_differencing_crops_dataset.csv
#   Architecture B      : DETRENDING    — reads 01b_detrending_crops_dataset.csv
#   ARIMA               : LOADED from saved .pkl (fast, uses pre-trained model from Phase 3)
#   Models loaded from  : ../models-crops/  (xgboost_a_no_extras.json, xgboost_b_no_extras.json,
#                                             arima_baseline_no_extras.pkl)
#   Trained by          : 03_model_training_no_extras.py
# =============================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMAResults
from sklearn.metrics import root_mean_squared_error
from xgboost import plot_importance

# --- 1. CONFIGURATION & DATA LOADING ---
print("Initiating Phase 4: Model Evaluation (No Extras, Crops, No ARIMA Recalculation)...\n")
PROCESSED_DIR = '../data-crops/02_processed/'
FINAL_DIR = '../data-crops/03_final/'
MODELS_DIR = '../models-crops/'
TEST_START = '2025-01-01'

print("Loading Test Data...")
# Load the data so we can slice out the Test Sets
df_master = pd.read_csv(os.path.join(PROCESSED_DIR, '01_master_crops_dataset.csv'), index_col='Date', parse_dates=['Date'])
df_a = pd.read_csv(os.path.join(FINAL_DIR, '01a_differencing_crops_dataset.csv'), index_col='Date', parse_dates=['Date'])
df_b = pd.read_csv(os.path.join(FINAL_DIR, '01b_detrending_crops_dataset.csv'), index_col='Date', parse_dates=['Date'])

# Isolate the exact variables needed for the final exam
test_a = df_a[TEST_START:]
test_b = df_b[TEST_START:]

# These datasets have NO lag, rolling, or cross-feature columns.
# We only drop raw prices + the stationarity-specific columns.
drop_cols_a = [
    'Wheat_Close', 'Corn_Close', 'Sugar_Close', 'Brent_Crude_Close', 'DXY_Close', 'EGP_USD_Close',
    'Wheat_Close_LogReturn',
    'Egypt_Inflation_YoY', 'CBE_Interest_Rate'
]
X_test_a = test_a.drop(columns=drop_cols_a)

drop_cols_b = [
    'Wheat_Close', 'Corn_Close', 'Sugar_Close', 'Brent_Crude_Close', 'DXY_Close', 'EGP_USD_Close',
    'Wheat_Close_Trend', 'Wheat_Close_Residual',
    'Corn_Close_Trend',
    'Sugar_Close_Trend',
    'Brent_Crude_Close_Trend',
    'DXY_Close_Trend',
    'EGP_USD_Close_Trend'
]
X_test_b = test_b.drop(columns=drop_cols_b)

# --- 2. MODEL LOADING ---
print("Loading trained models from disk...")
# Load ARIMA Baseline
arima_fitted = ARIMAResults.load(os.path.join(MODELS_DIR, 'arima_baseline_no_extras.pkl'))

# Load XGBoost A (Initialize empty model, then load weights)
xgb_a = xgb.XGBRegressor()
xgb_a.load_model(os.path.join(MODELS_DIR, 'xgboost_a_no_extras.json'))

# Load XGBoost B
xgb_b = xgb.XGBRegressor()
xgb_b.load_model(os.path.join(MODELS_DIR, 'xgboost_b_no_extras.json'))

# --- 3. ARIMA BASELINE FORECAST ---
print("Generating Forecasts...")
arima_predictions = arima_fitted.forecast(steps=len(test_a))
arima_predictions.index = test_a.index

# --- 4. XGBOOST A: PREDICTION & REVERSAL ---
preds_log_a = xgb_a.predict(X_test_a)

yesterday_price_a = df_master.loc[X_test_a.index, 'Wheat_Close'].shift(1)
# Fetch the last known price right before the test set begins
last_val_date = df_master[:'2024-12-31'].index[-1]
yesterday_price_a.iloc[0] = df_master.loc[last_val_date, 'Wheat_Close']

xgb_a_price_predictions = yesterday_price_a * np.exp(preds_log_a)

# --- 5. XGBOOST B: PREDICTION & REVERSAL ---
preds_resid_b = xgb_b.predict(X_test_b)
trend_expected_b = df_b.loc[X_test_b.index, 'Wheat_Close_Trend']
xgb_b_price_predictions = trend_expected_b + preds_resid_b

# --- 6. CALCULATE FINAL SCORES (RMSE) ---
actual_prices = df_master.loc[test_a.index, 'Wheat_Close']

rmse_arima = root_mean_squared_error(actual_prices, arima_predictions)
rmse_xgb_a = root_mean_squared_error(actual_prices, xgb_a_price_predictions)
rmse_xgb_b = root_mean_squared_error(actual_prices, xgb_b_price_predictions)

print("\n=== FINAL EXAM RESULTS - NO EXTRAS (Test Set: 2025 - Present) ===")
print(f"ARIMA Baseline RMSE:        ${rmse_arima:.2f}")
print(f"XGBoost A (Log Returns):    ${rmse_xgb_a:.2f}")
print(f"XGBoost B (Detrended):      ${rmse_xgb_b:.2f}\n")

# --- 7. VISUALIZATION ---
print("Generating Visualizations...")
plt.figure(figsize=(16, 8))
plt.plot(actual_prices.index, actual_prices, label='Actual Wheat Price', color='black', linewidth=2)
plt.plot(arima_predictions.index, arima_predictions, label=f'ARIMA (RMSE: ${rmse_arima:.2f})', color='gray', linestyle='dashed')
plt.plot(xgb_a_price_predictions.index, xgb_a_price_predictions, label=f'XGBoost A (RMSE: ${rmse_xgb_a:.2f})', color='blue', alpha=0.7)
plt.plot(xgb_b_price_predictions.index, xgb_b_price_predictions, label=f'XGBoost B (RMSE: ${rmse_xgb_b:.2f})', color='red', alpha=0.7)

plt.title('Wheat Forecast: No Extras - Baseline vs. XGBoost (Test Set)', fontsize=16)
plt.ylabel('Wheat Price (USD/Bu)', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(12, 8))
plot_importance(xgb_a, max_num_features=10, importance_type='weight',
                title='Top 10 Drivers - No Extras (XGBoost A)',
                xlabel='F-Score (Number of splits)', color='darkblue')
plt.show()
