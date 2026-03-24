import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import root_mean_squared_error
from xgboost import plot_importance
import warnings

warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION & DATA LOADING ---
print("Initiating Phase 4: Model Evaluation Protocol...\n")
PROCESSED_DIR = '../data-extra-variables/02_processed/'
FINAL_DIR = '../data-extra-variables/03_final/'
MODELS_DIR = '../models-extra-variables/'
TEST_START = '2025-01-01'
TRAIN_VAL_END = '2024-12-31' # The exact day before the test set begins

print("Loading Test Data...")
df_master = pd.read_csv(os.path.join(PROCESSED_DIR, '01_master_metals_dataset.csv'), index_col='Date', parse_dates=['Date'])
df_a = pd.read_csv(os.path.join(FINAL_DIR, '01a_engineered_differencing_metals_dataset.csv'), index_col='Date', parse_dates=['Date'])
df_b = pd.read_csv(os.path.join(FINAL_DIR, '01b_engineered_detrending_metals_dataset.csv'), index_col='Date', parse_dates=['Date'])

test_a = df_a[TEST_START:]
test_b = df_b[TEST_START:]

drop_cols_a = ['Gold_Close', 'Silver_Close', 'DXY_Close', 'SP500_Close', 'VIX_Close', 'EGP_USD_Close',
               'Gold_Close_LogReturn', 'Silver_Close_LogReturn', 'DXY_Close_LogReturn',
               'SP500_Close_LogReturn', 'VIX_Close_LogReturn', 'EGP_USD_Close_LogReturn']
X_test_a = test_a.drop(columns=drop_cols_a)

drop_cols_b = ['Gold_Close', 'Silver_Close', 'DXY_Close', 'SP500_Close', 'VIX_Close', 'EGP_USD_Close',
               'Gold_Close_Trend', 'Gold_Close_Residual', 'Silver_Close_Trend', 'Silver_Close_Residual',
               'DXY_Close_Trend', 'DXY_Close_Residual', 'SP500_Close_Trend', 'SP500_Close_Residual',
               'VIX_Close_Trend', 'VIX_Close_Residual', 'EGP_USD_Close_Trend', 'EGP_USD_Close_Residual']
X_test_b = test_b.drop(columns=drop_cols_b)

# --- 2. MODEL LOADING ---
print("Loading trained XGBoost models from disk...")
xgb_a = xgb.XGBRegressor()
xgb_a.load_model(os.path.join(MODELS_DIR, 'xgboost_a.json'))

xgb_b = xgb.XGBRegressor()
xgb_b.load_model(os.path.join(MODELS_DIR, 'xgboost_b.json'))

# --- 3. ARIMA BASELINE: ONE-STEP-AHEAD ROLLING FORECAST ---
print("\nGenerating ARIMA Rolling Forecast (This takes a few minutes)...")

# We grab the pure historical data up to the last day of 2024
history = list(df_master['Gold_Close'][:TRAIN_VAL_END])
# We grab the actual answers for the test set
test_actuals = df_master['Gold_Close'][TEST_START:]

arima_predictions_list = []

# The Walk-Forward Loop
for t in range(len(test_actuals)):
    # 1. Initialize the model with the current history
    model = ARIMA(history, order=(5, 1, 0))
    # 2. Fit the mathematical equations
    model_fit = model.fit()
    # 3. Predict exactly 1 day into the future
    yhat = model_fit.forecast()[0]
    arima_predictions_list.append(yhat)
    # 4. Append the TRUE actual price to the history for tomorrow's calculation
    history.append(test_actuals.iloc[t])
    
    # Print progress every 50 days
    if (t + 1) % 50 == 0:
        print(f"  --> ARIMA Progress: {t + 1} / {len(test_actuals)} days predicted...")

# Convert the predictions list back into a pandas Series aligned with the dates
arima_predictions = pd.Series(arima_predictions_list, index=test_actuals.index)
print("ARIMA Forecast Complete.\n")

# --- 4. XGBOOST A: PREDICTION & REVERSAL ---
print("Generating XGBoost Forecasts...")
preds_log_a = xgb_a.predict(X_test_a)

yesterday_price_a = df_master.loc[X_test_a.index, 'Gold_Close'].shift(1)
yesterday_price_a.iloc[0] = df_master.loc[TRAIN_VAL_END, 'Gold_Close']
xgb_a_price_predictions = yesterday_price_a * np.exp(preds_log_a)

# --- 5. XGBOOST B: PREDICTION & REVERSAL ---
preds_resid_b = xgb_b.predict(X_test_b)
trend_expected_b = df_b.loc[X_test_b.index, 'Gold_Close_Trend']
xgb_b_price_predictions = trend_expected_b + preds_resid_b

# --- 6. CALCULATE FINAL SCORES (RMSE) ---
actual_prices = df_master.loc[test_a.index, 'Gold_Close']

rmse_arima = root_mean_squared_error(actual_prices, arima_predictions)
rmse_xgb_a = root_mean_squared_error(actual_prices, xgb_a_price_predictions)
rmse_xgb_b = root_mean_squared_error(actual_prices, xgb_b_price_predictions)

print("\n=== FINAL EXAM RESULTS (Test Set: 2025 - Present) ===")
print(f"ARIMA Baseline RMSE:        ${rmse_arima:.2f}")
print(f"XGBoost A (Log Returns):    ${rmse_xgb_a:.2f}")
print(f"XGBoost B (Detrended):      ${rmse_xgb_b:.2f}\n")

# --- 7. VISUALIZATION ---
print("Generating Visualizations...")
plt.figure(figsize=(16, 8))
plt.plot(actual_prices.index, actual_prices, label='Actual Gold Price', color='black', linewidth=2)
plt.plot(arima_predictions.index, arima_predictions, label=f'ARIMA (RMSE: ${rmse_arima:.2f})', color='gray', linestyle='dashed')
plt.plot(xgb_a_price_predictions.index, xgb_a_price_predictions, label=f'XGBoost A (RMSE: ${rmse_xgb_a:.2f})', color='blue', alpha=0.7)
plt.plot(xgb_b_price_predictions.index, xgb_b_price_predictions, label=f'XGBoost B (RMSE: ${rmse_xgb_b:.2f})', color='red', alpha=0.7)

plt.title('Global Gold Forecast: 1-Day Rolling Horizon (Test Set)', fontsize=16)
plt.ylabel('Gold Price (USD)', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(12, 8))
plot_importance(xgb_a, max_num_features=10, importance_type='weight', 
                title='Top 10 Drivers of Global Gold Prices (XGBoost A)',
                xlabel='F-Score (Number of splits)', color='darkblue')
plt.show()