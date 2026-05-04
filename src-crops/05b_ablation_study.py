# =============================================================================
# PIPELINE IDENTITY
#   Extra Features      : NO (No Extras — only raw prices + stationarity columns)
#   Stationarity Method : DIFFERENCING  — reads 01a_differencing_crops_dataset.csv
#   Ablation Logic      : Global Only (no local Egyptian data) vs Hybrid (Global + Local)
#   Target              : Wheat_Close_LogReturn
#   Trained by          : This script trains its own internal models (not from Phase 3 files)
# =============================================================================
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

print("Initiating Phase 5: Ablation Study (Crops - No Extras)...\n")
print("Research Question: Does adding local Egyptian economic data improve Wheat forecasting?\n")

# --- 1. DIRECTORY SETUP ---
ROOT_DIR = '../data-crops/'
MASTER_FILE = os.path.join(ROOT_DIR, '02_processed', '01_master_crops_dataset.csv')
DIFF_FILE = os.path.join(ROOT_DIR, '03_final', '01a_differencing_crops_dataset.csv')

# --- 2. DATA LOADING & TIMELINE SPLIT ---
df_master = pd.read_csv(MASTER_FILE, index_col='Date', parse_dates=['Date'])
df_diff = pd.read_csv(DIFF_FILE, index_col='Date', parse_dates=['Date'])

TRAIN_END = '2023-12-31'
VAL_START = '2024-01-01'
VAL_END = '2024-12-31'
TEST_START = '2025-01-01'

train = df_diff[:TRAIN_END]
val = df_diff[VAL_START:VAL_END]
test = df_diff[TEST_START:]

# --- 3. TARGET ISOLATION & LEAKAGE PREVENTION ---
# Drop raw (non-stationary) prices — they must not be used as features.
# Only drop Wheat's own LogReturn (target leakage).
# Other assets' LogReturns (Corn, Sugar, Brent, DXY, EGP_USD) are valid input features.
potential_leakage = [
    'Wheat_Close', 'Corn_Close', 'Sugar_Close', 'Brent_Crude_Close', 'DXY_Close', 'EGP_USD_Close',
    'Wheat_Close_LogReturn'
]
cols_to_drop = [c for c in potential_leakage if c in df_diff.columns]

# The Hybrid Dataset has EVERYTHING (global + local) — target is Wheat Close log return
y_train = train['Wheat_Close_LogReturn']
X_train_hybrid = train.drop(columns=cols_to_drop)

y_val = val['Wheat_Close_LogReturn']
X_val_hybrid = val.drop(columns=cols_to_drop)

y_test = test['Wheat_Close_LogReturn']
X_test_hybrid = test.drop(columns=cols_to_drop)

# --- 4. FEATURE ISOLATION (The Ablation Logic) ---
all_features = X_train_hybrid.columns.tolist()

# Local Egyptian feature keywords — these are what we are testing the impact of
local_keywords = ['EGP', 'Inflation', 'CBE']

# Global Only: keep everything EXCEPT local Egyptian features
# (crops: Wheat + Corn/Sugar, AND macro: Brent, DXY)
global_only_cols = [c for c in all_features if not any(kw in c for kw in local_keywords)]
X_train_global = X_train_hybrid[global_only_cols]
X_val_global = X_val_hybrid[global_only_cols]
X_test_global = X_test_hybrid[global_only_cols]

print(f"Hybrid Features (Global + Local):  {len(X_train_hybrid.columns)}")
print(f"Global-Only Features (no local):   {len(X_train_global.columns)}")
print(f"Local Features being tested:       {len(X_train_hybrid.columns) - len(X_train_global.columns)}\n")

# --- 5. MODEL TRAINING ---
print("Training Hybrid Model (Global + Local Egyptian features)...")
xgb_hybrid = xgb.XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=42, early_stopping_rounds=50)
xgb_hybrid.fit(X_train_hybrid, y_train, eval_set=[(X_val_hybrid, y_val)], verbose=False)

print("Training Global-Only Model (no local Egyptian data)...")
xgb_global = xgb.XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=42, early_stopping_rounds=50)
xgb_global.fit(X_train_global, y_train, eval_set=[(X_val_global, y_val)], verbose=False)

# --- 6. ONE-STEP-AHEAD EVALUATION (Mathematical Reversal) ---
print("\nExecuting Final Exam & Price Reversal...")

# Get predictions
preds_hybrid = xgb_hybrid.predict(X_test_hybrid)
preds_global = xgb_global.predict(X_test_global)

# Fetch yesterday's actual prices to reverse the Log Return
yesterday_price = df_master.loc[X_test_hybrid.index, 'Wheat_Close'].shift(1)
yesterday_price.iloc[0] = df_master.loc['2024-12-31', 'Wheat_Close']  # Fill first day

# Reverse math: P_today = P_yesterday * exp(Log_Return)
price_hybrid = yesterday_price * np.exp(preds_hybrid)
price_global = yesterday_price * np.exp(preds_global)

actual_prices = df_master.loc[test.index, 'Wheat_Close']

# Calculate metrics
rmse_hybrid = root_mean_squared_error(actual_prices, price_hybrid)
rmse_global = root_mean_squared_error(actual_prices, price_global)

mape_hybrid = mean_absolute_percentage_error(actual_prices, price_hybrid)
mape_global = mean_absolute_percentage_error(actual_prices, price_global)

mae_hybrid = mean_absolute_error(actual_prices, price_hybrid)
mae_global = mean_absolute_error(actual_prices, price_global)

mse_hybrid = mean_squared_error(actual_prices, price_hybrid)
mse_global = mean_squared_error(actual_prices, price_global)

r2_hybrid = r2_score(actual_prices, price_hybrid)
r2_global = r2_score(actual_prices, price_global)

print("=== ABLATION STUDY RESULTS (RMSE) ===")
print(f"Global-Only Model (no EGP/Inflation/CBE):  ${rmse_global:.2f} | MAPE: {mape_global:.4f} | MAE: {mae_global:.2f} | MSE: {mse_global:.2f} | R2: {r2_global:.4f}")
print(f"Hybrid Model (Global + Local Egyptian):    ${rmse_hybrid:.2f} | MAPE: {mape_hybrid:.4f} | MAE: {mae_hybrid:.2f} | MSE: {mse_hybrid:.2f} | R2: {r2_hybrid:.4f}")
improvement = rmse_global - rmse_hybrid
print(f"\nLocal data contribution: ${improvement:.2f} RMSE improvement" if improvement > 0
      else f"\nLocal data had no benefit: ${abs(improvement):.2f} RMSE degradation\n")

# --- 7. VISUALIZATION ---
# Plot 1: The RMSE Comparison Bar Chart
models = ['Global Only\n(No Local Data)', 'Hybrid\n(Global + Local Egyptian)']
scores = [rmse_global, rmse_hybrid]
colors = ['steelblue', 'darkblue']

plt.figure(figsize=(10, 6))
bars = plt.bar(models, scores, color=colors, width=0.5)
plt.title('Ablation Study: Impact of Local Egyptian Data on Wheat Forecast (No Extras)', fontsize=13)
plt.ylabel('RMSE in USD ($/Bu)', fontsize=12)
plt.ylim(0, max(scores) * 1.2)

# Add exact numbers on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'${yval:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.show()

# Plot 2: Time Series Overlay
plt.figure(figsize=(16, 8))
plt.plot(actual_prices.index, actual_prices, label='Actual Wheat Price', color='black', linewidth=2)
plt.plot(price_global.index, price_global, label=f'Global Only (RMSE: ${rmse_global:.2f})', color='steelblue', alpha=0.8, linestyle='dashed')
plt.plot(price_hybrid.index, price_hybrid, label=f'Hybrid — Global + Local (RMSE: ${rmse_hybrid:.2f})', color='darkblue', linewidth=2)

plt.title('Wheat Forecast: Global Only vs Hybrid (No Extras)', fontsize=16)
plt.ylabel('Wheat Price (USD/Bu)', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()
