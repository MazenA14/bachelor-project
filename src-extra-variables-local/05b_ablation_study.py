import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
import warnings

warnings.filterwarnings("ignore")

print("Initiating Phase 5: Ablation Study (Global vs. Local vs. Hybrid) - NO EXTRAS...\n")

# --- 1. DIRECTORY SETUP (Using your new folder structure) ---
ROOT_DIR = '../data-extra-variables-local/'
MASTER_FILE = os.path.join(ROOT_DIR, '02_processed', '01_master_metals_dataset.csv')
DIFF_FILE = os.path.join(ROOT_DIR, '03_final', '01a_differencing_metals_dataset.csv')

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
# We use the exact drop list from 03_model_training_no_extras.py
drop_cols_a = ['Gold_Close', 'Silver_Close', 'DXY_Close', 'EGP_USD_Close', 'Local_Gold_24k_EGP',
               'Gold_Close_LogReturn', 'Silver_Close_LogReturn', 'DXY_Close_LogReturn', 'EGP_USD_Close_LogReturn', 'Local_Gold_24k_EGP_LogReturn']

cols_to_drop = [c for c in drop_cols_a if c in df_diff.columns]

# The Hybrid Dataset has EVERYTHING
y_train = train['Local_Gold_24k_EGP_LogReturn']
X_train_hybrid = train.drop(columns=cols_to_drop)

y_val = val['Local_Gold_24k_EGP_LogReturn']
X_val_hybrid = val.drop(columns=cols_to_drop)

y_test = test['Local_Gold_24k_EGP_LogReturn']
X_test_hybrid = test.drop(columns=cols_to_drop)

# --- 4. FEATURE ISOLATION (The Ablation Logic) ---
all_features = X_train_hybrid.columns.tolist()

# Define keywords that belong to each domain
global_keywords = ['Silver', 'DXY', 'VIX', 'SP500', '10Yr', 'Ratio']
local_keywords = ['EGP', 'Inflation', 'CBE', 'Local_Gold']

# A. Global Only Matrix (Drop all Egyptian/local keywords)
global_cols = [c for c in all_features if not any(local_word in c for local_word in local_keywords)]
X_train_global, X_val_global, X_test_global = X_train_hybrid[global_cols], X_val_hybrid[global_cols], X_test_hybrid[global_cols]

# B. Local Only Matrix (Drop all global keywords)
local_cols = [c for c in all_features if not any(global_word in c for global_word in global_keywords)]
X_train_local, X_val_local, X_test_local = X_train_hybrid[local_cols], X_val_hybrid[local_cols], X_test_hybrid[local_cols]

print(f"Hybrid Features: {len(X_train_hybrid.columns)}")
print(f"Global Features: {len(X_train_global.columns)}")
print(f"Local Features:  {len(X_train_local.columns)}\n")

# --- 5. MODEL TRAINING ---
print("Training Hybrid Model (All Variables)...")
# Before tuning
# xgb_hybrid = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=5, random_state=42, early_stopping_rounds=50)

# After tuning
xgb_hybrid = xgb.XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_hybrid.fit(X_train_hybrid, y_train, eval_set=[(X_val_hybrid, y_val)], verbose=False)

print("Training Pure Global Model...")
# Before tuning
# xgb_global = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=5, random_state=42, early_stopping_rounds=50)

# After tuning
xgb_global = xgb.XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_global.fit(X_train_global, y_train, eval_set=[(X_val_global, y_val)], verbose=False)

print("Training Pure Local Model...")
# After tuning
xgb_local = xgb.XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_local.fit(X_train_local, y_train, eval_set=[(X_val_local, y_val)], verbose=False)

# --- 6. ONE-STEP-AHEAD EVALUATION (Mathematical Reversal) ---
print("\nExecuting Final Exam & Price Reversal...")

# Get predictions
preds_hybrid = xgb_hybrid.predict(X_test_hybrid)
preds_global = xgb_global.predict(X_test_global)
preds_local = xgb_local.predict(X_test_local)

# Fetch yesterday's actual prices to reverse the Log Return
yesterday_price = df_master.loc[X_test_hybrid.index, 'Local_Gold_24k_EGP'].shift(1)
yesterday_price.iloc[0] = df_master.loc['2024-12-31', 'Local_Gold_24k_EGP'] # Fill first day

# Reverse math: P_today = P_yesterday * exp(Log_Return)
price_hybrid = yesterday_price * np.exp(preds_hybrid)
price_global = yesterday_price * np.exp(preds_global)
price_local = yesterday_price * np.exp(preds_local)

actual_prices = df_master.loc[test.index, 'Local_Gold_24k_EGP']

# Calculate RMSE
rmse_hybrid = root_mean_squared_error(actual_prices, price_hybrid)
rmse_global = root_mean_squared_error(actual_prices, price_global)
rmse_local = root_mean_squared_error(actual_prices, price_local)

print("=== ABLATION STUDY RESULTS (RMSE) ===")
print(f"Hybrid Model (All Data):        ${rmse_hybrid:.2f}")
print(f"Global Model (Wall St Only):    ${rmse_global:.2f}")
print(f"Local Model (Egypt Only):       ${rmse_local:.2f}\n")

# --- 7. VISUALIZATION ---
# Plot 1: The RMSE Comparison Bar Chart
models = ['Global Only (Wall St)', 'Local Only (Egypt)', 'Hybrid (All Data)']
scores = [rmse_global, rmse_local, rmse_hybrid]
colors = ['#5B9BD5', '#ED7D31', '#2E4057']

plt.figure(figsize=(10, 6))
bars = plt.bar(models, scores, color=colors)
plt.title('Ablation Study: Forecasting Error (RMSE) by Feature Set (No Extras)', fontsize=14)
plt.ylabel('Error in EGP', fontsize=12)

# Add exact numbers on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')
plt.show()

# Plot 2: Time Series Overlay
plt.figure(figsize=(16, 8))
plt.plot(actual_prices.index, actual_prices, label='Actual Local Gold Price', color='black', linewidth=2)
plt.plot(price_global.index, price_global, label=f'Global Only (RMSE: {rmse_global:.2f})', color='#5B9BD5', alpha=0.7)
plt.plot(price_local.index, price_local, label=f'Local Only (RMSE: {rmse_local:.2f})', color='#ED7D31', alpha=0.7)
plt.plot(price_hybrid.index, price_hybrid, label=f'Hybrid Model (RMSE: {rmse_hybrid:.2f})', color='#2E4057', linewidth=2)

plt.title('Local Gold (24K EGP) Forecast: Testing Feature Isolation (No Extras)', fontsize=16)
plt.ylabel('Gold Price (EGP)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

