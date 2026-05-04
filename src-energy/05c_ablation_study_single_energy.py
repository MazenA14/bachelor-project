# =============================================================================
# PIPELINE IDENTITY
#   Ablation Logic      : Cross-Energy Feature Contribution (Brent ↔ NatGas)
#   Stationarity Method : DIFFERENCING — reads 01a_differencing_energy_dataset.csv
#   Local Egyptian Data : INCLUDED
#   Part A Target       : Brent_Crude_Close_LogReturn  (with vs. without NatGas)
#   Part B Target       : Natural_Gas_Close_LogReturn   (with vs. without Brent)
# =============================================================================
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import warnings

warnings.filterwarnings("ignore")

print("Initiating Phase 5c: Ablation Study (Cross-Energy Isolation)...\n")

# --- 1. DIRECTORY SETUP ---
ROOT_DIR = '../data-energy/'
MASTER_FILE = os.path.join(ROOT_DIR, '02_processed', '01_master_energy_dataset.csv')
DIFF_FILE = os.path.join(ROOT_DIR, '03_final', '01a_differencing_energy_dataset.csv')

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

# =====================================================================
#  PART A — BRENT CRUDE PREDICTION (with vs. without NatGas features)
# =====================================================================
print("=" * 60)
print("  PART A: BRENT CRUDE PRICE PREDICTION")
print("=" * 60)

# --- 3A. DROP LIST FOR BRENT TARGET ---
# Drop raw (non-stationary) prices.
# Only drop Brent's own LogReturn (target leakage).
# KEEP Natural_Gas_Close_LogReturn — valid input feature for Brent prediction.
drop_cols_brent_full = ['Brent_Crude_Close', 'Natural_Gas_Close', 'DXY_Close',
                        'VIX_Close', 'SP500_Close', 'EGP_USD_Close',
                        'Brent_Crude_Close_LogReturn']

cols_to_drop_brent_full = [c for c in drop_cols_brent_full if c in df_diff.columns]

# Target
y_train_brent = train['Brent_Crude_Close_LogReturn']
y_val_brent   = val['Brent_Crude_Close_LogReturn']
y_test_brent  = test['Brent_Crude_Close_LogReturn']

# Full feature set (includes Natural_Gas_Close_LogReturn as an input)
X_train_full_brent = train.drop(columns=cols_to_drop_brent_full)
X_val_full_brent   = val.drop(columns=cols_to_drop_brent_full)
X_test_full_brent  = test.drop(columns=cols_to_drop_brent_full)

# Brent-Only feature set: additionally remove NatGas-related columns
natgas_keywords = ['Natural_Gas', 'NatGas', 'Ratio']
brent_only_cols = [c for c in X_train_full_brent.columns
                   if not any(kw in c for kw in natgas_keywords)]

X_train_brent_only = X_train_full_brent[brent_only_cols]
X_val_brent_only   = X_val_full_brent[brent_only_cols]
X_test_brent_only  = X_test_full_brent[brent_only_cols]

print(f"\nBrent — Full features:        {len(X_train_full_brent.columns)}  →  {list(X_train_full_brent.columns)}")
print(f"Brent — Brent-only features:  {len(X_train_brent_only.columns)}  →  {list(X_train_brent_only.columns)}\n")

# --- 4A. MODEL TRAINING ---
print("Training Brent Full Model (with NatGas features)...")
xgb_brent_full = xgb.XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=5,
    random_state=42, early_stopping_rounds=50)
xgb_brent_full.fit(X_train_full_brent, y_train_brent,
                   eval_set=[(X_val_full_brent, y_val_brent)], verbose=False)

print("Training Brent-Only Model (without NatGas features)...")
xgb_brent_only = xgb.XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=5,
    random_state=42, early_stopping_rounds=50)
xgb_brent_only.fit(X_train_brent_only, y_train_brent,
                   eval_set=[(X_val_brent_only, y_val_brent)], verbose=False)

# --- 5A. PRICE REVERSAL & EVALUATION ---
preds_brent_full = xgb_brent_full.predict(X_test_full_brent)
preds_brent_only = xgb_brent_only.predict(X_test_brent_only)

yesterday_brent = df_master.loc[X_test_full_brent.index, 'Brent_Crude_Close'].shift(1)
yesterday_brent.iloc[0] = df_master.loc['2024-12-31', 'Brent_Crude_Close']

price_brent_full = yesterday_brent * np.exp(preds_brent_full)
price_brent_only = yesterday_brent * np.exp(preds_brent_only)
actual_brent = df_master.loc[test.index, 'Brent_Crude_Close']

rmse_brent_full = root_mean_squared_error(actual_brent, price_brent_full)
mse_brent_full = mean_squared_error(actual_brent, price_brent_full)
mae_brent_full = mean_absolute_error(actual_brent, price_brent_full)
mape_brent_full = mean_absolute_percentage_error(actual_brent, price_brent_full)
r2_brent_full = r2_score(actual_brent, price_brent_full)

rmse_brent_only = root_mean_squared_error(actual_brent, price_brent_only)
mse_brent_only = mean_squared_error(actual_brent, price_brent_only)
mae_brent_only = mean_absolute_error(actual_brent, price_brent_only)
mape_brent_only = mean_absolute_percentage_error(actual_brent, price_brent_only)
r2_brent_only = r2_score(actual_brent, price_brent_only)

print("\n=== BRENT ABLATION RESULTS (RMSE) ===")
print(f"Full Model (with NatGas):       ${rmse_brent_full:.2f} | MAPE: {mape_brent_full:.4f} | MAE: {mae_brent_full:.2f} | MSE: {mse_brent_full:.2f} | R2: {r2_brent_full:.4f}")
print(f"Brent-Only Model (no NatGas):   ${rmse_brent_only:.2f} | MAPE: {mape_brent_only:.4f} | MAE: {mae_brent_only:.2f} | MSE: {mse_brent_only:.2f} | R2: {r2_brent_only:.4f}\n")

# =====================================================================
#  PART B — NATURAL GAS PREDICTION (with vs. without Brent features)
# =====================================================================
print("=" * 60)
print("  PART B: NATURAL GAS PRICE PREDICTION")
print("=" * 60)

# --- 3B. DROP LIST FOR NATGAS TARGET ---
# Drop raw prices. Only drop NatGas's own LogReturn (target leakage).
# KEEP Brent_Crude_Close_LogReturn — valid input feature for NatGas prediction.
drop_cols_natgas_full = ['Brent_Crude_Close', 'Natural_Gas_Close', 'DXY_Close',
                         'VIX_Close', 'SP500_Close', 'EGP_USD_Close',
                         'Natural_Gas_Close_LogReturn']

cols_to_drop_natgas_full = [c for c in drop_cols_natgas_full if c in df_diff.columns]

# Target
y_train_natgas = train['Natural_Gas_Close_LogReturn']
y_val_natgas   = val['Natural_Gas_Close_LogReturn']
y_test_natgas  = test['Natural_Gas_Close_LogReturn']

# Full feature set (includes Brent_Crude_Close_LogReturn as an input)
X_train_full_natgas = train.drop(columns=cols_to_drop_natgas_full)
X_val_full_natgas   = val.drop(columns=cols_to_drop_natgas_full)
X_test_full_natgas  = test.drop(columns=cols_to_drop_natgas_full)

# NatGas-Only feature set: additionally remove Brent-related columns
brent_keywords = ['Brent', 'Ratio']
natgas_only_cols = [c for c in X_train_full_natgas.columns
                    if not any(kw in c for kw in brent_keywords)]

X_train_natgas_only = X_train_full_natgas[natgas_only_cols]
X_val_natgas_only   = X_val_full_natgas[natgas_only_cols]
X_test_natgas_only  = X_test_full_natgas[natgas_only_cols]

print(f"\nNatGas — Full features:          {len(X_train_full_natgas.columns)}  →  {list(X_train_full_natgas.columns)}")
print(f"NatGas — NatGas-only features:   {len(X_train_natgas_only.columns)}  →  {list(X_train_natgas_only.columns)}\n")

# --- 4B. MODEL TRAINING ---
print("Training NatGas Full Model (with Brent features)...")
xgb_natgas_full = xgb.XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=5,
    random_state=42, early_stopping_rounds=50)
xgb_natgas_full.fit(X_train_full_natgas, y_train_natgas,
                    eval_set=[(X_val_full_natgas, y_val_natgas)], verbose=False)

print("Training NatGas-Only Model (without Brent features)...")
xgb_natgas_only = xgb.XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=5,
    random_state=42, early_stopping_rounds=50)
xgb_natgas_only.fit(X_train_natgas_only, y_train_natgas,
                    eval_set=[(X_val_natgas_only, y_val_natgas)], verbose=False)

# --- 5B. PRICE REVERSAL & EVALUATION ---
preds_natgas_full = xgb_natgas_full.predict(X_test_full_natgas)
preds_natgas_only = xgb_natgas_only.predict(X_test_natgas_only)

yesterday_natgas = df_master.loc[X_test_full_natgas.index, 'Natural_Gas_Close'].shift(1)
yesterday_natgas.iloc[0] = df_master.loc['2024-12-31', 'Natural_Gas_Close']

price_natgas_full = yesterday_natgas * np.exp(preds_natgas_full)
price_natgas_only = yesterday_natgas * np.exp(preds_natgas_only)
actual_natgas = df_master.loc[test.index, 'Natural_Gas_Close']

rmse_natgas_full = root_mean_squared_error(actual_natgas, price_natgas_full)
mse_natgas_full = mean_squared_error(actual_natgas, price_natgas_full)
mae_natgas_full = mean_absolute_error(actual_natgas, price_natgas_full)
mape_natgas_full = mean_absolute_percentage_error(actual_natgas, price_natgas_full)
r2_natgas_full = r2_score(actual_natgas, price_natgas_full)

rmse_natgas_only = root_mean_squared_error(actual_natgas, price_natgas_only)
mse_natgas_only = mean_squared_error(actual_natgas, price_natgas_only)
mae_natgas_only = mean_absolute_error(actual_natgas, price_natgas_only)
mape_natgas_only = mean_absolute_percentage_error(actual_natgas, price_natgas_only)
r2_natgas_only = r2_score(actual_natgas, price_natgas_only)

print("\n=== NATGAS ABLATION RESULTS (RMSE) ===")
print(f"Full Model (with Brent):        ${rmse_natgas_full:.2f} | MAPE: {mape_natgas_full:.4f} | MAE: {mae_natgas_full:.2f} | MSE: {mse_natgas_full:.2f} | R2: {r2_natgas_full:.4f}")
print(f"NatGas-Only Model (no Brent):   ${rmse_natgas_only:.2f} | MAPE: {mape_natgas_only:.4f} | MAE: {mae_natgas_only:.2f} | MSE: {mse_natgas_only:.2f} | R2: {r2_natgas_only:.4f}\n")

# =====================================================================
#  VISUALIZATION
# =====================================================================

# --- Plot 1: RMSE Comparison Bar Chart (Both Commodities) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Brent bars
models_brent = ['Full (w/ NatGas)', 'Brent-Only']
scores_brent = [rmse_brent_full, rmse_brent_only]
colors_brent = ['saddlebrown', 'peru']
bars_b = axes[0].bar(models_brent, scores_brent, color=colors_brent, edgecolor='black', linewidth=0.5)
axes[0].set_title('Brent Forecast: Impact of NatGas Features', fontsize=13)
axes[0].set_ylabel('RMSE (USD/bbl)', fontsize=12)
for bar in bars_b:
    yval = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, yval + 0.02,
                 f'${yval:.2f}', ha='center', va='bottom', fontweight='bold')

# NatGas bars
models_natgas = ['Full (w/ Brent)', 'NatGas-Only']
scores_natgas = [rmse_natgas_full, rmse_natgas_only]
colors_natgas = ['teal', 'mediumaquamarine']
bars_n = axes[1].bar(models_natgas, scores_natgas, color=colors_natgas, edgecolor='black', linewidth=0.5)
axes[1].set_title('NatGas Forecast: Impact of Brent Features', fontsize=13)
axes[1].set_ylabel('RMSE (USD/MMBtu)', fontsize=12)
for bar in bars_n:
    yval = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2, yval + 0.02,
                 f'${yval:.2f}', ha='center', va='bottom', fontweight='bold')

plt.suptitle('Ablation Study: Cross-Energy Feature Contribution', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# --- Plot 2: Brent Time-Series Overlay ---
plt.figure(figsize=(16, 8))
plt.plot(actual_brent.index, actual_brent, label='Actual Brent Crude Price', color='black', linewidth=2)
plt.plot(price_brent_full.index, price_brent_full,
         label=f'Full Model w/ NatGas (RMSE: ${rmse_brent_full:.2f})', color='saddlebrown', linewidth=2)
plt.plot(price_brent_only.index, price_brent_only,
         label=f'Brent-Only Model (RMSE: ${rmse_brent_only:.2f})', color='peru', alpha=0.7)
plt.title('Brent Crude Forecast: With vs. Without NatGas Features', fontsize=16)
plt.ylabel('Brent Crude Price (USD/bbl)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

# --- Plot 3: NatGas Time-Series Overlay ---
plt.figure(figsize=(16, 8))
plt.plot(actual_natgas.index, actual_natgas, label='Actual Natural Gas Price', color='black', linewidth=2)
plt.plot(price_natgas_full.index, price_natgas_full,
         label=f'Full Model w/ Brent (RMSE: ${rmse_natgas_full:.2f})', color='teal', linewidth=2)
plt.plot(price_natgas_only.index, price_natgas_only,
         label=f'NatGas-Only Model (RMSE: ${rmse_natgas_only:.2f})', color='mediumaquamarine', alpha=0.7)
plt.title('Natural Gas Forecast: With vs. Without Brent Features', fontsize=16)
plt.ylabel('Natural Gas Price (USD/MMBtu)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

print("\n--- CROSS-ENERGY ABLATION STUDY COMPLETE ---")
