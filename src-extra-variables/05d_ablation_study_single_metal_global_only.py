import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
import warnings

warnings.filterwarnings("ignore")

print("Initiating Phase 5d: Ablation Study (Single-Metal Isolation, Global Only)...\n")

# --- 1. DIRECTORY SETUP ---
ROOT_DIR = '../data-extra-variables/'
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

# --- 2b. STRIP LOCAL EGYPTIAN FEATURES FROM ALL DATA ---
local_keywords = ['EGP', 'Inflation', 'CBE']
local_cols = [c for c in df_diff.columns if any(kw in c for kw in local_keywords)]
print(f"Removing local Egyptian features: {local_cols}\n")

train = train.drop(columns=local_cols)
val = val.drop(columns=local_cols)
test = test.drop(columns=local_cols)

# =====================================================================
#  PART A — GOLD PRICE PREDICTION (with vs. without Silver features)
# =====================================================================
print("=" * 60)
print("  PART A: GOLD PRICE PREDICTION (Global Only)")
print("=" * 60)

# --- 3A. DROP LIST FOR GOLD TARGET ---
# Drop raw prices + Gold's own LogReturn (target leakage).
# KEEP Silver_Close_LogReturn — it's a valid input feature when predicting Gold.
drop_cols_gold_full = ['Gold_Close', 'Silver_Close', 'DXY_Close',
                       'Gold_Close_LogReturn',
                       'DXY_Close_LogReturn']

cols_to_drop_gold_full = [c for c in drop_cols_gold_full if c in train.columns]

# Target
y_train_gold = train['Gold_Close_LogReturn']
y_val_gold   = val['Gold_Close_LogReturn']
y_test_gold  = test['Gold_Close_LogReturn']

# Full feature set (includes Silver_Close_LogReturn as an input)
X_train_full_gold = train.drop(columns=cols_to_drop_gold_full)
X_val_full_gold   = val.drop(columns=cols_to_drop_gold_full)
X_test_full_gold  = test.drop(columns=cols_to_drop_gold_full)

# Gold-Only feature set: additionally remove Silver-related columns
silver_keywords = ['Silver', 'Ratio']   # Gold_Silver_Ratio is cross-metal
gold_only_cols = [c for c in X_train_full_gold.columns
                  if not any(kw in c for kw in silver_keywords)]

X_train_gold_only = X_train_full_gold[gold_only_cols]
X_val_gold_only   = X_val_full_gold[gold_only_cols]
X_test_gold_only  = X_test_full_gold[gold_only_cols]

print(f"\nGold — Full features:       {len(X_train_full_gold.columns)}  →  {list(X_train_full_gold.columns)}")
print(f"Gold — Gold-only features:  {len(X_train_gold_only.columns)}  →  {list(X_train_gold_only.columns)}\n")

# --- 4A. MODEL TRAINING ---
print("Training Gold Full Model (with Silver features)...")
xgb_gold_full = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.01, max_depth=3,
    subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_gold_full.fit(X_train_full_gold, y_train_gold,
                  eval_set=[(X_val_full_gold, y_val_gold)], verbose=False)

print("Training Gold-Only Model (without Silver features)...")
xgb_gold_only_model = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.01, max_depth=3,
    subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_gold_only_model.fit(X_train_gold_only, y_train_gold,
                  eval_set=[(X_val_gold_only, y_val_gold)], verbose=False)

# --- 5A. PRICE REVERSAL & EVALUATION ---
preds_gold_full = xgb_gold_full.predict(X_test_full_gold)
preds_gold_only = xgb_gold_only_model.predict(X_test_gold_only)

yesterday_gold = df_master.loc[X_test_full_gold.index, 'Gold_Close'].shift(1)
yesterday_gold.iloc[0] = df_master.loc['2024-12-31', 'Gold_Close']

price_gold_full = yesterday_gold * np.exp(preds_gold_full)
price_gold_only = yesterday_gold * np.exp(preds_gold_only)
actual_gold = df_master.loc[test.index, 'Gold_Close']

rmse_gold_full = root_mean_squared_error(actual_gold, price_gold_full)
rmse_gold_only = root_mean_squared_error(actual_gold, price_gold_only)

print("\n=== GOLD ABLATION RESULTS (RMSE) ===")
print(f"Full Model (with Silver):     ${rmse_gold_full:.2f}")
print(f"Gold-Only Model (no Silver):  ${rmse_gold_only:.2f}\n")

# =====================================================================
#  PART B — SILVER PRICE PREDICTION (with vs. without Gold features)
# =====================================================================
print("=" * 60)
print("  PART B: SILVER PRICE PREDICTION (Global Only)")
print("=" * 60)

# --- 3B. DROP LIST FOR SILVER TARGET ---
# Drop raw prices + Silver's own LogReturn (target leakage).
# KEEP Gold_Close_LogReturn — it's a valid input feature when predicting Silver.
drop_cols_silver_full = ['Gold_Close', 'Silver_Close', 'DXY_Close',
                         'Silver_Close_LogReturn',
                         'DXY_Close_LogReturn']

cols_to_drop_silver_full = [c for c in drop_cols_silver_full if c in train.columns]

# Target
y_train_silver = train['Silver_Close_LogReturn']
y_val_silver   = val['Silver_Close_LogReturn']
y_test_silver  = test['Silver_Close_LogReturn']

# Full feature set (includes Gold_Close_LogReturn as an input)
X_train_full_silver = train.drop(columns=cols_to_drop_silver_full)
X_val_full_silver   = val.drop(columns=cols_to_drop_silver_full)
X_test_full_silver  = test.drop(columns=cols_to_drop_silver_full)

# Silver-Only feature set: additionally remove Gold-related columns
gold_keywords = ['Gold', 'Ratio']   # Gold_Silver_Ratio is cross-metal
silver_only_cols = [c for c in X_train_full_silver.columns
                    if not any(kw in c for kw in gold_keywords)]

X_train_silver_only = X_train_full_silver[silver_only_cols]
X_val_silver_only   = X_val_full_silver[silver_only_cols]
X_test_silver_only  = X_test_full_silver[silver_only_cols]

print(f"\nSilver — Full features:         {len(X_train_full_silver.columns)}  →  {list(X_train_full_silver.columns)}")
print(f"Silver — Silver-only features:  {len(X_train_silver_only.columns)}  →  {list(X_train_silver_only.columns)}\n")

# --- 4B. MODEL TRAINING ---
print("Training Silver Full Model (with Gold features)...")
xgb_silver_full = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.01, max_depth=3,
    subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_silver_full.fit(X_train_full_silver, y_train_silver,
                    eval_set=[(X_val_full_silver, y_val_silver)], verbose=False)

print("Training Silver-Only Model (without Gold features)...")
xgb_silver_only_model = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.01, max_depth=3,
    subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_silver_only_model.fit(X_train_silver_only, y_train_silver,
                    eval_set=[(X_val_silver_only, y_val_silver)], verbose=False)

# --- 5B. PRICE REVERSAL & EVALUATION ---
preds_silver_full = xgb_silver_full.predict(X_test_full_silver)
preds_silver_only = xgb_silver_only_model.predict(X_test_silver_only)

yesterday_silver = df_master.loc[X_test_full_silver.index, 'Silver_Close'].shift(1)
yesterday_silver.iloc[0] = df_master.loc['2024-12-31', 'Silver_Close']

price_silver_full = yesterday_silver * np.exp(preds_silver_full)
price_silver_only = yesterday_silver * np.exp(preds_silver_only)
actual_silver = df_master.loc[test.index, 'Silver_Close']

rmse_silver_full = root_mean_squared_error(actual_silver, price_silver_full)
rmse_silver_only = root_mean_squared_error(actual_silver, price_silver_only)

print("\n=== SILVER ABLATION RESULTS (RMSE) ===")
print(f"Full Model (with Gold):       ${rmse_silver_full:.2f}")
print(f"Silver-Only Model (no Gold):  ${rmse_silver_only:.2f}\n")

# =====================================================================
#  VISUALIZATION
# =====================================================================

# --- Plot 1: RMSE Comparison Bar Chart (Both Metals) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gold bars
models_gold = ['Full (w/ Silver)', 'Gold-Only']
scores_gold = [rmse_gold_full, rmse_gold_only]
colors_gold = ['goldenrod', 'gold']
bars_g = axes[0].bar(models_gold, scores_gold, color=colors_gold, edgecolor='black', linewidth=0.5)
axes[0].set_title('Gold Forecast: Impact of Silver Features', fontsize=13)
axes[0].set_ylabel('RMSE (USD)', fontsize=12)
for bar in bars_g:
    yval = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, yval + 0.5,
                 f'${yval:.2f}', ha='center', va='bottom', fontweight='bold')

# Silver bars
models_silver = ['Full (w/ Gold)', 'Silver-Only']
scores_silver = [rmse_silver_full, rmse_silver_only]
colors_silver = ['darkgray', 'silver']
bars_s = axes[1].bar(models_silver, scores_silver, color=colors_silver, edgecolor='black', linewidth=0.5)
axes[1].set_title('Silver Forecast: Impact of Gold Features', fontsize=13)
axes[1].set_ylabel('RMSE (USD)', fontsize=12)
for bar in bars_s:
    yval = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2, yval + 0.05,
                 f'${yval:.2f}', ha='center', va='bottom', fontweight='bold')

plt.suptitle('Ablation Study: Cross-Metal Feature Contribution (Global Only)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# --- Plot 2: Gold Time-Series Overlay ---
plt.figure(figsize=(16, 8))
plt.plot(actual_gold.index, actual_gold, label='Actual Gold Price', color='black', linewidth=2)
plt.plot(price_gold_full.index, price_gold_full,
         label=f'Full Model w/ Silver (RMSE: ${rmse_gold_full:.2f})', color='goldenrod', linewidth=2)
plt.plot(price_gold_only.index, price_gold_only,
         label=f'Gold-Only Model (RMSE: ${rmse_gold_only:.2f})', color='gold', alpha=0.7)
plt.title('Gold Price Forecast: With vs. Without Silver Features (Global Only)', fontsize=16)
plt.ylabel('Gold Price (USD)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

# --- Plot 3: Silver Time-Series Overlay ---
plt.figure(figsize=(16, 8))
plt.plot(actual_silver.index, actual_silver, label='Actual Silver Price', color='black', linewidth=2)
plt.plot(price_silver_full.index, price_silver_full,
         label=f'Full Model w/ Gold (RMSE: ${rmse_silver_full:.2f})', color='darkgray', linewidth=2)
plt.plot(price_silver_only.index, price_silver_only,
         label=f'Silver-Only Model (RMSE: ${rmse_silver_only:.2f})', color='silver', alpha=0.7)
plt.title('Silver Price Forecast: With vs. Without Gold Features (Global Only)', fontsize=16)
plt.ylabel('Silver Price (USD)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

print("\n--- SINGLE-METAL ABLATION STUDY (GLOBAL ONLY) COMPLETE ---")
