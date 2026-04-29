# =============================================================================
# PIPELINE IDENTITY
#   Ablation Logic      : Cross-Commodity Feature Contribution (Single Crop vs. All Three)
#   Stationarity Method : DIFFERENCING — reads 01a_differencing_crops_dataset.csv
#   Local Egyptian Data : EXCLUDED (Global Only)
#   Part A Target       : Wheat_Close_LogReturn  (with vs. without Corn & Sugar)
#   Part B Target       : Corn_Close_LogReturn   (with vs. without Wheat & Sugar)
#   Part C Target       : Sugar_Close_LogReturn  (with vs. without Wheat & Corn)
# =============================================================================
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
import warnings

warnings.filterwarnings("ignore")

print("Initiating Phase 5d: Ablation Study (Single Crop vs. All Three, Global Only)...\n")

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

# --- 2b. STRIP LOCAL EGYPTIAN FEATURES FROM ALL DATA ---
local_keywords = ['EGP', 'Inflation', 'CBE']
local_cols = [c for c in df_diff.columns if any(kw in c for kw in local_keywords)]
print(f"Removing local Egyptian features: {local_cols}\n")

train = train.drop(columns=local_cols)
val = val.drop(columns=local_cols)
test = test.drop(columns=local_cols)

def evaluate_commodity(target_prefix, other_keywords):
    print("=" * 60)
    print(f"  PREDICTING {target_prefix.upper()} PRICE (Global Only)")
    print("=" * 60)

    # All potential targets so we only drop the ones that leak
    all_targets = ['Wheat_Close', 'Corn_Close', 'Sugar_Close', 'Brent_Crude_Close', 'DXY_Close', 'EGP_USD_Close']
    
    # We drop ALL raw non-stationary prices, plus this specific commodity's LogReturn
    target_col = f'{target_prefix}_Close_LogReturn'
    drop_cols = all_targets + [target_col]
    cols_to_drop = [c for c in drop_cols if c in train.columns]

    y_train = train[target_col]
    y_val   = val[target_col]
    y_test  = test[target_col]

    # Full feature set
    X_train_full = train.drop(columns=cols_to_drop)
    X_val_full   = val.drop(columns=cols_to_drop)
    X_test_full  = test.drop(columns=cols_to_drop)

    # Single-Crop Only feature set
    single_crop_cols = [c for c in X_train_full.columns
                        if not any(kw in c for kw in other_keywords)]

    X_train_single = X_train_full[single_crop_cols]
    X_val_single   = X_val_full[single_crop_cols]
    X_test_single  = X_test_full[single_crop_cols]

    print(f"\n{target_prefix} — Full features:        {len(X_train_full.columns)} features")
    print(f"{target_prefix} — Single-only features: {len(X_train_single.columns)} features\n")

    # Train Full Model
    print(f"Training {target_prefix} Full Model (with {', '.join([k for k in other_keywords if k != 'Ratio'])} features)...")
    xgb_full = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.01, max_depth=3,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, early_stopping_rounds=50)
    xgb_full.fit(X_train_full, y_train,
                 eval_set=[(X_val_full, y_val)], verbose=False)

    # Train Single-Only Model
    print(f"Training {target_prefix}-Only Model (without other crops)...")
    xgb_single = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.01, max_depth=3,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, early_stopping_rounds=50)
    xgb_single.fit(X_train_single, y_train,
                   eval_set=[(X_val_single, y_val)], verbose=False)

    # Evaluate
    preds_full = xgb_full.predict(X_test_full)
    preds_single = xgb_single.predict(X_test_single)

    yesterday_price = df_master.loc[X_test_full.index, f'{target_prefix}_Close'].shift(1)
    yesterday_price.iloc[0] = df_master.loc['2024-12-31', f'{target_prefix}_Close']

    price_full = yesterday_price * np.exp(preds_full)
    price_single = yesterday_price * np.exp(preds_single)
    actual_price = df_master.loc[test.index, f'{target_prefix}_Close']

    rmse_full = root_mean_squared_error(actual_price, price_full)
    rmse_single = root_mean_squared_error(actual_price, price_single)

    print(f"\n=== {target_prefix.upper()} ABLATION RESULTS (RMSE) ===")
    print(f"Full Model (All Crops):     ${rmse_full:.2f}")
    print(f"{target_prefix}-Only Model:           ${rmse_single:.2f}\n")

    return rmse_full, rmse_single, actual_price, price_full, price_single

# Part A: Wheat
rmse_wheat_full, rmse_wheat_single, actual_wheat, price_wheat_full, price_wheat_single = evaluate_commodity('Wheat', ['Corn', 'Sugar', 'Ratio'])

# Part B: Corn
rmse_corn_full, rmse_corn_single, actual_corn, price_corn_full, price_corn_single = evaluate_commodity('Corn', ['Wheat', 'Sugar', 'Ratio'])

# Part C: Sugar
rmse_sugar_full, rmse_sugar_single, actual_sugar, price_sugar_full, price_sugar_single = evaluate_commodity('Sugar', ['Wheat', 'Corn', 'Ratio'])


# =====================================================================
#  VISUALIZATION
# =====================================================================

# --- Plot 1: RMSE Comparison Bar Chart (All 3 Commodities) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

def plot_bar(ax, models, scores, colors, title, ylabel):
    bars = ax.bar(models, scores, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=12)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + (yval*0.01),
                f'${yval:.2f}', ha='center', va='bottom', fontweight='bold')

plot_bar(axes[0], ['All Crops', 'Wheat-Only'], [rmse_wheat_full, rmse_wheat_single], ['saddlebrown', 'peru'], 'Wheat Forecast', 'RMSE (USD/Bu)')
plot_bar(axes[1], ['All Crops', 'Corn-Only'], [rmse_corn_full, rmse_corn_single], ['gold', 'khaki'], 'Corn Forecast', 'RMSE (USD/Bu)')
plot_bar(axes[2], ['All Crops', 'Sugar-Only'], [rmse_sugar_full, rmse_sugar_single], ['teal', 'mediumaquamarine'], 'Sugar Forecast', 'RMSE (USd/lb)')

plt.suptitle('Ablation Study: Single Crop vs. All Three Features (Global Only)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# --- Plot 2: Time-Series Overlays ---
def plot_ts(actual, price_full, price_single, rmse_full, rmse_single, name, color_full, color_single, unit):
    plt.figure(figsize=(16, 8))
    plt.plot(actual.index, actual, label=f'Actual {name} Price', color='black', linewidth=2)
    plt.plot(price_full.index, price_full,
             label=f'Full Model (All Crops) (RMSE: ${rmse_full:.2f})', color=color_full, linewidth=2)
    plt.plot(price_single.index, price_single,
             label=f'{name}-Only Model (RMSE: ${rmse_single:.2f})', color=color_single, alpha=0.7)
    plt.title(f'{name} Forecast: With vs. Without Other Crop Features (Global Only)', fontsize=16)
    plt.ylabel(f'{name} Price ({unit})', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()

plot_ts(actual_wheat, price_wheat_full, price_wheat_single, rmse_wheat_full, rmse_wheat_single, 'Wheat', 'saddlebrown', 'peru', 'USD/Bu')
plot_ts(actual_corn, price_corn_full, price_corn_single, rmse_corn_full, rmse_corn_single, 'Corn', 'gold', 'khaki', 'USD/Bu')
plot_ts(actual_sugar, price_sugar_full, price_sugar_single, rmse_sugar_full, rmse_sugar_single, 'Sugar', 'teal', 'mediumaquamarine', 'USd/lb')

print("\n--- SINGLE CROP VS ALL THREE ABLATION STUDY (GLOBAL ONLY) COMPLETE ---")
