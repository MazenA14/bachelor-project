import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error,
                             root_mean_squared_error,
                             r2_score)
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CEEMDAN-XGBoost Pipeline (STATIONARY)  --  Step 3c: Evaluation
# =============================================================================
# The stationary sub-models predict their respective stationary IMFs.
# These predictions are summed to reconstruct the predicted Log Return.
# Finally, the Log Return is reversed to calculate the absolute Gold Price.
#
# This completely bypasses XGBoost's inability to extrapolate, as the model
# operates entirely within the stationary, bounded domain of log returns.
# =============================================================================

# --- 1. CONFIGURATION ---
PROCESSED_DIR = '../data-extra-variables/02_processed/'
FINAL_DIR     = '../data-extra-variables/03_final/'
MODELS_DIR    = '../models-extra-variables/'
PLOTS_DIR     = '../plots-extra-variables/ceemdan/'

os.makedirs(PLOTS_DIR, exist_ok=True)

TEST_START    = '2025-01-01'
TRAIN_VAL_END = '2024-12-31'

print("=" * 60)
print("  CEEMDAN-XGBoost (STATIONARY) -- Phase 3c: Evaluation")
print("=" * 60)

# --- 2. LOAD DATA ---
df_master = pd.read_csv(
    os.path.join(PROCESSED_DIR, '01_master_metals_dataset.csv'),
    index_col='Date', parse_dates=['Date']
)
df_imf = pd.read_csv(
    os.path.join(FINAL_DIR, '02c_ceemdan_stationary_dataset.csv'),
    index_col='Date', parse_dates=['Date']
)
print(f"[OK] Datasets loaded")

# Identify IMF target columns
imf_targets = [c for c in df_imf.columns
               if (c.startswith('IMF') or c == 'Residue')
               and '_' not in c]

exog_features = [c for c in df_imf.columns
                 if not c.startswith('IMF')
                 and c != 'Residue'
                 and not c.startswith('Residue_')
                 and c not in ['Gold_Close', 'Gold_Close_LogReturn']]

# --- 3. TEST SPLIT ---
test = df_imf[TEST_START:]
actual_prices = df_master.loc[test.index, 'Gold_Close']

# --- 4. LOAD STATIONARY SUB-MODELS & PREDICT PER COMPONENT ---
print("\nGenerating per-component stationary predictions...")

component_preds = {}
for target in imf_targets:
    model_path = os.path.join(MODELS_DIR, f'ceemdan_stationary_xgb_{target.lower()}.json')
    if not os.path.exists(model_path):
        print(f"  [!]  Model not found for {target}: {model_path}")
        continue

    imf_own_features = [c for c in df_imf.columns if c.startswith(target + '_')]
    feature_cols = imf_own_features + exog_features
    X_test = test[feature_cols]

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    preds = model.predict(X_test)
    component_preds[target] = pd.Series(preds, index=test.index)
    print(f"  [OK] {target} predicted")

# --- 5. MULTI-SCALE FUSION & PRICE REVERSAL ---
print("\nReconstructing Log Returns and reversing to Absolute Price...")

# 1. Sum components to get predicted Log Return
predicted_log_returns = pd.DataFrame(component_preds).sum(axis=1)

# 2. Get yesterday's absolute price
yesterday_price = df_master.loc[test.index, 'Gold_Close'].shift(1)
# 3. Fill the very first day of the test set with the last day of the validation set
last_val_date = df_master[:TRAIN_VAL_END].index[-1]
yesterday_price.iloc[0] = df_master.loc[last_val_date, 'Gold_Close']

# 4. Reverse the log return formula: P_t = P_{t-1} * exp(R_t)
ceemdan_stationary_price_preds = yesterday_price * np.exp(predicted_log_returns)

print(f"  Reconstruction range: ${ceemdan_stationary_price_preds.min():.2f} - ${ceemdan_stationary_price_preds.max():.2f}")

# --- 6. METRICS ---
print("\n" + "=" * 60)
print("  CEEMDAN-XGBoost (STATIONARY)  --  TEST SET RESULTS")
print("=" * 60)

mae  = mean_absolute_error(actual_prices, ceemdan_stationary_price_preds)
mape = mean_absolute_percentage_error(actual_prices, ceemdan_stationary_price_preds) * 100
mse  = mean_squared_error(actual_prices, ceemdan_stationary_price_preds)
rmse = root_mean_squared_error(actual_prices, ceemdan_stationary_price_preds)
r2   = r2_score(actual_prices, ceemdan_stationary_price_preds)

print(f"  MAE   : ${mae:,.2f}")
print(f"  MAPE  : {mape:.4f}%")
print(f"  MSE   : {mse:,.2f}")
print(f"  RMSE  : ${rmse:,.2f}")
print(f"  R2    : {r2:.6f}")

# --- 7. COMPARISON WITH ALL EXISTING MODELS ---
print("\n" + "-" * 60)
print("  Cross-Model Comparison (All Architectures)")
print("-" * 60)

comparison_models = {}

# XGBoost A (Log Returns)
try:
    df_a = pd.read_csv(
        os.path.join(FINAL_DIR, '01a_engineered_differencing_metals_dataset.csv'),
        index_col='Date', parse_dates=['Date']
    )
    test_a = df_a[TEST_START:]
    drop_cols_a = ['Gold_Close', 'Silver_Close', 'DXY_Close', 'SP500_Close',
                   'VIX_Close', 'EGP_USD_Close',
                   'Gold_Close_LogReturn', 'Silver_Close_LogReturn',
                   'DXY_Close_LogReturn', 'SP500_Close_LogReturn',
                   'VIX_Close_LogReturn', 'EGP_USD_Close_LogReturn']
    X_test_a = test_a.drop(columns=drop_cols_a)

    xgb_a = xgb.XGBRegressor()
    xgb_a.load_model(os.path.join(MODELS_DIR, 'xgboost_a.json'))
    preds_log_a = xgb_a.predict(X_test_a)

    yesterday_price_a = df_master.loc[X_test_a.index, 'Gold_Close'].shift(1)
    yesterday_price_a.iloc[0] = df_master.loc[last_val_date, 'Gold_Close']
    xgb_a_preds = yesterday_price_a * np.exp(preds_log_a)

    rmse_a = root_mean_squared_error(actual_prices, xgb_a_preds)
    mape_a = mean_absolute_percentage_error(actual_prices, xgb_a_preds)
    mae_a = mean_absolute_error(actual_prices, xgb_a_preds)
    mse_a = mean_squared_error(actual_prices, xgb_a_preds)
    r2_a   = r2_score(actual_prices, xgb_a_preds)
    comparison_models['XGBoost A (Log Returns)'] = {
        'preds': xgb_a_preds, 'rmse': rmse_a, 'r2': r2_a
    }
    print(f"  XGBoost A      ->  RMSE: ${rmse_a:,.2f}  |  MAPE: {mape_a:.4f}  |  MAE: {mae_a:.2f}  |  MSE: {mse_a:.2f}  |  R2: {r2_a:.6f}")
except Exception as e:
    print(f"  [!]  Could not load XGBoost A: {e}")

# CEEMDAN Enriched (Absolute Price Decomp)
try:
    df_imf_en = pd.read_csv(
        os.path.join(FINAL_DIR, '02b_ceemdan_enriched_dataset.csv'),
        index_col='Date', parse_dates=['Date']
    )
    imf_targets_en = [c for c in df_imf_en.columns if (c.startswith('IMF') or c == 'Residue') and '_' not in c]
    test_en = df_imf_en[TEST_START:]
    exog_en = [c for c in df_imf_en.columns if not c.startswith('IMF') and c != 'Residue' and not c.startswith('Residue_') and c != 'Gold_Close']
    
    en_preds = {}
    for target in imf_targets_en:
        mp = os.path.join(MODELS_DIR, f'ceemdan_enriched_xgb_{target.lower()}.json')
        if not os.path.exists(mp): continue
        fc = [c for c in df_imf_en.columns if c.startswith(target + '_')] + exog_en
        m = xgb.XGBRegressor()
        m.load_model(mp)
        en_preds[target] = pd.Series(m.predict(test_en[fc]), index=test_en.index)
    en_price = pd.DataFrame(en_preds).sum(axis=1)
    rmse_en = root_mean_squared_error(actual_prices, en_price)
    mape_en = mean_absolute_percentage_error(actual_prices, en_price)
    mae_en = mean_absolute_error(actual_prices, en_price)
    mse_en = mean_squared_error(actual_prices, en_price)
    r2_en   = r2_score(actual_prices, en_price)
    comparison_models['CEEMDAN Enriched (Abs Price)'] = {
        'preds': en_price, 'rmse': rmse_en, 'r2': r2_en
    }
    print(f"  CEEMDAN Enriched ->  RMSE: ${rmse_en:,.2f}  |  MAPE: {mape_en:.4f}  |  MAE: {mae_en:.2f}  |  MSE: {mse_en:.2f}  |  R2: {r2_en:.6f}")
except Exception as e:
    print(f"  [!]  Could not load CEEMDAN Enriched: {e}")

print(f"\n  CEEMDAN Stationary ->  RMSE: ${rmse:,.2f}  |  MAPE: {mape:.4f}  |  MAE: {mae:.2f}  |  MSE: {mse:.2f}  |  R2: {r2:.6f}")

# --- 8. VISUALISATIONS ---
print("\nGenerating plots...")

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(actual_prices.index, actual_prices,
        label='Actual Gold Price', color='black', linewidth=2)
ax.plot(ceemdan_stationary_price_preds.index, ceemdan_stationary_price_preds,
        label=f'CEEMDAN-XGB Stationary (RMSE: ${rmse:,.2f})',
        color='darkorange', linewidth=2, alpha=0.9)

colors = {'XGBoost A (Log Returns)': ('blue', '--'),
          'CEEMDAN Enriched (Abs Price)': ('red', '--')}
for name, info in comparison_models.items():
    c, ls = colors.get(name, ('green', '--'))
    ax.plot(info['preds'].index, info['preds'],
            label=f'{name} (RMSE: ${info["rmse"]:,.2f})',
            linestyle=ls, linewidth=1.5, color=c, alpha=0.7)

ax.set_title('Gold Price Forecast: CEEMDAN Stationary vs Architecture A (Test Set)', fontsize=15)
ax.set_ylabel('Gold Price (USD)', fontsize=12)
ax.set_xlabel('Date', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
overlay_path = os.path.join(PLOTS_DIR, 'ceemdan_stationary_vs_models.png')
plt.savefig(overlay_path, dpi=150)
plt.show()
print(f"  Overlay plot saved to: {overlay_path}")

# ---- 8b. Metrics summary table ----
print("\n" + "=" * 60)
print("  FINAL METRICS SUMMARY")
print("=" * 60)

summary_rows = []
summary_rows.append({
    'Model': 'CEEMDAN-XGB Stationary',
    'MAE': mae, 'MAPE (%)': mape, 'MSE': mse, 'RMSE': rmse, 'R2': r2
})
for name, info in comparison_models.items():
    ap = actual_prices.loc[info['preds'].index]
    summary_rows.append({
        'Model': name,
        'MAE': mean_absolute_error(ap, info['preds']),
        'MAPE (%)': mean_absolute_percentage_error(ap, info['preds']) * 100,
        'MSE': mean_squared_error(ap, info['preds']),
        'RMSE': info['rmse'],
        'R2': info['r2'],
    })

summary_df = pd.DataFrame(summary_rows).set_index('Model')
print(summary_df.to_string(float_format=lambda x: f'{x:.4f}'))

summary_path = os.path.join(PLOTS_DIR, 'ceemdan_stationary_metrics_summary.csv')
summary_df.to_csv(summary_path)
print(f"\n  Metrics CSV saved to: {summary_path}")

print("\n" + "=" * 60)
print("  CEEMDAN-XGBoost (STATIONARY) EVALUATION COMPLETE")
print("=" * 60)
