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
# CEEMDAN-XGBoost Pipeline (ENRICHED)  --  Step 3b: Evaluation
# =============================================================================
# Each enriched sub-model predicts its own IMF component on the test set.
# The individual predictions are summed (multi-scale fusion) to reconstruct
# the final predicted gold price.
#
# This variant uses enriched sub-models that had access to exogenous
# macro-economic features during training, closing the information gap
# relative to the standard XGBoost architectures.
#
# Metrics: MAE, MAPE, MSE, RMSE, R-squared
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
print("  CEEMDAN-XGBoost (ENRICHED) -- Phase 3b: Evaluation")
print("=" * 60)

# --- 2. LOAD DATA ---
df_master = pd.read_csv(
    os.path.join(PROCESSED_DIR, '01_master_metals_dataset.csv'),
    index_col='Date', parse_dates=['Date']
)
df_imf = pd.read_csv(
    os.path.join(FINAL_DIR, '02b_ceemdan_enriched_dataset.csv'),
    index_col='Date', parse_dates=['Date']
)
print(f"[OK] Datasets loaded")

# Identify IMF target columns
imf_targets = [c for c in df_imf.columns
               if (c.startswith('IMF') or c == 'Residue')
               and '_' not in c]
print(f"  Components: {imf_targets}")

# Identify shared exogenous feature columns (same logic as training)
exog_features = [c for c in df_imf.columns
                 if not c.startswith('IMF')
                 and c != 'Residue'
                 and not c.startswith('Residue_')
                 and c != 'Gold_Close']

# --- 3. TEST SPLIT ---
test = df_imf[TEST_START:]
actual_prices = df_master.loc[test.index, 'Gold_Close']

# --- 4. LOAD ENRICHED SUB-MODELS & PREDICT PER COMPONENT ---
print("\nGenerating per-component predictions (enriched models)...")

component_preds = {}
for target in imf_targets:
    model_path = os.path.join(MODELS_DIR, f'ceemdan_enriched_xgb_{target.lower()}.json')
    if not os.path.exists(model_path):
        print(f"  [!]  Model not found for {target}: {model_path}")
        continue

    # Build feature set: IMF-own features + exogenous
    imf_own_features = [c for c in df_imf.columns if c.startswith(target + '_')]
    feature_cols = imf_own_features + exog_features
    X_test = test[feature_cols]

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    preds = model.predict(X_test)
    component_preds[target] = pd.Series(preds, index=test.index)
    print(f"  [OK] {target}  predicted ({len(preds)} days, {len(feature_cols)} features)")

# --- 5. MULTI-SCALE FUSION (Reconstruction) ---
print("\nReconstructing final price forecast via multi-scale fusion...")

# Sum all component predictions -> reconstructed gold price
pred_sum = pd.DataFrame(component_preds).sum(axis=1)
ceemdan_enriched_predictions = pred_sum  # already in price space (IMFs sum to price)

print(f"  Reconstruction range: ${ceemdan_enriched_predictions.min():.2f} - ${ceemdan_enriched_predictions.max():.2f}")

# --- 6. METRICS ---
print("\n" + "=" * 60)
print("  CEEMDAN-XGBoost (ENRICHED)  --  TEST SET RESULTS")
print("=" * 60)

mae  = mean_absolute_error(actual_prices, ceemdan_enriched_predictions)
mape = mean_absolute_percentage_error(actual_prices, ceemdan_enriched_predictions) * 100
mse  = mean_squared_error(actual_prices, ceemdan_enriched_predictions)
rmse = root_mean_squared_error(actual_prices, ceemdan_enriched_predictions)
r2   = r2_score(actual_prices, ceemdan_enriched_predictions)

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

# CEEMDAN-XGBoost Pure (IMF-only, variant A)
try:
    df_imf_pure = pd.read_csv(
        os.path.join(FINAL_DIR, '02_ceemdan_imfs_dataset.csv'),
        index_col='Date', parse_dates=['Date']
    )
    imf_targets_pure = [c for c in df_imf_pure.columns
                        if (c.startswith('IMF') or c == 'Residue')
                        and '_' not in c]
    test_pure = df_imf_pure[TEST_START:]
    pure_preds = {}
    for target in imf_targets_pure:
        mp = os.path.join(MODELS_DIR, f'ceemdan_xgb_{target.lower()}.json')
        if not os.path.exists(mp):
            continue
        fc = [c for c in df_imf_pure.columns if c.startswith(target + '_')]
        m = xgb.XGBRegressor()
        m.load_model(mp)
        pure_preds[target] = pd.Series(m.predict(test_pure[fc]), index=test_pure.index)
    pure_price = pd.DataFrame(pure_preds).sum(axis=1)
    rmse_pure = root_mean_squared_error(actual_prices, pure_price)
    mape_pure = mean_absolute_percentage_error(actual_prices, pure_price)
    mae_pure = mean_absolute_error(actual_prices, pure_price)
    mse_pure = mean_squared_error(actual_prices, pure_price)
    r2_pure   = r2_score(actual_prices, pure_price)
    comparison_models['CEEMDAN-XGB Pure (IMF-only)'] = {
        'preds': pure_price, 'rmse': rmse_pure, 'r2': r2_pure
    }
    print(f"  CEEMDAN Pure   ->  RMSE: ${rmse_pure:,.2f}  |  MAPE: {mape_pure:.4f}  |  MAE: {mae_pure:.2f}  |  MSE: {mse_pure:.2f}  |  R2: {r2_pure:.6f}")
except Exception as e:
    print(f"  [!]  Could not load CEEMDAN Pure: {e}")

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
    last_val_date = df_master[:TRAIN_VAL_END].index[-1]
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

# XGBoost B (Detrended)
try:
    df_b = pd.read_csv(
        os.path.join(FINAL_DIR, '01b_engineered_detrending_metals_dataset.csv'),
        index_col='Date', parse_dates=['Date']
    )
    test_b = df_b[TEST_START:]
    drop_cols_b = ['Gold_Close', 'Silver_Close', 'DXY_Close', 'SP500_Close',
                   'VIX_Close', 'EGP_USD_Close',
                   'Gold_Close_Trend', 'Gold_Close_Residual',
                   'Silver_Close_Trend', 'Silver_Close_Residual',
                   'DXY_Close_Trend', 'DXY_Close_Residual',
                   'SP500_Close_Trend', 'SP500_Close_Residual',
                   'VIX_Close_Trend', 'VIX_Close_Residual',
                   'EGP_USD_Close_Trend', 'EGP_USD_Close_Residual']
    X_test_b = test_b.drop(columns=drop_cols_b)

    xgb_b = xgb.XGBRegressor()
    xgb_b.load_model(os.path.join(MODELS_DIR, 'xgboost_b.json'))
    preds_resid_b = xgb_b.predict(X_test_b)
    trend_b = df_b.loc[X_test_b.index, 'Gold_Close_Trend']
    xgb_b_preds = trend_b + preds_resid_b

    rmse_b = root_mean_squared_error(actual_prices, xgb_b_preds)
    mape_b = mean_absolute_percentage_error(actual_prices, xgb_b_preds)
    mae_b = mean_absolute_error(actual_prices, xgb_b_preds)
    mse_b = mean_squared_error(actual_prices, xgb_b_preds)
    r2_b   = r2_score(actual_prices, xgb_b_preds)
    comparison_models['XGBoost B (Detrended)'] = {
        'preds': xgb_b_preds, 'rmse': rmse_b, 'r2': r2_b
    }
    print(f"  XGBoost B      ->  RMSE: ${rmse_b:,.2f}  |  MAPE: {mape_b:.4f}  |  MAE: {mae_b:.2f}  |  MSE: {mse_b:.2f}  |  R2: {r2_b:.6f}")
except Exception as e:
    print(f"  [!]  Could not load XGBoost B: {e}")

print(f"\n  CEEMDAN Enriched ->  RMSE: ${rmse:,.2f}  |  MAPE: {mape:.4f}  |  MAE: {mae:.2f}  |  MSE: {mse:.2f}  |  R2: {r2:.6f}")

# --- 8. VISUALISATIONS ---
print("\nGenerating plots...")

# ---- 8a. Full overlay comparison ----
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(actual_prices.index, actual_prices,
        label='Actual Gold Price', color='black', linewidth=2)
ax.plot(ceemdan_enriched_predictions.index, ceemdan_enriched_predictions,
        label=f'CEEMDAN-XGB Enriched (RMSE: ${rmse:,.2f})',
        color='darkorange', linewidth=1.5, alpha=0.85)

colors = {'CEEMDAN-XGB Pure (IMF-only)': ('gray', '--'),
          'XGBoost A (Log Returns)': ('blue', '--'),
          'XGBoost B (Detrended)': ('red', '--')}
for name, info in comparison_models.items():
    c, ls = colors.get(name, ('green', '--'))
    ax.plot(info['preds'].index, info['preds'],
            label=f'{name} (RMSE: ${info["rmse"]:,.2f})',
            linestyle=ls, linewidth=1, color=c, alpha=0.6)

ax.set_title('Gold Price Forecast: CEEMDAN-XGBoost Enriched vs All Models (Test Set)', fontsize=14)
ax.set_ylabel('Gold Price (USD)', fontsize=12)
ax.set_xlabel('Date', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
overlay_path = os.path.join(PLOTS_DIR, 'ceemdan_enriched_vs_all_models.png')
plt.savefig(overlay_path, dpi=150)
plt.show()
print(f"  Overlay plot saved to: {overlay_path}")

# ---- 8b. Per-component prediction breakdown ----
n_components = len(component_preds)
fig, axes = plt.subplots(n_components, 1, figsize=(16, 3 * n_components), sharex=True)
if n_components == 1:
    axes = [axes]

for ax, (comp_name, comp_pred) in zip(axes, component_preds.items()):
    actual_comp = test[comp_name]
    ax.plot(test.index, actual_comp, label=f'Actual {comp_name}', color='black', linewidth=0.8)
    ax.plot(test.index, comp_pred, label=f'Predicted {comp_name}',
            color='steelblue', linewidth=0.8, alpha=0.8)
    ax.set_ylabel(comp_name, fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

axes[0].set_title('Enriched Per-Component Predictions on Test Set', fontsize=13)
plt.xlabel('Date', fontsize=11)
plt.tight_layout()
comp_path = os.path.join(PLOTS_DIR, 'ceemdan_enriched_component_predictions.png')
plt.savefig(comp_path, dpi=150)
plt.show()
print(f"  Component plot saved to: {comp_path}")

# ---- 8c. Metrics summary table ----
print("\n" + "=" * 60)
print("  FINAL METRICS SUMMARY (ALL MODELS)")
print("=" * 60)

summary_rows = []
summary_rows.append({
    'Model': 'CEEMDAN-XGB Enriched',
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

# Save summary
summary_path = os.path.join(PLOTS_DIR, 'ceemdan_enriched_metrics_summary.csv')
summary_df.to_csv(summary_path)
print(f"\n  Metrics CSV saved to: {summary_path}")

print("\n" + "=" * 60)
print("  CEEMDAN-XGBoost (ENRICHED) EVALUATION COMPLETE")
print("=" * 60)
