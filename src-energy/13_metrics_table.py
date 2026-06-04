import os
import pandas as pd
import numpy as np
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

print("Generating Metrics Table for Brent Crude Oil and Natural Gas Models...\n")

# --- CONFIGURATION ---
PROCESSED_DIR = '../data-energy/02_processed/'
FINAL_DIR     = '../data-energy/03_final/'
MODELS_DIR    = '../models-energy/'
TRAIN_END     = '2023-12-31'
VAL_START     = '2024-01-01'
VAL_END       = '2024-12-31'
TEST_START    = '2025-01-01'

# --- DATA LOADING ---
df_master  = pd.read_csv(os.path.join(PROCESSED_DIR, '01_master_energy_dataset.csv'),                    index_col='Date', parse_dates=['Date'])
df_eng_a   = pd.read_csv(os.path.join(FINAL_DIR, '01a_engineered_differencing_energy_dataset.csv'),      index_col='Date', parse_dates=['Date'])
df_eng_b   = pd.read_csv(os.path.join(FINAL_DIR, '01b_engineered_detrending_energy_dataset.csv'),        index_col='Date', parse_dates=['Date'])
df_base_a  = pd.read_csv(os.path.join(FINAL_DIR, '01a_differencing_energy_dataset.csv'),                 index_col='Date', parse_dates=['Date'])
df_base_b  = pd.read_csv(os.path.join(FINAL_DIR, '01b_detrending_energy_dataset.csv'),                   index_col='Date', parse_dates=['Date'])

# Split helpers
def split(df):
    return df[:TRAIN_END], df[VAL_START:VAL_END], df[TEST_START:]

train_eng_a,  val_eng_a,  test_eng_a  = split(df_eng_a)
train_eng_b,  val_eng_b,  test_eng_b  = split(df_eng_b)
train_base_a, val_base_a, test_base_a = split(df_base_a)
train_base_b, val_base_b, test_base_b = split(df_base_b)

# Drop columns for Arch A (differencing)
DROP_A = [
    'Brent_Crude_Close', 'Natural_Gas_Close', 'DXY_Close', 'VIX_Close', 'SP500_Close', 'EGP_USD_Close',
    'Brent_Crude_Close_LogReturn', 'Natural_Gas_Close_LogReturn', 'DXY_Close_LogReturn',
    'VIX_Close_LogReturn', 'SP500_Close_LogReturn', 'EGP_USD_Close_LogReturn',
    'US_10Yr_Yield_Diff', 'Egypt_Inflation_YoY', 'CBE_Interest_Rate'
]

# Drop columns for Arch B (detrending)
DROP_B = [
    'Brent_Crude_Close', 'Natural_Gas_Close', 'DXY_Close', 'VIX_Close', 'SP500_Close', 'EGP_USD_Close',
    'Brent_Crude_Close_Trend', 'Brent_Crude_Close_Residual',
    'Natural_Gas_Close_Trend', 'Natural_Gas_Close_Residual',
    'DXY_Close_Trend', 'DXY_Close_Residual',
    'VIX_Close_Trend', 'VIX_Close_Residual',
    'SP500_Close_Trend', 'SP500_Close_Residual',
    'EGP_USD_Close_Trend', 'EGP_USD_Close_Residual',
    'Egypt_Inflation_YoY', 'CBE_Interest_Rate'
]

def make_X(df, drop_cols):
    existing = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=existing)

# Feature matrices
X_tr_ea = make_X(train_eng_a, DROP_A);   X_va_ea = make_X(val_eng_a, DROP_A);   X_te_ea = make_X(test_eng_a, DROP_A)
X_tr_eb = make_X(train_eng_b, DROP_B);   X_va_eb = make_X(val_eng_b, DROP_B);   X_te_eb = make_X(test_eng_b, DROP_B)
X_tr_ba = make_X(train_base_a, DROP_A);  X_va_ba = make_X(val_base_a, DROP_A);  X_te_ba = make_X(test_base_a, DROP_A)
X_tr_bb = make_X(train_base_b, DROP_B);  X_va_bb = make_X(val_base_b, DROP_B);  X_te_bb = make_X(test_base_b, DROP_B)

# Actual prices
actual_oil = df_master.loc[test_eng_a.index, 'Brent_Crude_Close']
actual_gas = df_master.loc[test_eng_a.index, 'Natural_Gas_Close']

# Last known prices before test set (for Arch A reversal)
last_val_date       = df_master[:VAL_END].index[-1]
last_oil_price      = df_master.loc[last_val_date, 'Brent_Crude_Close']
last_gas_price      = df_master.loc[last_val_date, 'Natural_Gas_Close']

def compute_metrics(actual, predicted):
    rmse = root_mean_squared_error(actual, predicted)
    mae  = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)
    mse  = mean_squared_error(actual, predicted)
    r2   = r2_score(actual, predicted)
    return rmse, mae, mape, mse, r2

# ============================================================
# PANEL A: BRENT CRUDE OIL
# ============================================================
print("--- Computing Brent Crude Oil (Panel A) metrics ---")

# 1. Oil XGBoost Arch A (Full)
oil_xgb_a_full = xgb.XGBRegressor()
oil_xgb_a_full.load_model(os.path.join(MODELS_DIR, 'xgboost_a.json'))
preds_log = oil_xgb_a_full.predict(X_te_ea)
yp = df_master.loc[X_te_ea.index, 'Brent_Crude_Close'].shift(1)
yp.iloc[0] = last_oil_price
oil_arch_a_full_preds = yp * np.exp(preds_log)
oil_arch_a_full = compute_metrics(actual_oil, oil_arch_a_full_preds)

# 2. Oil XGBoost Arch A (No Extras)
oil_xgb_a_ne = xgb.XGBRegressor()
oil_xgb_a_ne.load_model(os.path.join(MODELS_DIR, 'xgboost_a_no_extras.json'))
preds_log = oil_xgb_a_ne.predict(X_te_ba)
yp = df_master.loc[X_te_ba.index, 'Brent_Crude_Close'].shift(1)
yp.iloc[0] = last_oil_price
oil_arch_a_ne_preds = yp * np.exp(preds_log)
oil_arch_a_ne = compute_metrics(actual_oil, oil_arch_a_ne_preds)

# 3. Oil XGBoost Arch B (Full)
oil_xgb_b_full = xgb.XGBRegressor()
oil_xgb_b_full.load_model(os.path.join(MODELS_DIR, 'xgboost_b.json'))
preds_resid = oil_xgb_b_full.predict(X_te_eb)
trend = df_eng_b.loc[X_te_eb.index, 'Brent_Crude_Close_Trend']
oil_arch_b_full_preds = trend + preds_resid
oil_arch_b_full = compute_metrics(actual_oil, oil_arch_b_full_preds)

# 4. Oil XGBoost Arch B (No Extras)
oil_xgb_b_ne = xgb.XGBRegressor()
oil_xgb_b_ne.load_model(os.path.join(MODELS_DIR, 'xgboost_b_no_extras.json'))
preds_resid = oil_xgb_b_ne.predict(X_te_bb)
trend = df_base_b.loc[X_te_bb.index, 'Brent_Crude_Close_Trend']
oil_arch_b_ne_preds = trend + preds_resid
oil_arch_b_ne = compute_metrics(actual_oil, oil_arch_b_ne_preds)

# 5. Oil ARIMA(5,1,0) Static
arima_oil_static = ARIMAResults.load(os.path.join(MODELS_DIR, 'arima_baseline.pkl'))
arima_oil_static_preds = arima_oil_static.forecast(steps=len(actual_oil))
arima_oil_static_preds.index = actual_oil.index
oil_arima_static = compute_metrics(actual_oil, arima_oil_static_preds)

# 6. Oil ARIMA(5,1,0) Walk-Forward
print("  Running Oil ARIMA Walk-Forward...")
history_oil = list(df_master['Brent_Crude_Close'][:VAL_END])
oil_wf_preds = []
for t in range(len(actual_oil)):
    m = ARIMA(history_oil, order=(5, 1, 0)).fit()
    oil_wf_preds.append(m.forecast()[0])
    history_oil.append(actual_oil.iloc[t])
    if (t + 1) % 50 == 0:
        print(f"    Oil WF: {t+1}/{len(actual_oil)}")
oil_arima_wf_preds = pd.Series(oil_wf_preds, index=actual_oil.index)
oil_arima_wf = compute_metrics(actual_oil, oil_arima_wf_preds)

# ============================================================
# PANEL B: NATURAL GAS
# ============================================================
print("\n--- Computing Natural Gas (Panel B) metrics ---")

# 1. Gas XGBoost Arch A (Full)
gas_xgb_a_full = xgb.XGBRegressor()
gas_xgb_a_full.load_model(os.path.join(MODELS_DIR, 'xgboost_a_gas.json'))
preds_log = gas_xgb_a_full.predict(X_te_ea)
yp = df_master.loc[X_te_ea.index, 'Natural_Gas_Close'].shift(1)
yp.iloc[0] = last_gas_price
gas_arch_a_full_preds = yp * np.exp(preds_log)
gas_arch_a_full = compute_metrics(actual_gas, gas_arch_a_full_preds)

# 2. Gas XGBoost Arch A (No Extras)
gas_xgb_a_ne = xgb.XGBRegressor()
gas_xgb_a_ne.load_model(os.path.join(MODELS_DIR, 'xgboost_a_no_extras_gas.json'))
preds_log = gas_xgb_a_ne.predict(X_te_ba)
yp = df_master.loc[X_te_ba.index, 'Natural_Gas_Close'].shift(1)
yp.iloc[0] = last_gas_price
gas_arch_a_ne_preds = yp * np.exp(preds_log)
gas_arch_a_ne = compute_metrics(actual_gas, gas_arch_a_ne_preds)

# 3. Gas XGBoost Arch B (Full)
gas_xgb_b_full = xgb.XGBRegressor()
gas_xgb_b_full.load_model(os.path.join(MODELS_DIR, 'xgboost_b_gas.json'))
preds_resid = gas_xgb_b_full.predict(X_te_eb)
trend = df_eng_b.loc[X_te_eb.index, 'Natural_Gas_Close_Trend']
gas_arch_b_full_preds = trend + preds_resid
gas_arch_b_full = compute_metrics(actual_gas, gas_arch_b_full_preds)

# 4. Gas XGBoost Arch B (No Extras)
gas_xgb_b_ne = xgb.XGBRegressor()
gas_xgb_b_ne.load_model(os.path.join(MODELS_DIR, 'xgboost_b_no_extras_gas.json'))
preds_resid = gas_xgb_b_ne.predict(X_te_bb)
trend = df_base_b.loc[X_te_bb.index, 'Natural_Gas_Close_Trend']
gas_arch_b_ne_preds = trend + preds_resid
gas_arch_b_ne = compute_metrics(actual_gas, gas_arch_b_ne_preds)

# 5. Gas ARIMA(5,1,0) Static
arima_gas_static = ARIMAResults.load(os.path.join(MODELS_DIR, 'arima_baseline_gas.pkl'))
arima_gas_static_preds = arima_gas_static.forecast(steps=len(actual_gas))
arima_gas_static_preds.index = actual_gas.index
gas_arima_static = compute_metrics(actual_gas, arima_gas_static_preds)

# 6. Gas ARIMA(5,1,0) Walk-Forward
print("  Running Gas ARIMA Walk-Forward...")
history_gas = list(df_master['Natural_Gas_Close'][:VAL_END])
gas_wf_preds = []
for t in range(len(actual_gas)):
    m = ARIMA(history_gas, order=(5, 1, 0)).fit()
    gas_wf_preds.append(m.forecast()[0])
    history_gas.append(actual_gas.iloc[t])
    if (t + 1) % 50 == 0:
        print(f"    Gas WF: {t+1}/{len(actual_gas)}")
gas_arima_wf_preds = pd.Series(gas_wf_preds, index=actual_gas.index)
gas_arima_wf = compute_metrics(actual_gas, gas_arima_wf_preds)

# ============================================================
# PRINT TABLE
# ============================================================
header = f"{'Model / Condition':<35} {'RMSE':>10} {'MAE':>10} {'MAPE':>10} {'MSE':>14} {'R²':>8}"
sep    = "-" * len(header)

rows_oil = [
    ("XGBoost: Arch A (Full)",         oil_arch_a_full),
    ("XGBoost: Arch A (No Extras)",    oil_arch_a_ne),
    ("XGBoost: Arch B (Full)",         oil_arch_b_full),
    ("XGBoost: Arch B (No Extras)",    oil_arch_b_ne),
    ("ARIMA(5,1,0) Static",            oil_arima_static),
    ("ARIMA(5,1,0) Walk-Forward",      oil_arima_wf),
]

rows_gas = [
    ("XGBoost: Arch A (Full)",         gas_arch_a_full),
    ("XGBoost: Arch A (No Extras)",    gas_arch_a_ne),
    ("XGBoost: Arch B (Full)",         gas_arch_b_full),
    ("XGBoost: Arch B (No Extras)",    gas_arch_b_ne),
    ("ARIMA(5,1,0) Static",            gas_arima_static),
    ("ARIMA(5,1,0) Walk-Forward",      gas_arima_wf),
]

print("\n" + "=" * len(header))
print(header)
print("=" * len(header))
print(f"Panel A: Brent Crude Oil (BZ=F)")
print(sep)
for name, (rmse, mae, mape, mse, r2) in rows_oil:
    print(f"{name:<35} {rmse:>10.4f} {mae:>10.4f} {mape:>10.4f} {mse:>14.4f} {r2:>8.4f}")
print(sep)
print(f"Panel B: Natural Gas (NG=F)")
print(sep)
for name, (rmse, mae, mape, mse, r2) in rows_gas:
    print(f"{name:<35} {rmse:>10.4f} {mae:>10.4f} {mape:>10.4f} {mse:>14.4f} {r2:>8.4f}")
print("=" * len(header))

# Also save to CSV
all_rows = (
    [("Panel A: Brent Crude Oil (BZ=F)", name, *vals) for name, vals in rows_oil] +
    [("Panel B: Natural Gas (NG=F)", name, *vals) for name, vals in rows_gas]
)
df_out = pd.DataFrame(all_rows, columns=["Panel", "Model / Condition", "RMSE", "MAE", "MAPE", "MSE", "R2"])
out_path = '../data-energy/metrics_table.csv'
df_out.to_csv(out_path, index=False)
print(f"\nTable saved to: {out_path}")
