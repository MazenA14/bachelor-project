import os
import pandas as pd
import numpy as np
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

print("Generating Metrics Table for Gold and Silver Models...\n")

# --- CONFIGURATION ---
PROCESSED_DIR = '../data-extra-variables/02_processed/'
FINAL_DIR     = '../data-extra-variables/03_final/'
MODELS_DIR    = '../models-extra-variables/'
TRAIN_END     = '2023-12-31'
VAL_START     = '2024-01-01'
VAL_END       = '2024-12-31'
TEST_START    = '2025-01-01'

# --- DATA LOADING ---
df_master  = pd.read_csv(os.path.join(PROCESSED_DIR, '01_master_metals_dataset.csv'),  index_col='Date', parse_dates=['Date'])
df_eng_a   = pd.read_csv(os.path.join(FINAL_DIR, '01a_engineered_differencing_metals_dataset.csv'), index_col='Date', parse_dates=['Date'])
df_eng_b   = pd.read_csv(os.path.join(FINAL_DIR, '01b_engineered_detrending_metals_dataset.csv'),   index_col='Date', parse_dates=['Date'])
df_base_a  = pd.read_csv(os.path.join(FINAL_DIR, '01a_differencing_metals_dataset.csv'),            index_col='Date', parse_dates=['Date'])
df_base_b  = pd.read_csv(os.path.join(FINAL_DIR, '01b_detrending_metals_dataset.csv'),              index_col='Date', parse_dates=['Date'])

# Split helpers
def split(df):
    return df[:TRAIN_END], df[VAL_START:VAL_END], df[TEST_START:]

train_eng_a,  val_eng_a,  test_eng_a  = split(df_eng_a)
train_eng_b,  val_eng_b,  test_eng_b  = split(df_eng_b)
train_base_a, val_base_a, test_base_a = split(df_base_a)
train_base_b, val_base_b, test_base_b = split(df_base_b)

# Drop columns for Arch A (differencing)
DROP_A = ['Gold_Close', 'Silver_Close', 'DXY_Close', 'SP500_Close', 'VIX_Close', 'EGP_USD_Close',
          'Gold_Close_LogReturn', 'Silver_Close_LogReturn', 'DXY_Close_LogReturn',
          'SP500_Close_LogReturn', 'VIX_Close_LogReturn', 'EGP_USD_Close_LogReturn']

# Drop columns for Arch B (detrending)
DROP_B = ['Gold_Close', 'Silver_Close', 'DXY_Close', 'SP500_Close', 'VIX_Close', 'EGP_USD_Close',
          'Gold_Close_Trend', 'Gold_Close_Residual', 'Silver_Close_Trend', 'Silver_Close_Residual',
          'DXY_Close_Trend', 'DXY_Close_Residual', 'SP500_Close_Trend', 'SP500_Close_Residual',
          'VIX_Close_Trend', 'VIX_Close_Residual', 'EGP_USD_Close_Trend', 'EGP_USD_Close_Residual']

def make_X(df, drop_cols):
    existing = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=existing)

# Feature matrices
X_tr_ea = make_X(train_eng_a, DROP_A);   X_va_ea = make_X(val_eng_a, DROP_A);   X_te_ea = make_X(test_eng_a, DROP_A)
X_tr_eb = make_X(train_eng_b, DROP_B);   X_va_eb = make_X(val_eng_b, DROP_B);   X_te_eb = make_X(test_eng_b, DROP_B)
X_tr_ba = make_X(train_base_a, DROP_A);  X_va_ba = make_X(val_base_a, DROP_A);  X_te_ba = make_X(test_base_a, DROP_A)
X_tr_bb = make_X(train_base_b, DROP_B);  X_va_bb = make_X(val_base_b, DROP_B);  X_te_bb = make_X(test_base_b, DROP_B)

# Actual prices
actual_gold   = df_master.loc[test_eng_a.index, 'Gold_Close']
actual_silver = df_master.loc[test_eng_a.index, 'Silver_Close']

# Last known prices before test set (for Arch A reversal)
last_val_date = df_master[:VAL_END].index[-1]
last_gold_price   = df_master.loc[last_val_date, 'Gold_Close']
last_silver_price = df_master.loc[last_val_date, 'Silver_Close']

def compute_metrics(actual, predicted):
    rmse = root_mean_squared_error(actual, predicted)
    mae  = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)
    mse  = mean_squared_error(actual, predicted)
    r2   = r2_score(actual, predicted)
    return rmse, mae, mape, mse, r2

# ============================================================
# PANEL A: GOLD
# ============================================================
print("--- Computing Gold (Panel A) metrics ---")

# 1. Gold XGBoost Arch A (Full) — load saved model
gold_xgb_a_full = xgb.XGBRegressor()
gold_xgb_a_full.load_model(os.path.join(MODELS_DIR, 'xgboost_a.json'))
preds_log = gold_xgb_a_full.predict(X_te_ea)
yp = df_master.loc[X_te_ea.index, 'Gold_Close'].shift(1)
yp.iloc[0] = last_gold_price
gold_arch_a_full_preds = yp * np.exp(preds_log)
gold_arch_a_full = compute_metrics(actual_gold, gold_arch_a_full_preds)

# 2. Gold XGBoost Arch A (No Extras) — load saved model
gold_xgb_a_ne = xgb.XGBRegressor()
gold_xgb_a_ne.load_model(os.path.join(MODELS_DIR, 'xgboost_a_no_extras.json'))
preds_log = gold_xgb_a_ne.predict(X_te_ba)
yp = df_master.loc[X_te_ba.index, 'Gold_Close'].shift(1)
yp.iloc[0] = last_gold_price
gold_arch_a_ne_preds = yp * np.exp(preds_log)
gold_arch_a_ne = compute_metrics(actual_gold, gold_arch_a_ne_preds)

# 3. Gold XGBoost Arch B (Full) — load saved model
gold_xgb_b_full = xgb.XGBRegressor()
gold_xgb_b_full.load_model(os.path.join(MODELS_DIR, 'xgboost_b.json'))
preds_resid = gold_xgb_b_full.predict(X_te_eb)
trend = df_eng_b.loc[X_te_eb.index, 'Gold_Close_Trend']
gold_arch_b_full_preds = trend + preds_resid
gold_arch_b_full = compute_metrics(actual_gold, gold_arch_b_full_preds)

# 4. Gold XGBoost Arch B (No Extras) — load saved model
gold_xgb_b_ne = xgb.XGBRegressor()
gold_xgb_b_ne.load_model(os.path.join(MODELS_DIR, 'xgboost_b_no_extras.json'))
preds_resid = gold_xgb_b_ne.predict(X_te_bb)
trend = df_base_b.loc[X_te_bb.index, 'Gold_Close_Trend']
gold_arch_b_ne_preds = trend + preds_resid
gold_arch_b_ne = compute_metrics(actual_gold, gold_arch_b_ne_preds)

# 5. Gold ARIMA(5,1,0) Static — load saved model
arima_gold_static = ARIMAResults.load(os.path.join(MODELS_DIR, 'arima_baseline.pkl'))
arima_gold_static_preds = arima_gold_static.forecast(steps=len(actual_gold))
arima_gold_static_preds.index = actual_gold.index
gold_arima_static = compute_metrics(actual_gold, arima_gold_static_preds)

# 6. Gold ARIMA(5,1,0) Walk-Forward
print("  Running Gold ARIMA Walk-Forward...")
history_gold = list(df_master['Gold_Close'][:VAL_END])
gold_wf_preds = []
for t in range(len(actual_gold)):
    m = ARIMA(history_gold, order=(5, 1, 0)).fit()
    gold_wf_preds.append(m.forecast()[0])
    history_gold.append(actual_gold.iloc[t])
    if (t + 1) % 50 == 0:
        print(f"    Gold WF: {t+1}/{len(actual_gold)}")
gold_arima_wf_preds = pd.Series(gold_wf_preds, index=actual_gold.index)
gold_arima_wf = compute_metrics(actual_gold, gold_arima_wf_preds)

# ============================================================
# PANEL B: SILVER
# ============================================================
print("\n--- Computing Silver (Panel B) metrics ---")

xgb_params = dict(n_estimators=1000, learning_rate=0.05, max_depth=5, random_state=42, early_stopping_rounds=50)

# 1. Silver XGBoost Arch A (Full) — train on Silver_Close_LogReturn
print("  Training Silver Arch A (Full)...")
silver_xgb_a_full = xgb.XGBRegressor(**xgb_params)
silver_xgb_a_full.fit(X_tr_ea, train_eng_a['Silver_Close_LogReturn'],
                      eval_set=[(X_va_ea, val_eng_a['Silver_Close_LogReturn'])], verbose=False)
preds_log = silver_xgb_a_full.predict(X_te_ea)
yp = df_master.loc[X_te_ea.index, 'Silver_Close'].shift(1)
yp.iloc[0] = last_silver_price
silver_arch_a_full_preds = yp * np.exp(preds_log)
silver_arch_a_full = compute_metrics(actual_silver, silver_arch_a_full_preds)

# 2. Silver XGBoost Arch A (No Extras) — train on Silver_Close_LogReturn
print("  Training Silver Arch A (No Extras)...")
silver_xgb_a_ne = xgb.XGBRegressor(**xgb_params)
silver_xgb_a_ne.fit(X_tr_ba, train_base_a['Silver_Close_LogReturn'],
                    eval_set=[(X_va_ba, val_base_a['Silver_Close_LogReturn'])], verbose=False)
preds_log = silver_xgb_a_ne.predict(X_te_ba)
yp = df_master.loc[X_te_ba.index, 'Silver_Close'].shift(1)
yp.iloc[0] = last_silver_price
silver_arch_a_ne_preds = yp * np.exp(preds_log)
silver_arch_a_ne = compute_metrics(actual_silver, silver_arch_a_ne_preds)

# 3. Silver XGBoost Arch B (Full) — train on Silver_Close_Residual
print("  Training Silver Arch B (Full)...")
silver_xgb_b_full = xgb.XGBRegressor(**xgb_params)
silver_xgb_b_full.fit(X_tr_eb, train_eng_b['Silver_Close_Residual'],
                      eval_set=[(X_va_eb, val_eng_b['Silver_Close_Residual'])], verbose=False)
preds_resid = silver_xgb_b_full.predict(X_te_eb)
trend = df_eng_b.loc[X_te_eb.index, 'Silver_Close_Trend']
silver_arch_b_full_preds = trend + preds_resid
silver_arch_b_full = compute_metrics(actual_silver, silver_arch_b_full_preds)

# 4. Silver XGBoost Arch B (No Extras) — train on Silver_Close_Residual
print("  Training Silver Arch B (No Extras)...")
silver_xgb_b_ne = xgb.XGBRegressor(**xgb_params)
silver_xgb_b_ne.fit(X_tr_bb, train_base_b['Silver_Close_Residual'],
                    eval_set=[(X_va_bb, val_base_b['Silver_Close_Residual'])], verbose=False)
preds_resid = silver_xgb_b_ne.predict(X_te_bb)
trend = df_base_b.loc[X_te_bb.index, 'Silver_Close_Trend']
silver_arch_b_ne_preds = trend + preds_resid
silver_arch_b_ne = compute_metrics(actual_silver, silver_arch_b_ne_preds)

# 5. Silver ARIMA(5,1,0) Static
print("  Fitting Silver ARIMA Static...")
silver_arima_train = df_master['Silver_Close'][:VAL_END]
silver_arima_model = ARIMA(silver_arima_train, order=(5, 1, 0)).fit()
silver_arima_static_preds = silver_arima_model.forecast(steps=len(actual_silver))
silver_arima_static_preds.index = actual_silver.index
silver_arima_static = compute_metrics(actual_silver, silver_arima_static_preds)

# 6. Silver ARIMA(5,1,0) Walk-Forward
print("  Running Silver ARIMA Walk-Forward...")
history_silver = list(df_master['Silver_Close'][:VAL_END])
silver_wf_preds = []
for t in range(len(actual_silver)):
    m = ARIMA(history_silver, order=(5, 1, 0)).fit()
    silver_wf_preds.append(m.forecast()[0])
    history_silver.append(actual_silver.iloc[t])
    if (t + 1) % 50 == 0:
        print(f"    Silver WF: {t+1}/{len(actual_silver)}")
silver_arima_wf_preds = pd.Series(silver_wf_preds, index=actual_silver.index)
silver_arima_wf = compute_metrics(actual_silver, silver_arima_wf_preds)

# ============================================================
# PRINT TABLE
# ============================================================
header = f"{'Model / Condition':<35} {'RMSE':>10} {'MAE':>10} {'MAPE':>10} {'MSE':>14} {'R²':>8}"
sep    = "-" * len(header)

rows_gold = [
    ("XGBoost: Arch A (Full)",         gold_arch_a_full),
    ("XGBoost: Arch A (No Extras)",    gold_arch_a_ne),
    ("XGBoost: Arch B (Full)",         gold_arch_b_full),
    ("XGBoost: Arch B (No Extras)",    gold_arch_b_ne),
    ("ARIMA(5,1,0) Static",            gold_arima_static),
    ("ARIMA(5,1,0) Walk-Forward",      gold_arima_wf),
]

rows_silver = [
    ("XGBoost: Arch A (Full)",         silver_arch_a_full),
    ("XGBoost: Arch A (No Extras)",    silver_arch_a_ne),
    ("XGBoost: Arch B (Full)",         silver_arch_b_full),
    ("XGBoost: Arch B (No Extras)",    silver_arch_b_ne),
    ("ARIMA(5,1,0) Static",            silver_arima_static),
    ("ARIMA(5,1,0) Walk-Forward",      silver_arima_wf),
]

print("\n" + "=" * len(header))
print(header)
print("=" * len(header))
print(f"Panel A: Gold (GC=F)")
print(sep)
for name, (rmse, mae, mape, mse, r2) in rows_gold:
    print(f"{name:<35} {rmse:>10.4f} {mae:>10.4f} {mape:>10.4f} {mse:>14.4f} {r2:>8.4f}")
print(sep)
print(f"Panel B: Silver (SI=F)")
print(sep)
for name, (rmse, mae, mape, mse, r2) in rows_silver:
    print(f"{name:<35} {rmse:>10.4f} {mae:>10.4f} {mape:>10.4f} {mse:>14.4f} {r2:>8.4f}")
print("=" * len(header))

# Also save to CSV
all_rows = (
    [("Panel A: Gold (GC=F)", name, *vals) for name, vals in rows_gold] +
    [("Panel B: Silver (SI=F)", name, *vals) for name, vals in rows_silver]
)
df_out = pd.DataFrame(all_rows, columns=["Panel", "Model / Condition", "RMSE", "MAE", "MAPE", "MSE", "R2"])
out_path = '../data-extra-variables/metrics_table.csv'
df_out.to_csv(out_path, index=False)
print(f"\nTable saved to: {out_path}")
