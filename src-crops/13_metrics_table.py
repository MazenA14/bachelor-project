import os
import pandas as pd
import numpy as np
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

print("Generating Metrics Table for Wheat, Corn, and Sugar Models...\n")

# --- CONFIGURATION ---
PROCESSED_DIR = '../data-crops/02_processed/'
FINAL_DIR     = '../data-crops/03_final/'
MODELS_DIR    = '../models-crops/'
TRAIN_END     = '2023-12-31'
VAL_START     = '2024-01-01'
VAL_END       = '2024-12-31'
TEST_START    = '2025-01-01'

# --- DATA LOADING ---
df_master = pd.read_csv(os.path.join(PROCESSED_DIR, '01_master_crops_dataset.csv'),                            index_col='Date', parse_dates=['Date'])
df_eng_a  = pd.read_csv(os.path.join(FINAL_DIR, '01a_engineered_differencing_crops_dataset.csv'),             index_col='Date', parse_dates=['Date'])
df_eng_b  = pd.read_csv(os.path.join(FINAL_DIR, '01b_engineered_detrending_crops_dataset.csv'),               index_col='Date', parse_dates=['Date'])
df_base_a = pd.read_csv(os.path.join(FINAL_DIR, '01a_differencing_crops_dataset.csv'),                        index_col='Date', parse_dates=['Date'])
df_base_b = pd.read_csv(os.path.join(FINAL_DIR, '01b_detrending_crops_dataset.csv'),                          index_col='Date', parse_dates=['Date'])

def split(df):
    return df[:TRAIN_END], df[VAL_START:VAL_END], df[TEST_START:]

_, _, test_eng_a  = split(df_eng_a)
_, _, test_eng_b  = split(df_eng_b)
_, _, test_base_a = split(df_base_a)
_, _, test_base_b = split(df_base_b)

# Common raw-price and macro columns to drop
_PRICES = ['Wheat_Close', 'Corn_Close', 'Sugar_Close', 'Brent_Crude_Close', 'DXY_Close', 'EGP_USD_Close']
_MACRO  = ['Egypt_Inflation_YoY', 'CBE_Interest_Rate']
_TRENDS = ['Wheat_Close_Trend', 'Corn_Close_Trend', 'Sugar_Close_Trend',
           'Brent_Crude_Close_Trend', 'DXY_Close_Trend', 'EGP_USD_Close_Trend']

def drop_a(commodity):
    return _PRICES + [f'{commodity}_Close_LogReturn'] + _MACRO

def drop_b(commodity):
    return _PRICES + _TRENDS + [f'{commodity}_Close_Residual']

def make_X(df, drop_cols):
    existing = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=existing)

# Test feature matrices per commodity
X_te_ea = {c: make_X(test_eng_a,  drop_a(c)) for c in ['Wheat', 'Corn', 'Sugar']}
X_te_eb = {c: make_X(test_eng_b,  drop_b(c)) for c in ['Wheat', 'Corn', 'Sugar']}
X_te_ba = {c: make_X(test_base_a, drop_a(c)) for c in ['Wheat', 'Corn', 'Sugar']}
X_te_bb = {c: make_X(test_base_b, drop_b(c)) for c in ['Wheat', 'Corn', 'Sugar']}

# Actual test prices
test_idx = test_eng_a.index
actual = {c: df_master.loc[test_idx, f'{c}_Close'] for c in ['Wheat', 'Corn', 'Sugar']}

# Last price before test set (for Arch A reversal)
last_val_date = df_master[:VAL_END].index[-1]
last_price = {c: df_master.loc[last_val_date, f'{c}_Close'] for c in ['Wheat', 'Corn', 'Sugar']}

# Model file suffixes
_SUFFIX = {'Wheat': '', 'Corn': '_corn', 'Sugar': '_sugar'}

def compute_metrics(actual, predicted):
    rmse = root_mean_squared_error(actual, predicted)
    mae  = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)
    mse  = mean_squared_error(actual, predicted)
    r2   = r2_score(actual, predicted)
    return rmse, mae, mape, mse, r2

def predict_arch_a(model, X_test, commodity):
    preds_log = model.predict(X_test)
    yp = df_master.loc[X_test.index, f'{commodity}_Close'].shift(1)
    yp.iloc[0] = last_price[commodity]
    return yp * np.exp(preds_log)

def predict_arch_b(model, X_test, commodity, df_b_src):
    preds_resid = model.predict(X_test)
    trend = df_b_src.loc[X_test.index, f'{commodity}_Close_Trend']
    return trend + preds_resid

results = {}

for commodity in ['Wheat', 'Corn', 'Sugar']:
    sfx = _SUFFIX[commodity]
    act = actual[commodity]
    print(f"--- Computing {commodity} metrics ---")

    # XGBoost Arch A Full
    m = xgb.XGBRegressor()
    m.load_model(os.path.join(MODELS_DIR, f'xgboost_a{sfx}.json'))
    preds = predict_arch_a(m, X_te_ea[commodity], commodity)
    arch_a_full = compute_metrics(act, preds)

    # XGBoost Arch A No Extras
    m = xgb.XGBRegressor()
    m.load_model(os.path.join(MODELS_DIR, f'xgboost_a_no_extras{sfx}.json'))
    preds = predict_arch_a(m, X_te_ba[commodity], commodity)
    arch_a_ne = compute_metrics(act, preds)

    # XGBoost Arch B Full
    m = xgb.XGBRegressor()
    m.load_model(os.path.join(MODELS_DIR, f'xgboost_b{sfx}.json'))
    preds = predict_arch_b(m, X_te_eb[commodity], commodity, df_eng_b)
    arch_b_full = compute_metrics(act, preds)

    # XGBoost Arch B No Extras
    m = xgb.XGBRegressor()
    m.load_model(os.path.join(MODELS_DIR, f'xgboost_b_no_extras{sfx}.json'))
    preds = predict_arch_b(m, X_te_bb[commodity], commodity, df_base_b)
    arch_b_ne = compute_metrics(act, preds)

    # ARIMA Static
    arima_static = ARIMAResults.load(os.path.join(MODELS_DIR, f'arima_baseline{sfx}.pkl'))
    static_preds = arima_static.forecast(steps=len(act))
    static_preds.index = act.index
    arima_static_metrics = compute_metrics(act, static_preds)

    # ARIMA Walk-Forward
    print(f"  Running {commodity} ARIMA Walk-Forward...")
    history = list(df_master[f'{commodity}_Close'][:VAL_END])
    wf_preds = []
    for t in range(len(act)):
        m_arima = ARIMA(history, order=(5, 1, 0)).fit()
        wf_preds.append(m_arima.forecast()[0])
        history.append(act.iloc[t])
        if (t + 1) % 50 == 0:
            print(f"    {commodity} WF: {t+1}/{len(act)}")
    wf_series = pd.Series(wf_preds, index=act.index)
    arima_wf_metrics = compute_metrics(act, wf_series)

    results[commodity] = [
        ("XGBoost: Arch A (Full)",      arch_a_full),
        ("XGBoost: Arch A (No Extras)", arch_a_ne),
        ("XGBoost: Arch B (Full)",      arch_b_full),
        ("XGBoost: Arch B (No Extras)", arch_b_ne),
        ("ARIMA(5,1,0) Static",         arima_static_metrics),
        ("ARIMA(5,1,0) Walk-Forward",   arima_wf_metrics),
    ]

# --- PRINT TABLE ---
header = f"{'Model / Condition':<35} {'RMSE':>10} {'MAE':>10} {'MAPE':>10} {'MSE':>14} {'R²':>8}"
sep    = "-" * len(header)

panels = [
    ("Panel A: Wheat (ZW=F)",  results['Wheat']),
    ("Panel B: Corn (ZC=F)",   results['Corn']),
    ("Panel C: Sugar (SB=F)",  results['Sugar']),
]

print("\n" + "=" * len(header))
print(header)
print("=" * len(header))
for panel_label, rows in panels:
    print(panel_label)
    print(sep)
    for name, (rmse, mae, mape, mse, r2) in rows:
        print(f"{name:<35} {rmse:>10.4f} {mae:>10.4f} {mape:>10.4f} {mse:>14.4f} {r2:>8.4f}")
    print(sep)
print("=" * len(header))

# Save to CSV
all_rows = [
    (label.split(":")[0], name, *vals)
    for label, rows in panels
    for name, vals in rows
]
df_out = pd.DataFrame(all_rows, columns=["Panel", "Model / Condition", "RMSE", "MAE", "MAPE", "MSE", "R2"])
out_path = '../data-crops/metrics_table.csv'
df_out.to_csv(out_path, index=False)
print(f"\nTable saved to: {out_path}")
