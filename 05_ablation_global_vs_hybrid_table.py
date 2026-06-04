"""
Ablation Study: Global-Only vs Hybrid (Global + Local Egyptian Data)
Generates a unified metrics table for all 7 commodities.

Feature set definition:
  - Global-only : no EGP / Inflation / CBE columns
  - Hybrid      : full feature set (includes Egyptian local data)
"""
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    root_mean_squared_error, mean_absolute_percentage_error,
    mean_absolute_error, mean_squared_error, r2_score,
)
import warnings

warnings.filterwarnings("ignore")

LOCAL_KEYWORDS = ['EGP', 'Inflation', 'CBE']

TRAIN_END  = '2023-12-31'
VAL_START  = '2024-01-01'
VAL_END    = '2024-12-31'
TEST_START = '2025-01-01'


def split(df):
    return df[:TRAIN_END], df[VAL_START:VAL_END], df[TEST_START:]


def make_X(df, drop_cols):
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


def global_only(X):
    cols = [c for c in X.columns if not any(kw in c for kw in LOCAL_KEYWORDS)]
    return X[cols]


def compute_metrics(actual, predicted):
    return (
        root_mean_squared_error(actual, predicted),
        mean_absolute_error(actual, predicted),
        mean_absolute_percentage_error(actual, predicted),
        r2_score(actual, predicted),
    )


def reverse_log_return(df_master, price_col, index, preds):
    yp = df_master.loc[index, price_col].shift(1)
    yp.iloc[0] = df_master.loc['2024-12-31', price_col]
    return yp * np.exp(preds)


def run_ablation(X_train_h, X_val_h, X_test_h,
                 y_train, y_val,
                 df_master, price_col, test_index,
                 xgb_params, label):
    X_train_g = global_only(X_train_h)
    X_val_g   = global_only(X_val_h)
    X_test_g  = global_only(X_test_h)

    print(f"  Training {label} Hybrid  ({len(X_train_h.columns)} features)...")
    m_hybrid = xgb.XGBRegressor(**xgb_params)
    m_hybrid.fit(X_train_h, y_train, eval_set=[(X_val_h, y_val)], verbose=False)

    print(f"  Training {label} Global  ({len(X_train_g.columns)} features)...")
    m_global = xgb.XGBRegressor(**xgb_params)
    m_global.fit(X_train_g, y_train, eval_set=[(X_val_g, y_val)], verbose=False)

    actual = df_master.loc[test_index, price_col]
    price_h = reverse_log_return(df_master, price_col, test_index, m_hybrid.predict(X_test_h))
    price_g = reverse_log_return(df_master, price_col, test_index, m_global.predict(X_test_g))

    return compute_metrics(actual, price_g), compute_metrics(actual, price_h)


# ============================================================
#  EXTRA-VARIABLES  (Gold & Silver)
# ============================================================
print("Loading extra-variables data (Gold & Silver)...")
ROOT_EV  = './data-extra-variables/'
master_ev = pd.read_csv(ROOT_EV + '02_processed/01_master_metals_dataset.csv',
                        index_col='Date', parse_dates=['Date'])
diff_ev   = pd.read_csv(ROOT_EV + '03_final/01a_differencing_metals_dataset.csv',
                        index_col='Date', parse_dates=['Date'])

tr_ev, va_ev, te_ev = split(diff_ev)

# Columns always dropped in both Gold and Silver ablations (raw prices + cross-asset LogReturns)
# (mirrors 05b_ablation_study.py for extra-variables)
EV_RAW_DROP = ['Gold_Close', 'Silver_Close', 'DXY_Close', 'EGP_USD_Close',
               'DXY_Close_LogReturn', 'EGP_USD_Close_LogReturn']

EV_PARAMS = dict(n_estimators=500, learning_rate=0.01, max_depth=3,
                 subsample=0.8, colsample_bytree=0.8, random_state=42)

# --- Gold ---
print("\n[Gold]")
gold_drop = EV_RAW_DROP + ['Gold_Close_LogReturn', 'Silver_Close_LogReturn']
gold_metrics_g, gold_metrics_h = run_ablation(
    make_X(tr_ev, gold_drop), make_X(va_ev, gold_drop), make_X(te_ev, gold_drop),
    tr_ev['Gold_Close_LogReturn'], va_ev['Gold_Close_LogReturn'],
    master_ev, 'Gold_Close', te_ev.index,
    EV_PARAMS, 'Gold',
)

# --- Silver ---
print("\n[Silver]")
silver_drop = EV_RAW_DROP + ['Silver_Close_LogReturn', 'Gold_Close_LogReturn']
silver_metrics_g, silver_metrics_h = run_ablation(
    make_X(tr_ev, silver_drop), make_X(va_ev, silver_drop), make_X(te_ev, silver_drop),
    tr_ev['Silver_Close_LogReturn'], va_ev['Silver_Close_LogReturn'],
    master_ev, 'Silver_Close', te_ev.index,
    EV_PARAMS, 'Silver',
)

# ============================================================
#  ENERGY  (Brent Crude & Natural Gas)
# ============================================================
print("\n\nLoading energy data (Brent & Natural Gas)...")
ROOT_EN   = './data-energy/'
master_en = pd.read_csv(ROOT_EN + '02_processed/01_master_energy_dataset.csv',
                        index_col='Date', parse_dates=['Date'])
diff_en   = pd.read_csv(ROOT_EN + '03_final/01a_differencing_energy_dataset.csv',
                        index_col='Date', parse_dates=['Date'])

tr_en, va_en, te_en = split(diff_en)

EN_RAW_DROP = ['Brent_Crude_Close', 'Natural_Gas_Close', 'DXY_Close',
               'VIX_Close', 'SP500_Close', 'EGP_USD_Close']

EN_PARAMS = dict(n_estimators=1000, learning_rate=0.05, max_depth=5,
                 random_state=42, early_stopping_rounds=50)

# --- Brent Crude ---
print("\n[Brent Crude]")
brent_drop = EN_RAW_DROP + ['Brent_Crude_Close_LogReturn']
brent_metrics_g, brent_metrics_h = run_ablation(
    make_X(tr_en, brent_drop), make_X(va_en, brent_drop), make_X(te_en, brent_drop),
    tr_en['Brent_Crude_Close_LogReturn'], va_en['Brent_Crude_Close_LogReturn'],
    master_en, 'Brent_Crude_Close', te_en.index,
    EN_PARAMS, 'Brent Crude',
)

# --- Natural Gas ---
print("\n[Natural Gas]")
natgas_drop = EN_RAW_DROP + ['Natural_Gas_Close_LogReturn']
natgas_metrics_g, natgas_metrics_h = run_ablation(
    make_X(tr_en, natgas_drop), make_X(va_en, natgas_drop), make_X(te_en, natgas_drop),
    tr_en['Natural_Gas_Close_LogReturn'], va_en['Natural_Gas_Close_LogReturn'],
    master_en, 'Natural_Gas_Close', te_en.index,
    EN_PARAMS, 'Natural Gas',
)

# ============================================================
#  CROPS  (Wheat, Corn, Sugar)
# ============================================================
print("\n\nLoading crops data (Wheat, Corn, Sugar)...")
ROOT_CR   = './data-crops/'
master_cr = pd.read_csv(ROOT_CR + '02_processed/01_master_crops_dataset.csv',
                        index_col='Date', parse_dates=['Date'])
diff_cr   = pd.read_csv(ROOT_CR + '03_final/01a_differencing_crops_dataset.csv',
                        index_col='Date', parse_dates=['Date'])

tr_cr, va_cr, te_cr = split(diff_cr)

CR_RAW_DROP = ['Wheat_Close', 'Corn_Close', 'Sugar_Close',
               'Brent_Crude_Close', 'DXY_Close', 'EGP_USD_Close']

CR_PARAMS = dict(n_estimators=500, learning_rate=0.01, max_depth=3,
                 subsample=0.8, colsample_bytree=0.8, random_state=42,
                 early_stopping_rounds=50)

# --- Wheat ---
print("\n[Wheat]")
wheat_drop = CR_RAW_DROP + ['Wheat_Close_LogReturn']
wheat_metrics_g, wheat_metrics_h = run_ablation(
    make_X(tr_cr, wheat_drop), make_X(va_cr, wheat_drop), make_X(te_cr, wheat_drop),
    tr_cr['Wheat_Close_LogReturn'], va_cr['Wheat_Close_LogReturn'],
    master_cr, 'Wheat_Close', te_cr.index,
    CR_PARAMS, 'Wheat',
)

# --- Corn ---
print("\n[Corn]")
corn_drop = CR_RAW_DROP + ['Corn_Close_LogReturn']
corn_metrics_g, corn_metrics_h = run_ablation(
    make_X(tr_cr, corn_drop), make_X(va_cr, corn_drop), make_X(te_cr, corn_drop),
    tr_cr['Corn_Close_LogReturn'], va_cr['Corn_Close_LogReturn'],
    master_cr, 'Corn_Close', te_cr.index,
    CR_PARAMS, 'Corn',
)

# --- Sugar ---
print("\n[Sugar]")
sugar_drop = CR_RAW_DROP + ['Sugar_Close_LogReturn']
sugar_metrics_g, sugar_metrics_h = run_ablation(
    make_X(tr_cr, sugar_drop), make_X(va_cr, sugar_drop), make_X(te_cr, sugar_drop),
    tr_cr['Sugar_Close_LogReturn'], va_cr['Sugar_Close_LogReturn'],
    master_cr, 'Sugar_Close', te_cr.index,
    CR_PARAMS, 'Sugar',
)

# ============================================================
#  BUILD & PRINT TABLE
# ============================================================
rows = [
    ('Gold',         'Global-only', *gold_metrics_g),
    ('Gold',         'Hybrid',      *gold_metrics_h),
    ('Silver',       'Global-only', *silver_metrics_g),
    ('Silver',       'Hybrid',      *silver_metrics_h),
    ('Brent Crude',  'Global-only', *brent_metrics_g),
    ('Brent Crude',  'Hybrid',      *brent_metrics_h),
    ('Natural Gas',  'Global-only', *natgas_metrics_g),
    ('Natural Gas',  'Hybrid',      *natgas_metrics_h),
    ('Wheat',        'Global-only', *wheat_metrics_g),
    ('Wheat',        'Hybrid',      *wheat_metrics_h),
    ('Corn',         'Global-only', *corn_metrics_g),
    ('Corn',         'Hybrid',      *corn_metrics_h),
    ('Sugar',        'Global-only', *sugar_metrics_g),
    ('Sugar',        'Hybrid',      *sugar_metrics_h),
]

df_table = pd.DataFrame(rows, columns=['Commodity', 'Feature Set', 'RMSE', 'MAE', 'MAPE', 'R²'])

# Console print
header = f"{'Commodity':<14} {'Feature Set':<13} {'RMSE':>10} {'MAE':>10} {'MAPE':>10} {'R²':>8}"
sep    = '-' * len(header)
print('\n\n' + '=' * len(header))
print(header)
print('=' * len(header))
prev_commodity = None
for _, r in df_table.iterrows():
    if r['Commodity'] != prev_commodity and prev_commodity is not None:
        print(sep)
    prev_commodity = r['Commodity']
    print(f"{r['Commodity']:<14} {r['Feature Set']:<13} {r['RMSE']:>10.4f} {r['MAE']:>10.4f} {r['MAPE']:>10.4f} {r['R²']:>8.4f}")
print('=' * len(header))

# Save CSV (matches the requested format exactly)
out_path = './ablation_global_vs_hybrid_table.csv'
df_table.to_csv(out_path, index=False)
print(f'\nTable saved to: {out_path}')
