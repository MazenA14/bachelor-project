"""
Ablation Study: 4-Condition Feature Set Table
For each of 7 commodities:
  1. Full           — cross-commodity features + local Egyptian data
  2. Reduced        — no cross-commodity features, keeps Egyptian data
  3. Global Only    — cross-commodity features, no Egyptian data
  4. No Extras (NE) — no cross-commodity features, no Egyptian data
"""
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    root_mean_squared_error, mean_absolute_percentage_error,
    mean_absolute_error, r2_score,
)
import warnings

warnings.filterwarnings("ignore")

LOCAL_KEYWORDS = ['EGP', 'Inflation', 'CBE']
TRAIN_END  = '2023-12-31'
VAL_START  = '2024-01-01'
VAL_END    = '2024-12-31'
TEST_START = '2025-01-01'


def split(df, train_end=TRAIN_END, val_start=VAL_START, val_end=VAL_END, test_start=TEST_START):
    return df[:train_end], df[val_start:val_end], df[test_start:]


def make_X(df, drop_cols):
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


def strip_local(X):
    keep = [c for c in X.columns if not any(kw in c for kw in LOCAL_KEYWORDS)]
    return X[keep]


def strip_cross(X, cross_kw):
    keep = [c for c in X.columns if not any(kw in c for kw in cross_kw)]
    return X[keep]


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


def train_eval(X_tr, X_va, X_te, y_tr, y_va, df_master, price_col, test_idx, params):
    m = xgb.XGBRegressor(**params)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    preds  = m.predict(X_te)
    actual = df_master.loc[test_idx, price_col]
    price  = reverse_log_return(df_master, price_col, test_idx, preds)
    return compute_metrics(actual, price)


def run_four_conditions(name, tr, va, te, base_drop, target_col, cross_kw,
                        df_master, price_col, params, cross_label):
    print(f"\n  [{name}]")

    # Base drop applied; target's own LogReturn is already in base_drop
    X_tr_b = make_X(tr, base_drop)
    X_va_b = make_X(va, base_drop)
    X_te_b = make_X(te, base_drop)

    # ── 1. Full (cross-commodity + Egyptian) ──────────────────────────────
    print(f"     1. Full                ({len(X_tr_b.columns)} feat) ...")
    m1 = train_eval(X_tr_b, X_va_b, X_te_b, tr[target_col], va[target_col],
                    df_master, price_col, te.index, params)

    # ── 2. Reduced (no cross-commodity, keeps Egyptian) ───────────────────
    X_tr_r = strip_cross(X_tr_b, cross_kw)
    X_va_r = strip_cross(X_va_b, cross_kw)
    X_te_r = strip_cross(X_te_b, cross_kw)
    print(f"     2. Reduced             ({len(X_tr_r.columns)} feat) ...")
    m2 = train_eval(X_tr_r, X_va_r, X_te_r, tr[target_col], va[target_col],
                    df_master, price_col, te.index, params)

    # ── 3. Global Only (cross-commodity, no Egyptian) ─────────────────────
    X_tr_g = strip_local(X_tr_b)
    X_va_g = strip_local(X_va_b)
    X_te_g = strip_local(X_te_b)
    print(f"     3. Global Only         ({len(X_tr_g.columns)} feat) ...")
    m3 = train_eval(X_tr_g, X_va_g, X_te_g, tr[target_col], va[target_col],
                    df_master, price_col, te.index, params)

    # ── 4. No Extras (no cross-commodity, no Egyptian) ────────────────────
    X_tr_n = strip_cross(X_tr_g, cross_kw)
    X_va_n = strip_cross(X_va_g, cross_kw)
    X_te_n = strip_cross(X_te_g, cross_kw)
    print(f"     4. No Extras (NE)      ({len(X_tr_n.columns)} feat) ...")
    m4 = train_eval(X_tr_n, X_va_n, X_te_n, tr[target_col], va[target_col],
                    df_master, price_col, te.index, params)

    return [
        (name, f'Full — Global + Local', *m1),
        ('',   f'Full — Global Only',    *m3),
        ('',   f'Reduced — Global + Local', *m2),
        ('',   f'Reduced — Global Only',    *m4),
    ]


rows = []

# ============================================================
#  EXTRA-VARIABLES  (Gold & Silver)
# ============================================================
print("=" * 60)
print("  EXTRA-VARIABLES  (Gold & Silver)")
print("=" * 60)

ROOT_EV   = './data-extra-variables/'
master_ev = pd.read_csv(ROOT_EV + '02_processed/01_master_metals_dataset.csv',
                        index_col='Date', parse_dates=['Date'])
diff_ev   = pd.read_csv(ROOT_EV + '03_final/01a_differencing_metals_dataset.csv',
                        index_col='Date', parse_dates=['Date'])

tr_ev, va_ev, te_ev = split(diff_ev)

# Raw-price columns always excluded (mirrors 05c logic for extra-variables)
EV_RAW = ['Gold_Close', 'Silver_Close', 'DXY_Close', 'EGP_USD_Close',
          'DXY_Close_LogReturn', 'EGP_USD_Close_LogReturn']

EV_PARAMS = dict(n_estimators=500, learning_rate=0.01, max_depth=3,
                 subsample=0.8, colsample_bytree=0.8, random_state=42)

rows += run_four_conditions(
    'Gold', tr_ev, va_ev, te_ev,
    base_drop   = EV_RAW + ['Gold_Close_LogReturn'],
    target_col  = 'Gold_Close_LogReturn',
    cross_kw    = ['Silver', 'Ratio'],
    df_master   = master_ev,
    price_col   = 'Gold_Close',
    params      = EV_PARAMS,
    cross_label = 'Silver',
)

rows += run_four_conditions(
    'Silver', tr_ev, va_ev, te_ev,
    base_drop   = EV_RAW + ['Silver_Close_LogReturn'],
    target_col  = 'Silver_Close_LogReturn',
    cross_kw    = ['Gold', 'Ratio'],
    df_master   = master_ev,
    price_col   = 'Silver_Close',
    params      = EV_PARAMS,
    cross_label = 'Gold',
)

# ============================================================
#  ENERGY  (Brent Crude & Natural Gas)
# ============================================================
print("\n" + "=" * 60)
print("  ENERGY  (Brent Crude & Natural Gas)")
print("=" * 60)

ROOT_EN   = './data-energy/'
master_en = pd.read_csv(ROOT_EN + '02_processed/01_master_energy_dataset.csv',
                        index_col='Date', parse_dates=['Date'])
diff_en   = pd.read_csv(ROOT_EN + '03_final/01a_differencing_energy_dataset.csv',
                        index_col='Date', parse_dates=['Date'])

tr_en, va_en, te_en = split(diff_en)

EN_RAW = ['Brent_Crude_Close', 'Natural_Gas_Close', 'DXY_Close',
          'VIX_Close', 'SP500_Close', 'EGP_USD_Close']

EN_PARAMS = dict(n_estimators=1000, learning_rate=0.05, max_depth=5,
                 random_state=42, early_stopping_rounds=50)

rows += run_four_conditions(
    'Brent Crude', tr_en, va_en, te_en,
    base_drop   = EN_RAW + ['Brent_Crude_Close_LogReturn'],
    target_col  = 'Brent_Crude_Close_LogReturn',
    cross_kw    = ['Natural_Gas', 'NatGas', 'Ratio'],
    df_master   = master_en,
    price_col   = 'Brent_Crude_Close',
    params      = EN_PARAMS,
    cross_label = 'Nat Gas',
)

rows += run_four_conditions(
    'Natural Gas', tr_en, va_en, te_en,
    base_drop   = EN_RAW + ['Natural_Gas_Close_LogReturn'],
    target_col  = 'Natural_Gas_Close_LogReturn',
    cross_kw    = ['Brent', 'Ratio'],
    df_master   = master_en,
    price_col   = 'Natural_Gas_Close',
    params      = EN_PARAMS,
    cross_label = 'Brent',
)

# ============================================================
#  CROPS  (Wheat, Corn, Sugar)
# ============================================================
print("\n" + "=" * 60)
print("  CROPS  (Wheat, Corn, Sugar)")
print("=" * 60)

ROOT_CR   = './data-crops/'
master_cr = pd.read_csv(ROOT_CR + '02_processed/01_master_crops_dataset.csv',
                        index_col='Date', parse_dates=['Date'])
diff_cr   = pd.read_csv(ROOT_CR + '03_final/01a_differencing_crops_dataset.csv',
                        index_col='Date', parse_dates=['Date'])

tr_cr, va_cr, te_cr = split(diff_cr)

CR_RAW = ['Wheat_Close', 'Corn_Close', 'Sugar_Close',
          'Brent_Crude_Close', 'DXY_Close', 'EGP_USD_Close']

CR_PARAMS = dict(n_estimators=500, learning_rate=0.01, max_depth=3,
                 subsample=0.8, colsample_bytree=0.8,
                 random_state=42, early_stopping_rounds=50)

rows += run_four_conditions(
    'Wheat', tr_cr, va_cr, te_cr,
    base_drop   = CR_RAW + ['Wheat_Close_LogReturn'],
    target_col  = 'Wheat_Close_LogReturn',
    cross_kw    = ['Corn', 'Sugar', 'Ratio'],
    df_master   = master_cr,
    price_col   = 'Wheat_Close',
    params      = CR_PARAMS,
    cross_label = 'Corn & Sugar',
)

rows += run_four_conditions(
    'Corn', tr_cr, va_cr, te_cr,
    base_drop   = CR_RAW + ['Corn_Close_LogReturn'],
    target_col  = 'Corn_Close_LogReturn',
    cross_kw    = ['Wheat', 'Sugar', 'Ratio'],
    df_master   = master_cr,
    price_col   = 'Corn_Close',
    params      = CR_PARAMS,
    cross_label = 'Wheat & Sugar',
)

rows += run_four_conditions(
    'Sugar', tr_cr, va_cr, te_cr,
    base_drop   = CR_RAW + ['Sugar_Close_LogReturn'],
    target_col  = 'Sugar_Close_LogReturn',
    cross_kw    = ['Wheat', 'Corn', 'Ratio'],
    df_master   = master_cr,
    price_col   = 'Sugar_Close',
    params      = CR_PARAMS,
    cross_label = 'Wheat & Corn',
)

# ============================================================
#  BUILD TABLE & SAVE
# ============================================================
cols = ['Target Commodity', 'Feature Set Condition', 'RMSE', 'MAE', 'MAPE', 'R²']
df_out = pd.DataFrame(rows, columns=cols)

# Console print
hdr = f"{'Commodity':<15} {'Feature Set Condition':<28} {'RMSE':>10} {'MAE':>10} {'MAPE':>10} {'R²':>8}"
sep = '-' * len(hdr)
print('\n\n' + '=' * len(hdr))
print(hdr)
print('=' * len(hdr))

prev = None
for _, r in df_out.iterrows():
    comm = r['Target Commodity']
    disp = comm if comm else ''
    if comm and comm != prev and prev is not None:
        print(sep)
    if comm:
        prev = comm
    print(f"{disp:<15} {r['Feature Set Condition']:<28} "
          f"{r['RMSE']:>10.4f} {r['MAE']:>10.4f} {r['MAPE']:>10.4f} {r['R²']:>8.4f}")
print('=' * len(hdr))

out_path = './ablation_four_conditions_table.csv'
df_out.to_csv(out_path, index=False)
print(f'\nTable saved to: {out_path}')
