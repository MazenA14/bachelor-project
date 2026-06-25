#!/usr/bin/env python
"""
defense_analysis.py
====================
Computes random-walk baseline, directional accuracy, and a simple long/flat
backtest for all 7 commodity XGBoost models to provide evidence of predictive
skill beyond persistence.

Run from the project root:   python defense_analysis.py
"""

import os
import textwrap
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    root_mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, r2_score,
)
import warnings
warnings.filterwarnings("ignore")

BASE          = os.path.dirname(os.path.abspath(__file__))
TEST_START    = '2025-01-01'
TRAIN_VAL_END = '2024-12-31'

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def p(path: str) -> str:
    return os.path.join(BASE, path)

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(p(path), index_col='Date', parse_dates=['Date'])

def load_model(path: str) -> xgb.XGBRegressor:
    m = xgb.XGBRegressor()
    m.load_model(p(path))
    return m

def compute_metrics(actual: pd.Series, pred: pd.Series) -> dict:
    return dict(
        rmse = root_mean_squared_error(actual, pred),
        mae  = mean_absolute_error(actual, pred),
        mape = mean_absolute_percentage_error(actual, pred) * 100,
        r2   = r2_score(actual, pred),
    )

def directional_accuracy(pred_lr: pd.Series, actual_lr: pd.Series) -> float:
    """% of days where sign(predicted log-return) == sign(actual log-return).
    Excludes true-flat days (actual return = 0) from the denominator."""
    idx  = pred_lr.index.intersection(actual_lr.index)
    p_d  = np.sign(pred_lr[idx])
    a_d  = np.sign(actual_lr[idx])
    mask = a_d != 0
    if mask.sum() == 0:
        return float('nan')
    return float((p_d[mask] == a_d[mask]).mean() * 100)

def backtest_long_flat(pred_lr: pd.Series, actual_lr: pd.Series) -> dict:
    """
    Long/flat strategy: go long on day t when predicted log-return > 0, else flat.
    No leverage.  Transaction costs excluded (stated explicitly in output).
    Returns annualised Sharpe on a risk-free rate of 0.
    """
    idx      = pred_lr.index.intersection(actual_lr.index)
    pos      = (pred_lr[idx] > 0).astype(float)
    strat_lr = pos * actual_lr[idx]
    bnh_lr   = actual_lr[idx]

    strat_cum = float(np.expm1(strat_lr.cumsum().iloc[-1]) * 100)
    bnh_cum   = float(np.expm1(bnh_lr.cumsum().iloc[-1])   * 100)

    strat_sh  = (float(strat_lr.mean() / strat_lr.std() * np.sqrt(252))
                 if strat_lr.std() > 0 else float('nan'))
    bnh_sh    = (float(bnh_lr.mean()   / bnh_lr.std()   * np.sqrt(252))
                 if bnh_lr.std()   > 0 else float('nan'))

    return dict(
        strat_cum = strat_cum,
        bnh_cum   = bnh_cum,
        strat_sh  = strat_sh,
        bnh_sh    = bnh_sh,
        long_pct  = float(pos.mean() * 100),
        n_days    = int(idx.size),
    )

# -----------------------------------------------------------------------------
# COMMODITY SPECIFICATIONS
# -----------------------------------------------------------------------------

SPECS = [

    # -- GOLD ------------------------------------------------------------------
    dict(
        commodity  = 'Gold',
        master     = 'data-extra-variables/02_processed/01_master_metals_dataset.csv',
        a_full_ds  = 'data-extra-variables/03_final/01a_engineered_differencing_metals_dataset.csv',
        b_full_ds  = 'data-extra-variables/03_final/01b_engineered_detrending_metals_dataset.csv',
        a_noxt_ds  = 'data-extra-variables/03_final/01a_differencing_metals_dataset.csv',
        b_noxt_ds  = 'data-extra-variables/03_final/01b_detrending_metals_dataset.csv',
        model_af   = 'models-extra-variables/xgboost_a.json',
        model_bf   = 'models-extra-variables/xgboost_b.json',
        model_an   = 'models-extra-variables/xgboost_a_no_extras.json',
        model_bn   = 'models-extra-variables/xgboost_b_no_extras.json',
        price_col  = 'Gold_Close',
        logret_col = 'Gold_Close_LogReturn',
        trend_col  = 'Gold_Close_Trend',
        raw_cols   = ['Gold_Close','Silver_Close','DXY_Close','SP500_Close','VIX_Close','EGP_USD_Close'],
        extra_a    = [],
        extra_b    = [],
    ),

    # -- SILVER ----------------------------------------------------------------
    dict(
        commodity  = 'Silver',
        master     = 'data-extra-variables/02_processed/01_master_metals_dataset.csv',
        a_full_ds  = 'data-extra-variables/03_final/01a_engineered_differencing_metals_dataset.csv',
        b_full_ds  = 'data-extra-variables/03_final/01b_engineered_detrending_metals_dataset.csv',
        a_noxt_ds  = 'data-extra-variables/03_final/01a_differencing_metals_dataset.csv',
        b_noxt_ds  = 'data-extra-variables/03_final/01b_detrending_metals_dataset.csv',
        model_af   = 'models-extra-variables/xgboost_a_silver.json',
        model_bf   = 'models-extra-variables/xgboost_b_silver.json',
        model_an   = 'models-extra-variables/xgboost_a_no_extras_silver.json',
        model_bn   = 'models-extra-variables/xgboost_b_no_extras_silver.json',
        price_col  = 'Silver_Close',
        logret_col = 'Silver_Close_LogReturn',
        trend_col  = 'Silver_Close_Trend',
        raw_cols   = ['Gold_Close','Silver_Close','DXY_Close','SP500_Close','VIX_Close','EGP_USD_Close'],
        extra_a    = [],
        extra_b    = [],
    ),

    # -- BRENT CRUDE -----------------------------------------------------------
    dict(
        commodity  = 'Brent Crude',
        master     = 'data-energy/02_processed/01_master_energy_dataset.csv',
        a_full_ds  = 'data-energy/03_final/01a_engineered_differencing_energy_dataset.csv',
        b_full_ds  = 'data-energy/03_final/01b_engineered_detrending_energy_dataset.csv',
        a_noxt_ds  = 'data-energy/03_final/01a_differencing_energy_dataset.csv',
        b_noxt_ds  = 'data-energy/03_final/01b_detrending_energy_dataset.csv',
        model_af   = 'models-energy/xgboost_a.json',
        model_bf   = 'models-energy/xgboost_b.json',
        model_an   = 'models-energy/xgboost_a_no_extras.json',
        model_bn   = 'models-energy/xgboost_b_no_extras.json',
        price_col  = 'Brent_Crude_Close',
        logret_col = 'Brent_Crude_Close_LogReturn',
        trend_col  = 'Brent_Crude_Close_Trend',
        raw_cols   = ['Brent_Crude_Close','Natural_Gas_Close','DXY_Close','SP500_Close','VIX_Close','EGP_USD_Close'],
        extra_a    = ['US_10Yr_Yield_Diff','Egypt_Inflation_YoY','CBE_Interest_Rate'],
        extra_b    = ['Egypt_Inflation_YoY','CBE_Interest_Rate'],
    ),

    # -- NATURAL GAS -----------------------------------------------------------
    dict(
        commodity  = 'Natural Gas',
        master     = 'data-energy/02_processed/01_master_energy_dataset.csv',
        a_full_ds  = 'data-energy/03_final/01a_engineered_differencing_energy_dataset.csv',
        b_full_ds  = 'data-energy/03_final/01b_engineered_detrending_energy_dataset.csv',
        a_noxt_ds  = 'data-energy/03_final/01a_differencing_energy_dataset.csv',
        b_noxt_ds  = 'data-energy/03_final/01b_detrending_energy_dataset.csv',
        model_af   = 'models-energy/xgboost_a_gas.json',
        model_bf   = 'models-energy/xgboost_b_gas.json',
        model_an   = 'models-energy/xgboost_a_no_extras_gas.json',
        model_bn   = 'models-energy/xgboost_b_no_extras_gas.json',
        price_col  = 'Natural_Gas_Close',
        logret_col = 'Natural_Gas_Close_LogReturn',
        trend_col  = 'Natural_Gas_Close_Trend',
        raw_cols   = ['Brent_Crude_Close','Natural_Gas_Close','DXY_Close','SP500_Close','VIX_Close','EGP_USD_Close'],
        extra_a    = ['US_10Yr_Yield_Diff','Egypt_Inflation_YoY','CBE_Interest_Rate'],
        extra_b    = ['Egypt_Inflation_YoY','CBE_Interest_Rate'],
    ),

    # -- WHEAT -----------------------------------------------------------------
    dict(
        commodity  = 'Wheat',
        master     = 'data-crops/02_processed/01_master_crops_dataset.csv',
        a_full_ds  = 'data-crops/03_final/01a_engineered_differencing_crops_dataset.csv',
        b_full_ds  = 'data-crops/03_final/01b_engineered_detrending_crops_dataset.csv',
        a_noxt_ds  = 'data-crops/03_final/01a_differencing_crops_dataset.csv',
        b_noxt_ds  = 'data-crops/03_final/01b_detrending_crops_dataset.csv',
        model_af   = 'models-crops/xgboost_a.json',
        model_bf   = 'models-crops/xgboost_b.json',
        model_an   = 'models-crops/xgboost_a_no_extras.json',
        model_bn   = 'models-crops/xgboost_b_no_extras.json',
        price_col  = 'Wheat_Close',
        logret_col = 'Wheat_Close_LogReturn',
        trend_col  = 'Wheat_Close_Trend',
        # Crops use explicit drop lists (not formula-derived) to match original scripts
        drop_a_explicit = [
            'Wheat_Close','Corn_Close','Sugar_Close','Brent_Crude_Close','DXY_Close','EGP_USD_Close',
            'Wheat_Close_LogReturn','Egypt_Inflation_YoY','CBE_Interest_Rate',
        ],
        drop_b_explicit = [
            'Wheat_Close','Corn_Close','Sugar_Close','Brent_Crude_Close','DXY_Close','EGP_USD_Close',
            'Wheat_Close_Trend','Wheat_Close_Residual',
            'Corn_Close_Trend','Sugar_Close_Trend','Brent_Crude_Close_Trend',
            'DXY_Close_Trend','EGP_USD_Close_Trend',
        ],
    ),

    # -- CORN ------------------------------------------------------------------
    dict(
        commodity  = 'Corn',
        master     = 'data-crops/02_processed/01_master_crops_dataset.csv',
        a_full_ds  = 'data-crops/03_final/01a_engineered_differencing_crops_dataset.csv',
        b_full_ds  = 'data-crops/03_final/01b_engineered_detrending_crops_dataset.csv',
        a_noxt_ds  = 'data-crops/03_final/01a_differencing_crops_dataset.csv',
        b_noxt_ds  = 'data-crops/03_final/01b_detrending_crops_dataset.csv',
        model_af   = 'models-crops/xgboost_a_corn.json',
        model_bf   = 'models-crops/xgboost_b_corn.json',
        model_an   = 'models-crops/xgboost_a_no_extras_corn.json',
        model_bn   = 'models-crops/xgboost_b_no_extras_corn.json',
        price_col  = 'Corn_Close',
        logret_col = 'Corn_Close_LogReturn',
        trend_col  = 'Corn_Close_Trend',
        drop_a_explicit = [
            'Wheat_Close','Corn_Close','Sugar_Close','Brent_Crude_Close','DXY_Close','EGP_USD_Close',
            'Corn_Close_LogReturn','Egypt_Inflation_YoY','CBE_Interest_Rate',
        ],
        drop_b_explicit = [
            'Wheat_Close','Corn_Close','Sugar_Close','Brent_Crude_Close','DXY_Close','EGP_USD_Close',
            'Wheat_Close_Trend',
            'Corn_Close_Trend','Corn_Close_Residual',
            'Sugar_Close_Trend','Brent_Crude_Close_Trend',
            'DXY_Close_Trend','EGP_USD_Close_Trend',
        ],
    ),

    # -- SUGAR -----------------------------------------------------------------
    dict(
        commodity  = 'Sugar',
        master     = 'data-crops/02_processed/01_master_crops_dataset.csv',
        a_full_ds  = 'data-crops/03_final/01a_engineered_differencing_crops_dataset.csv',
        b_full_ds  = 'data-crops/03_final/01b_engineered_detrending_crops_dataset.csv',
        a_noxt_ds  = 'data-crops/03_final/01a_differencing_crops_dataset.csv',
        b_noxt_ds  = 'data-crops/03_final/01b_detrending_crops_dataset.csv',
        model_af   = 'models-crops/xgboost_a_sugar.json',
        model_bf   = 'models-crops/xgboost_b_sugar.json',
        model_an   = 'models-crops/xgboost_a_no_extras_sugar.json',
        model_bn   = 'models-crops/xgboost_b_no_extras_sugar.json',
        price_col  = 'Sugar_Close',
        logret_col = 'Sugar_Close_LogReturn',
        trend_col  = 'Sugar_Close_Trend',
        drop_a_explicit = [
            'Wheat_Close','Corn_Close','Sugar_Close','Brent_Crude_Close','DXY_Close','EGP_USD_Close',
            'Sugar_Close_LogReturn','Egypt_Inflation_YoY','CBE_Interest_Rate',
        ],
        drop_b_explicit = [
            'Wheat_Close','Corn_Close','Sugar_Close','Brent_Crude_Close','DXY_Close','EGP_USD_Close',
            'Wheat_Close_Trend','Corn_Close_Trend',
            'Sugar_Close_Trend','Sugar_Close_Residual',
            'Brent_Crude_Close_Trend',
            'DXY_Close_Trend','EGP_USD_Close_Trend',
        ],
    ),
]


# -----------------------------------------------------------------------------
# CORE ANALYSIS PER COMMODITY
# -----------------------------------------------------------------------------

def build_drop_lists(spec: dict, df_a: pd.DataFrame, df_b: pd.DataFrame):
    """Return (drop_a, drop_b) matching each script's exact logic."""
    if 'drop_a_explicit' in spec:
        return spec['drop_a_explicit'], spec['drop_b_explicit']
    raw = spec['raw_cols']
    lr  = [f'{c}_LogReturn'   for c in raw]
    tr  = [f'{c}_Trend'       for c in raw]
    re  = [f'{c}_Residual'    for c in raw]
    drop_a = raw + lr + spec.get('extra_a', [])
    drop_b = raw + tr + re + spec.get('extra_b', [])
    return drop_a, drop_b


def analyse_commodity(spec: dict) -> dict:
    name = spec['commodity']
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # -- Load data --------------------------------------------
    master    = load_csv(spec['master'])
    df_af     = load_csv(spec['a_full_ds'])
    df_bf     = load_csv(spec['b_full_ds'])
    df_an     = load_csv(spec['a_noxt_ds'])
    df_bn     = load_csv(spec['b_noxt_ds'])

    test_af   = df_af[TEST_START:]
    test_bf   = df_bf[TEST_START:]
    test_an   = df_an[TEST_START:]
    test_bn   = df_bn[TEST_START:]

    price_col = spec['price_col']
    lr_col    = spec['logret_col']
    tr_col    = spec['trend_col']

    actual_prices     = master.loc[test_af.index, price_col]
    last_train_price  = master.loc[TRAIN_VAL_END, price_col]

    # Actual log-returns for the test period
    actual_lr = np.log(actual_prices / actual_prices.shift(1)).dropna()
    # Use the pre-computed column if available; recompute if not
    if lr_col in master.columns:
        actual_lr_full = master.loc[test_af.index, lr_col].dropna()
    else:
        actual_lr_full = actual_lr

    # -- Build feature matrices --------------------------------
    drop_a, drop_b = build_drop_lists(spec, df_af, df_bf)
    X_af = test_af.drop(columns=[c for c in drop_a if c in test_af.columns])
    X_bf = test_bf.drop(columns=[c for c in drop_b if c in test_bf.columns])
    X_an = test_an.drop(columns=[c for c in drop_a if c in test_an.columns])
    X_bn = test_bn.drop(columns=[c for c in drop_b if c in test_bn.columns])

    # -- Load models -------------------------------------------
    mdl_af = load_model(spec['model_af'])
    mdl_bf = load_model(spec['model_bf'])
    mdl_an = load_model(spec['model_an'])
    mdl_bn = load_model(spec['model_bn'])

    # -- Generate predictions ----------------------------------
    # Architecture A: model outputs log returns → reverse to prices
    def rev_lr(pred_lr_arr, idx):
        yesterday = master.loc[idx, price_col].shift(1)
        yesterday.iloc[0] = last_train_price
        return pd.Series(yesterday.values * np.exp(pred_lr_arr), index=idx)

    # Architecture B: model outputs residuals → add trend back
    def rev_detrend(pred_resid_arr, df_src, idx):
        return pd.Series(df_src.loc[idx, tr_col].values + pred_resid_arr, index=idx)

    pred_lr_af  = pd.Series(mdl_af.predict(X_af), index=X_af.index)
    pred_lr_an  = pd.Series(mdl_an.predict(X_an), index=X_an.index)

    prices_af   = rev_lr(pred_lr_af.values,  X_af.index)
    prices_an   = rev_lr(pred_lr_an.values,  X_an.index)
    prices_bf   = rev_detrend(mdl_bf.predict(X_bf), df_bf, X_bf.index)
    prices_bn   = rev_detrend(mdl_bn.predict(X_bn), df_bn, X_bn.index)

    # -- Random-walk baseline ----------------------------------
    rw_prices   = actual_prices.shift(1)
    rw_prices.iloc[0] = last_train_price

    # -- Level metrics -----------------------------------------
    m_af  = compute_metrics(actual_prices, prices_af)
    m_an  = compute_metrics(actual_prices, prices_an)
    m_bf  = compute_metrics(actual_prices, prices_bf)
    m_bn  = compute_metrics(actual_prices, prices_bn)
    m_rw  = compute_metrics(actual_prices, rw_prices)

    # -- Directional accuracy (Arch A only -- log-return sign) --
    # Actual log returns aligned to Arch A test index
    act_lr_idx = actual_lr_full.reindex(pred_lr_af.index).dropna()
    da_af  = directional_accuracy(pred_lr_af.reindex(act_lr_idx.index), act_lr_idx)
    da_an  = directional_accuracy(pred_lr_an.reindex(act_lr_idx.index), act_lr_idx)
    # RW always predicts 0 log-return → always "no movement" prediction
    rw_lr  = pd.Series(0.0, index=act_lr_idx.index)
    da_rw  = directional_accuracy(rw_lr, act_lr_idx)

    # Arch B directional: use implied price change sign
    prices_bf_aligned = prices_bf.reindex(actual_prices.index).dropna()
    prices_bn_aligned = prices_bn.reindex(actual_prices.index).dropna()
    actual_aligned    = actual_prices.reindex(prices_bf_aligned.index)
    prev_actual       = actual_aligned.shift(1)
    prev_actual.iloc[0] = last_train_price
    da_bf = float(
        ((np.sign(prices_bf_aligned - prev_actual) ==
          np.sign(actual_aligned   - prev_actual))[np.sign(actual_aligned - prev_actual) != 0]).mean() * 100
    )
    da_bn = float(
        ((np.sign(prices_bn_aligned - prev_actual) ==
          np.sign(actual_aligned   - prev_actual))[np.sign(actual_aligned - prev_actual) != 0]).mean() * 100
    )

    # -- Backtest (Arch A only) --------------------------------
    bt_af = backtest_long_flat(pred_lr_af.reindex(act_lr_idx.index), act_lr_idx)
    bt_an = backtest_long_flat(pred_lr_an.reindex(act_lr_idx.index), act_lr_idx)

    return dict(
        commodity = name,
        m_af=m_af, m_an=m_an, m_bf=m_bf, m_bn=m_bn, m_rw=m_rw,
        da_af=da_af, da_an=da_an, da_bf=da_bf, da_bn=da_bn, da_rw=da_rw,
        bt_af=bt_af, bt_an=bt_an,
        n_test=len(actual_prices),
    )


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def fmt(x, dec=4): return f'{x:.{dec}f}' if not np.isnan(x) else 'N/A'

def print_section(title):
    print(f'\n{"-"*72}')
    print(f'  {title}')
    print(f'{"-"*72}')

results = []
for spec in SPECS:
    results.append(analyse_commodity(spec))

# ======================================================================
# TABLE 1 - LEVEL METRICS: MODEL vs RANDOM WALK
# ======================================================================
print_section('TABLE 1 - LEVEL METRICS: XGBoost Arch A vs Random Walk Baseline')
print(f'{"Commodity":<14} {"Condition":<22} {"RMSE":>8} {"MAE":>8} {"MAPE%":>8} {"R2":>8}')
print('-'*72)
for r in results:
    c = r['commodity']
    for label, m in [
        ('Arch A Full',    r['m_af']),
        ('Arch A No Ext',  r['m_an']),
        ('Random Walk',    r['m_rw']),
    ]:
        print(f'{c:<14} {label:<22} {fmt(m["rmse"],3):>8} {fmt(m["mae"],3):>8} {fmt(m["mape"],3):>8} {fmt(m["r2"],4):>8}')
    print()

# ======================================================================
# TABLE 2 - DIRECTIONAL ACCURACY
# ======================================================================
print_section('TABLE 2 - DIRECTIONAL ACCURACY (% days correct sign)')
print(f'{"Commodity":<14} {"Arch A Full":>12} {"Arch A NoExt":>13} {"Arch B Full":>12} {"Arch B NoExt":>13} {"RW (~0%)":>10}')
print('-'*76)
for r in results:
    c = r['commodity']
    print(f'{c:<14} {fmt(r["da_af"],1):>12} {fmt(r["da_an"],1):>13} '
          f'{fmt(r["da_bf"],1):>12} {fmt(r["da_bn"],1):>13} {fmt(r["da_rw"],1):>10}')

# ======================================================================
# TABLE 3 - SIMPLE BACKTEST (Arch A, no transaction costs)
# ======================================================================
print_section('TABLE 3 - SIMPLE LONG/FLAT BACKTEST: Arch A vs Buy-and-Hold')
print('  (No transaction costs; Sharpe annualised at 252 days, Rf = 0)')
print(f'\n{"Commodity":<14} {"Condition":<18} {"Cum Ret%":>10} {"Sharpe":>8} {"Long%":>8}')
print('-'*60)
for r in results:
    c  = r['commodity']
    af = r['bt_af']
    an = r['bt_an']
    print(f'{c:<14} {"Arch A Full":<18} {fmt(af["strat_cum"],1):>10} {fmt(af["strat_sh"],2):>8} {fmt(af["long_pct"],1):>8}')
    print(f'{"":14} {"Arch A No Ext":<18} {fmt(an["strat_cum"],1):>10} {fmt(an["strat_sh"],2):>8} {fmt(an["long_pct"],1):>8}')
    print(f'{"":14} {"Buy-and-Hold":<18} {fmt(af["bnh_cum"],1):>10} {fmt(af["bnh_sh"],2):>8} {"100.0":>8}')
    print()

# ======================================================================
# TABLE 4 - CONDENSED HEAD-TO-HEAD SUMMARY (for slides)
# ======================================================================
print_section('TABLE 4 - CONDENSED SUMMARY: Arch A Full vs Random Walk')
print(f'{"Commodity":<14}  {"---- XGBoost Arch A Full ----":^34}  {"-- Random Walk --":^20}')
print(f'{"":14}  {"RMSE":>6} {"MAPE%":>7} {"R2":>7} {"Dir%":>7}  {"RMSE":>6} {"R2":>7} {"Dir%":>7}')
print('-'*72)
for r in results:
    m  = r['m_af']
    rw = r['m_rw']
    print(f'{r["commodity"]:<14}  '
          f'{fmt(m["rmse"],2):>6} {fmt(m["mape"],2):>7} {fmt(m["r2"],4):>7} {fmt(r["da_af"],1):>7}  '
          f'{fmt(rw["rmse"],2):>6} {fmt(rw["r2"],4):>7} {fmt(r["da_rw"],1):>7}')

# ======================================================================
# TABLE 5 - ARCH B vs RANDOM WALK (for completeness)
# ======================================================================
print_section('TABLE 5 - Arch B Full vs Random Walk')
print(f'{"Commodity":<14}  {"---- XGBoost Arch B Full ----":^34}  {"-- Random Walk --":^20}')
print(f'{"":14}  {"RMSE":>6} {"MAPE%":>7} {"R2":>7} {"Dir%":>7}  {"RMSE":>6} {"R2":>7}')
print('-'*66)
for r in results:
    m  = r['m_bf']
    rw = r['m_rw']
    print(f'{r["commodity"]:<14}  '
          f'{fmt(m["rmse"],2):>6} {fmt(m["mape"],2):>7} {fmt(m["r2"],4):>7} {fmt(r["da_bf"],1):>7}  '
          f'{fmt(rw["rmse"],2):>6} {fmt(rw["r2"],4):>7}')

print('\n\nNotes:')
print('  * MAPE expressed as %; Dir% = directional accuracy (% correct sign of next-day move).')
print('  * RW Dir% ~0%: the random walk predicts zero change every day, so it never')
print('    forecasts a direction -- wrong whenever the market actually moves.')
print('  * Backtest: long when predicted log-return > 0, else flat. No costs assumed.')
print('  * Sharpe annualised (x sqrt(252)), risk-free rate = 0.')
print('  * Arch B metrics on reconstructed prices are expected to be poor because')
print('    OLS detrending in-sample does not generalise to out-of-sample trend extrapolation.')
print()
