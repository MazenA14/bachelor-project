"""
Generates two time-series overlay figures comparing Full vs Reduced feature sets:
  - Gold:        Full (with Silver) vs Reduced (Silver removed)
  - Brent Crude: Full (with Nat Gas) vs Reduced (Nat Gas removed)
Both use the Global + Local (Hybrid) feature set.
"""
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
import warnings

warnings.filterwarnings("ignore")

TRAIN_END  = '2023-12-31'
VAL_START  = '2024-01-01'
VAL_END    = '2024-12-31'
TEST_START = '2025-01-01'


def split(df):
    return df[:TRAIN_END], df[VAL_START:VAL_END], df[TEST_START:]


def make_X(df, drop_cols):
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


def strip_cross(X, cross_kw):
    keep = [c for c in X.columns if not any(kw in c for kw in cross_kw)]
    return X[keep]


def reverse_log_return(df_master, price_col, index, preds):
    yp = df_master.loc[index, price_col].shift(1)
    yp.iloc[0] = df_master.loc['2024-12-31', price_col]
    return yp * np.exp(preds)


def train_predict(X_tr, X_va, X_te, y_tr, y_va, params):
    m = xgb.XGBRegressor(**params)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return m.predict(X_te)


# ============================================================
#  GOLD  (Extra-Variables)
# ============================================================
print("Loading extra-variables data...")
ROOT_EV   = './data-extra-variables/'
master_ev = pd.read_csv(ROOT_EV + '02_processed/01_master_metals_dataset.csv',
                        index_col='Date', parse_dates=['Date'])
diff_ev   = pd.read_csv(ROOT_EV + '03_final/01a_differencing_metals_dataset.csv',
                        index_col='Date', parse_dates=['Date'])

tr_ev, va_ev, te_ev = split(diff_ev)

EV_RAW = ['Gold_Close', 'Silver_Close', 'DXY_Close', 'EGP_USD_Close',
          'DXY_Close_LogReturn', 'EGP_USD_Close_LogReturn']
EV_PARAMS = dict(n_estimators=500, learning_rate=0.01, max_depth=3,
                 subsample=0.8, colsample_bytree=0.8, random_state=42)

gold_base_drop = EV_RAW + ['Gold_Close_LogReturn']
X_tr_gold = make_X(tr_ev, gold_base_drop)
X_va_gold = make_X(va_ev, gold_base_drop)
X_te_gold = make_X(te_ev, gold_base_drop)

X_tr_gold_r = strip_cross(X_tr_gold, ['Silver', 'Ratio'])
X_va_gold_r = strip_cross(X_va_gold, ['Silver', 'Ratio'])
X_te_gold_r = strip_cross(X_te_gold, ['Silver', 'Ratio'])

y_tr_gold = tr_ev['Gold_Close_LogReturn']
y_va_gold = va_ev['Gold_Close_LogReturn']

print(f"Gold — Full: {len(X_tr_gold.columns)} features, Reduced: {len(X_tr_gold_r.columns)} features")

print("Training Gold Full model...")
preds_gold_full = train_predict(X_tr_gold,   X_va_gold,   X_te_gold,   y_tr_gold, y_va_gold, EV_PARAMS)
print("Training Gold Reduced model...")
preds_gold_red  = train_predict(X_tr_gold_r, X_va_gold_r, X_te_gold_r, y_tr_gold, y_va_gold, EV_PARAMS)

actual_gold     = master_ev.loc[te_ev.index, 'Gold_Close']
price_gold_full = reverse_log_return(master_ev, 'Gold_Close', te_ev.index, preds_gold_full)
price_gold_red  = reverse_log_return(master_ev, 'Gold_Close', te_ev.index, preds_gold_red)

rmse_gold_full = root_mean_squared_error(actual_gold, price_gold_full)
rmse_gold_red  = root_mean_squared_error(actual_gold, price_gold_red)
mape_gold_full = mean_absolute_percentage_error(actual_gold, price_gold_full)
mape_gold_red  = mean_absolute_percentage_error(actual_gold, price_gold_red)

print(f"  Full   RMSE: ${rmse_gold_full:.2f}  MAPE: {mape_gold_full*100:.2f}%")
print(f"  Reduced RMSE: ${rmse_gold_red:.2f}  MAPE: {mape_gold_red*100:.2f}%")

# ============================================================
#  BRENT CRUDE  (Energy)
# ============================================================
print("\nLoading energy data...")
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

brent_base_drop = EN_RAW + ['Brent_Crude_Close_LogReturn']
X_tr_brent = make_X(tr_en, brent_base_drop)
X_va_brent = make_X(va_en, brent_base_drop)
X_te_brent = make_X(te_en, brent_base_drop)

X_tr_brent_r = strip_cross(X_tr_brent, ['Natural_Gas', 'NatGas', 'Ratio'])
X_va_brent_r = strip_cross(X_va_brent, ['Natural_Gas', 'NatGas', 'Ratio'])
X_te_brent_r = strip_cross(X_te_brent, ['Natural_Gas', 'NatGas', 'Ratio'])

y_tr_brent = tr_en['Brent_Crude_Close_LogReturn']
y_va_brent = va_en['Brent_Crude_Close_LogReturn']

print(f"Brent — Full: {len(X_tr_brent.columns)} features, Reduced: {len(X_tr_brent_r.columns)} features")

print("Training Brent Full model...")
preds_brent_full = train_predict(X_tr_brent,   X_va_brent,   X_te_brent,   y_tr_brent, y_va_brent, EN_PARAMS)
print("Training Brent Reduced model...")
preds_brent_red  = train_predict(X_tr_brent_r, X_va_brent_r, X_te_brent_r, y_tr_brent, y_va_brent, EN_PARAMS)

actual_brent      = master_en.loc[te_en.index, 'Brent_Crude_Close']
price_brent_full  = reverse_log_return(master_en, 'Brent_Crude_Close', te_en.index, preds_brent_full)
price_brent_red   = reverse_log_return(master_en, 'Brent_Crude_Close', te_en.index, preds_brent_red)

rmse_brent_full = root_mean_squared_error(actual_brent, price_brent_full)
rmse_brent_red  = root_mean_squared_error(actual_brent, price_brent_red)
mape_brent_full = mean_absolute_percentage_error(actual_brent, price_brent_full)
mape_brent_red  = mean_absolute_percentage_error(actual_brent, price_brent_red)

print(f"  Full   RMSE: ${rmse_brent_full:.2f}  MAPE: {mape_brent_full*100:.2f}%")
print(f"  Reduced RMSE: ${rmse_brent_red:.2f}  MAPE: {mape_brent_red*100:.2f}%")

# ============================================================
#  PLOT 1 — GOLD
# ============================================================
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(actual_gold.index, actual_gold,
        label='Actual Gold Price', color='black', linewidth=1.8)
ax.plot(price_gold_full.index, price_gold_full,
        label=f'Full (with Silver)  — RMSE: ${rmse_gold_full:.2f} | MAPE: {mape_gold_full*100:.2f}%',
        color='goldenrod', linewidth=1.6)
ax.plot(price_gold_red.index, price_gold_red,
        label=f'Reduced (no Silver) — RMSE: ${rmse_gold_red:.2f} | MAPE: {mape_gold_red*100:.2f}%',
        color='steelblue', linewidth=1.6, linestyle='--')

ax.set_title('Gold (GC=F): Full vs. Reduced Feature Set — Test Period Forecast',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Gold Price (USD/oz)', fontsize=11)
ax.set_xlabel('Date', fontsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=30, ha='right')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./figure_gold_full_vs_reduced.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figure_gold_full_vs_reduced.png")

# ============================================================
#  PLOT 2 — BRENT CRUDE
# ============================================================
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(actual_brent.index, actual_brent,
        label='Actual Brent Crude Price', color='black', linewidth=1.8)
ax.plot(price_brent_full.index, price_brent_full,
        label=f'Full (with Nat Gas)  — RMSE: ${rmse_brent_full:.2f} | MAPE: {mape_brent_full*100:.2f}%',
        color='saddlebrown', linewidth=1.6)
ax.plot(price_brent_red.index, price_brent_red,
        label=f'Reduced (no Nat Gas) — RMSE: ${rmse_brent_red:.2f} | MAPE: {mape_brent_red*100:.2f}%',
        color='steelblue', linewidth=1.6, linestyle='--')

ax.set_title('Brent Crude Oil (BZ=F): Full vs. Reduced Feature Set — Test Period Forecast',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Brent Crude Price (USD/bbl)', fontsize=11)
ax.set_xlabel('Date', fontsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=30, ha='right')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./figure_brent_full_vs_reduced.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figure_brent_full_vs_reduced.png")
