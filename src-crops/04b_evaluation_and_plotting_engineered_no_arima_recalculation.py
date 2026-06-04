# =============================================================================
# PIPELINE IDENTITY
#   Commodity           : Wheat (Crops Pipeline)
#   Extra Features      : YES (Engineered) + NO EXTRAS — all 4 variants compared
#   Architecture A      : DIFFERENCING  — reads 01a_engineered / 01a_differencing datasets
#   Architecture B      : DETRENDING    — reads 01b_engineered / 01b_detrending datasets
#   ARIMA               : LOADED from saved .pkl (fast, uses pre-trained model from Phase 3)
#   Models loaded from  : ../models-crops/  (arima_baseline.pkl)
# =============================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMAResults
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

print("Initiating Phase 4: Model Evaluation — Crops (Saved ARIMA, No Recalculation)...\n")
PROCESSED_DIR = '../data-crops/02_processed/'
FINAL_DIR     = '../data-crops/03_final/'
MODELS_DIR    = '../models-crops/'
TEST_START    = '2025-01-01'
TRAIN_VAL_END = '2024-12-31'

print("Loading datasets...")
df_master    = pd.read_csv(os.path.join(PROCESSED_DIR, '01_master_crops_dataset.csv'),                    index_col='Date', parse_dates=['Date'])
df_a_full    = pd.read_csv(os.path.join(FINAL_DIR,     '01a_engineered_differencing_crops_dataset.csv'), index_col='Date', parse_dates=['Date'])
df_b_full    = pd.read_csv(os.path.join(FINAL_DIR,     '01b_engineered_detrending_crops_dataset.csv'),   index_col='Date', parse_dates=['Date'])
df_a_noextra = pd.read_csv(os.path.join(FINAL_DIR,     '01a_differencing_crops_dataset.csv'),            index_col='Date', parse_dates=['Date'])
df_b_noextra = pd.read_csv(os.path.join(FINAL_DIR,     '01b_detrending_crops_dataset.csv'),              index_col='Date', parse_dates=['Date'])

test_a_full    = df_a_full[TEST_START:]
test_b_full    = df_b_full[TEST_START:]
test_a_noextra = df_a_noextra[TEST_START:]
test_b_noextra = df_b_noextra[TEST_START:]

drop_a = [
    'Wheat_Close', 'Corn_Close', 'Sugar_Close', 'Brent_Crude_Close', 'DXY_Close', 'EGP_USD_Close',
    'Wheat_Close_LogReturn', 'Egypt_Inflation_YoY', 'CBE_Interest_Rate'
]
drop_b = [
    'Wheat_Close', 'Corn_Close', 'Sugar_Close', 'Brent_Crude_Close', 'DXY_Close', 'EGP_USD_Close',
    'Wheat_Close_Trend', 'Wheat_Close_Residual',
    'Corn_Close_Trend', 'Sugar_Close_Trend', 'Brent_Crude_Close_Trend',
    'DXY_Close_Trend', 'EGP_USD_Close_Trend'
]

X_test_a_full    = test_a_full.drop(columns=drop_a)
X_test_b_full    = test_b_full.drop(columns=drop_b)
X_test_a_noextra = test_a_noextra.drop(columns=drop_a)
X_test_b_noextra = test_b_noextra.drop(columns=drop_b)

# --- 2. MODEL LOADING ---
print("Loading trained models from disk...")
arima_fitted  = ARIMAResults.load(os.path.join(MODELS_DIR, 'arima_baseline.pkl'))

xgb_a_full    = xgb.XGBRegressor(); xgb_a_full.load_model(os.path.join(MODELS_DIR, 'xgboost_a.json'))
xgb_b_full    = xgb.XGBRegressor(); xgb_b_full.load_model(os.path.join(MODELS_DIR, 'xgboost_b.json'))
xgb_a_noextra = xgb.XGBRegressor(); xgb_a_noextra.load_model(os.path.join(MODELS_DIR, 'xgboost_a_no_extras.json'))
xgb_b_noextra = xgb.XGBRegressor(); xgb_b_noextra.load_model(os.path.join(MODELS_DIR, 'xgboost_b_no_extras.json'))

# --- 3. ARIMA BASELINE FORECAST (from saved model, no walk-forward) ---
print("Generating ARIMA forecast from saved model...")
arima_predictions = arima_fitted.forecast(steps=len(test_a_full))
arima_predictions.index = test_a_full.index

# --- 4. XGBOOST PREDICTIONS & PRICE REVERSAL ---
print("Generating XGBoost predictions...")
actual_prices    = df_master.loc[test_a_full.index, 'Wheat_Close']
last_val_date    = df_master[:'2024-12-31'].index[-1]
last_train_price = df_master.loc[last_val_date, 'Wheat_Close']


def reverse_log_returns(preds_log, test_index):
    yesterday = df_master.loc[test_index, 'Wheat_Close'].shift(1)
    yesterday.iloc[0] = last_train_price
    return yesterday * np.exp(preds_log)


def reverse_detrend(preds_resid, df_source, test_index):
    return df_source.loc[test_index, 'Wheat_Close_Trend'] + preds_resid


xgb_a_full_prices    = reverse_log_returns(xgb_a_full.predict(X_test_a_full),          X_test_a_full.index)
xgb_a_noextra_prices = reverse_log_returns(xgb_a_noextra.predict(X_test_a_noextra),    X_test_a_noextra.index)
xgb_b_full_prices    = reverse_detrend(xgb_b_full.predict(X_test_b_full),    df_b_full,    X_test_b_full.index)
xgb_b_noextra_prices = reverse_detrend(xgb_b_noextra.predict(X_test_b_noextra), df_b_noextra, X_test_b_noextra.index)

# --- 5. METRICS ---
def metrics(actual, pred):
    return dict(
        rmse = root_mean_squared_error(actual, pred),
        mape = mean_absolute_percentage_error(actual, pred),
        mae  = mean_absolute_error(actual, pred),
        mse  = mean_squared_error(actual, pred),
        r2   = r2_score(actual, pred),
    )

m_arima     = metrics(actual_prices, arima_predictions)
m_a_full    = metrics(actual_prices, xgb_a_full_prices)
m_a_noextra = metrics(actual_prices, xgb_a_noextra_prices)
m_b_full    = metrics(actual_prices, xgb_b_full_prices)
m_b_noextra = metrics(actual_prices, xgb_b_noextra_prices)

print("\n=== EVALUATION RESULTS — WHEAT (Test Set: January 2025 – Present) ===")
print(f"ARIMA Baseline (Saved Model):                  RMSE=${m_arima['rmse']:.2f}  MAPE={m_arima['mape']:.4f}  MAE={m_arima['mae']:.2f}  R²={m_arima['r2']:.4f}")
print(f"Architecture A — Full Features (Log Returns):  RMSE=${m_a_full['rmse']:.2f}  MAPE={m_a_full['mape']:.4f}  MAE={m_a_full['mae']:.2f}  R²={m_a_full['r2']:.4f}")
print(f"Architecture A — No Extra Features:            RMSE=${m_a_noextra['rmse']:.2f}  MAPE={m_a_noextra['mape']:.4f}  MAE={m_a_noextra['mae']:.2f}  R²={m_a_noextra['r2']:.4f}")
print(f"Architecture B — Full Features (Detrending):   RMSE=${m_b_full['rmse']:.2f}  MAPE={m_b_full['mape']:.4f}  MAE={m_b_full['mae']:.2f}  R²={m_b_full['r2']:.4f}")
print(f"Architecture B — No Extra Features:            RMSE=${m_b_noextra['rmse']:.2f}  MAPE={m_b_noextra['mape']:.4f}  MAE={m_b_noextra['mae']:.2f}  R²={m_b_noextra['r2']:.4f}\n")

# --- 6. FEATURE NAME MAPPING ---
def rename_feature(name):
    asset_labels = {
        'Brent_Crude': 'Brent Crude Oil',
        'EGP_USD':     'Egyptian Pound / US Dollar',
        'Wheat':       'Wheat',
        'Corn':        'Corn',
        'Sugar':       'Sugar',
        'DXY':         'US Dollar Index (DXY)',
    }
    field_labels = {
        'Open':         'Opening Price',
        'High':         'Daily High Price',
        'Low':          'Daily Low Price',
        'Volume':       'Trading Volume',
        'Close_Lag1':   'Closing Price — 1-Day Lag',
        'Close_Lag3':   'Closing Price — 3-Day Lag',
        'Close_Lag7':   'Closing Price — 7-Day Lag',
        'Close_Roll14': 'Closing Price — 14-Day Rolling Average',
    }
    for asset_key, asset_name in asset_labels.items():
        prefix = asset_key + '_'
        if name.startswith(prefix):
            field = name[len(prefix):]
            return f'{asset_name} — {field_labels.get(field, field)}'
    return name


def get_importance_series(model, top_n=10):
    scores = model.get_booster().get_fscore()
    series = pd.Series(scores).sort_values(ascending=False).head(top_n).sort_values(ascending=True)
    series.index = [rename_feature(f) for f in series.index]
    return series

# --- 7. FORECAST VISUALIZATION: 2x2 PANEL GRID ---
print("Generating 2x2 forecast panel plot...")
panels = [
    {'pos': (0, 0), 'title': 'Architecture A — Full Features\n(Log Returns Stationarity with Lagged, Rolling & Cross Features)',
     'xgb': xgb_a_full_prices,    'rmse': m_a_full['rmse']},
    {'pos': (0, 1), 'title': 'Architecture A — No Extra Features\n(Log Returns Stationarity, Baseline Feature Set Only)',
     'xgb': xgb_a_noextra_prices, 'rmse': m_a_noextra['rmse']},
    {'pos': (1, 0), 'title': 'Architecture B — Full Features\n(Linear Detrending with Lagged, Rolling & Cross Features)',
     'xgb': xgb_b_full_prices,    'rmse': m_b_full['rmse']},
    {'pos': (1, 1), 'title': 'Architecture B — No Extra Features\n(Linear Detrending, Baseline Feature Set Only)',
     'xgb': xgb_b_noextra_prices, 'rmse': m_b_noextra['rmse']},
]

fig, axes = plt.subplots(2, 2, figsize=(22, 18))
fig.suptitle('Wheat Price Forecast — Test Period (January 2025 – Present)\nARIMA: Saved Model (No Walk-Forward Recalculation)', fontsize=14, fontweight='bold')

for p in panels:
    row, col = p['pos']
    ax = axes[row][col]
    ax.plot(actual_prices.index,     actual_prices,     label='Actual Closing Price',
            color='black',      linewidth=1.8, linestyle='-')
    ax.plot(p['xgb'].index,          p['xgb'],          label=f'XGBoost Prediction (RMSE: ${p["rmse"]:.2f})',
            color='steelblue',  linewidth=1.4, linestyle='--')
    ax.plot(arima_predictions.index, arima_predictions, label=f'ARIMA Prediction (RMSE: ${m_arima["rmse"]:.2f})',
            color='darkorange', linewidth=1.2, linestyle=':')
    ax.set_title(p['title'], fontsize=11, pad=10)
    ax.set_ylabel('Wheat Price (USD/Bu)', fontsize=10)
    ax.set_xlabel('Date', fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

plt.subplots_adjust(left=0.08, right=0.97, top=0.87, bottom=0.07, hspace=0.55, wspace=0.25)
plt.show()

# --- 8. FEATURE IMPORTANCE: 2x2 PANEL GRID ---
print("Generating 2x2 feature importance panel plot...")
imp_panels = [
    {'pos': (0, 0), 'model': xgb_a_full,    'title': 'Architecture A — Full Features\n(Log Returns Stationarity)'},
    {'pos': (0, 1), 'model': xgb_a_noextra, 'title': 'Architecture A — No Extra Features\n(Log Returns Stationarity)'},
    {'pos': (1, 0), 'model': xgb_b_full,    'title': 'Architecture B — Full Features\n(Linear Detrending)'},
    {'pos': (1, 1), 'model': xgb_b_noextra, 'title': 'Architecture B — No Extra Features\n(Linear Detrending)'},
]

fig2, axes2 = plt.subplots(2, 2, figsize=(26, 16))
fig2.suptitle('Top 10 Most Important Features by Model Variant — Wheat', fontsize=14, fontweight='bold')

for p in imp_panels:
    row, col = p['pos']
    ax = axes2[row][col]
    imp = get_importance_series(p['model'])
    imp.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(p['title'], fontsize=11, pad=12)
    ax.set_xlabel('F-Score (Number of Splits)', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')

plt.subplots_adjust(left=0.42, hspace=0.5, wspace=0.6, top=0.91)
plt.show()
