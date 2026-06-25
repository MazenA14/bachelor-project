import os
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

MODELS_DIR = 'models-extra-variables'

xgb_full = xgb.XGBRegressor()
xgb_full.load_model(os.path.join(MODELS_DIR, 'xgboost_a.json'))

xgb_noextra = xgb.XGBRegressor()
xgb_noextra.load_model(os.path.join(MODELS_DIR, 'xgboost_a_no_extras.json'))


def rename_feature(name):
    special = {'Gold_Silver_Ratio': 'Gold / Silver Price Ratio'}
    if name in special:
        return special[name]
    asset_labels = {
        'Gold':    'Gold',
        'Silver':  'Silver',
        'DXY':     'US Dollar Index (DXY)',
        'SP500':   'S&P 500 Index',
        'VIX':     'VIX Volatility Index',
        'EGP_USD': 'Egyptian Pound / US Dollar',
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
        if name.startswith(asset_key + '_'):
            field = name[len(asset_key) + 1:]
            return f'{asset_name} — {field_labels.get(field, field)}'
    return name


def get_importance_series(model, top_n=10):
    scores = model.get_booster().get_fscore()
    series = pd.Series(scores).sort_values(ascending=False).head(top_n).sort_values(ascending=True)
    series.index = [rename_feature(f) for f in series.index]
    return series


imp_full = get_importance_series(xgb_full)
imp_noextra = get_importance_series(xgb_noextra)

x_max = max(imp_full.max(), imp_noextra.max()) * 1.05

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))

imp_full.plot(kind='barh', ax=ax1, color='steelblue')
ax1.set_title('(Log Returns Stationarity)', fontsize=22, pad=16)
ax1.set_xlabel('F-Score (Number of Splits)', fontsize=18)
ax1.set_xlim(0, x_max)
ax1.tick_params(axis='both', labelsize=17)
ax1.grid(True, alpha=0.3, axis='x')

imp_noextra.plot(kind='barh', ax=ax2, color='steelblue')
ax2.set_title('(Log Returns Stationarity)', fontsize=22, pad=16)
ax2.set_xlabel('F-Score (Number of Splits)', fontsize=18)
ax2.set_xlim(0, x_max)
ax2.tick_params(axis='both', labelsize=17)
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('log_returns_stationarity.png', dpi=150, bbox_inches='tight')
plt.show()
