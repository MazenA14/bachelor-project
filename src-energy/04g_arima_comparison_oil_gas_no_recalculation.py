import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMAResults
from sklearn.metrics import root_mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION & DATA LOADING ---
print("Initiating Phase 4g: ARIMA Comparison — Brent Crude Oil vs Natural Gas (No Recalculation)...\n")
PROCESSED_DIR = '../data-energy/02_processed/'
FINAL_DIR     = '../data-energy/03_final/'
MODELS_DIR    = '../models-energy/'
TEST_START    = '2025-01-01'

print("Loading datasets...")
df_master    = pd.read_csv(os.path.join(PROCESSED_DIR, '01_master_energy_dataset.csv'),        index_col='Date', parse_dates=['Date'])
df_a_noextra = pd.read_csv(os.path.join(FINAL_DIR,     '01a_differencing_energy_dataset.csv'), index_col='Date', parse_dates=['Date'])

test_index = df_a_noextra[TEST_START:].index

# --- 2. MODEL LOADING ---
print("Loading ARIMA models from disk...")
arima_oil = ARIMAResults.load(os.path.join(MODELS_DIR, 'arima_baseline_no_extras.pkl'))
arima_gas = ARIMAResults.load(os.path.join(MODELS_DIR, 'arima_baseline_no_extras_gas.pkl'))

# --- 3. FORECASTS ---
print("Generating ARIMA forecasts...")
steps = len(test_index)

oil_preds = arima_oil.forecast(steps=steps)
oil_preds.index = test_index

gas_preds = arima_gas.forecast(steps=steps)
gas_preds.index = test_index

actual_oil = df_master.loc[test_index, 'Brent_Crude_Close']
actual_gas = df_master.loc[test_index, 'Natural_Gas_Close']

# --- 4. METRICS ---
def metrics(actual, pred):
    return dict(
        rmse = root_mean_squared_error(actual, pred),
    )

m_oil = metrics(actual_oil, oil_preds)
m_gas = metrics(actual_gas, gas_preds)

print("\n=== ARIMA EVALUATION RESULTS (No Walk-Forward Recalculation) ===")
print(f"Brent Crude Oil — RMSE=${m_oil['rmse']:.2f}")
print(f"Natural Gas     — RMSE=${m_gas['rmse']:.2f}\n")

# --- 5. PLOT ---
print("Generating side-by-side ARIMA forecast plot...")
fig, (ax_oil, ax_gas) = plt.subplots(1, 2, figsize=(20, 7))
fig.suptitle(
    'ARIMA Baseline Forecast — Brent Crude Oil vs Natural Gas\nTest Period (January 2025 – Present) | Saved Model, No Walk-Forward Recalculation',
    fontsize=14, fontweight='bold'
)

for ax, actual, preds, m, commodity, color, unit in [
    (ax_oil, actual_oil, oil_preds, m_oil, 'Brent Crude Oil', 'saddlebrown', 'USD/bbl'),
    (ax_gas, actual_gas, gas_preds, m_gas, 'Natural Gas',     'steelblue',  'USD/MMBtu'),
]:
    ax.plot(actual.index, actual, label='Actual Closing Price',
            color='black', linewidth=1.8, linestyle='-')
    ax.plot(preds.index, preds,
            label=f'ARIMA Prediction (RMSE: ${m["rmse"]:.2f})',
            color=color, linewidth=1.5, linestyle='--')
    ax.set_title(f'{commodity} Price Forecast (ARIMA — No Recalculation)', fontsize=12, pad=10)
    ax.set_ylabel(f'{commodity} Price ({unit})', fontsize=10)
    ax.set_xlabel('Date', fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()
