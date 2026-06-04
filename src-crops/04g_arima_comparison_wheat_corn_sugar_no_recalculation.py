import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMAResults
from sklearn.metrics import root_mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION & DATA LOADING ---
print("Initiating Phase 4g: ARIMA Comparison — Wheat vs Corn vs Sugar (No Recalculation)...\n")
PROCESSED_DIR = '../data-crops/02_processed/'
FINAL_DIR     = '../data-crops/03_final/'
MODELS_DIR    = '../models-crops/'
TEST_START    = '2025-01-01'

print("Loading datasets...")
df_master    = pd.read_csv(os.path.join(PROCESSED_DIR, '01_master_crops_dataset.csv'), index_col='Date', parse_dates=['Date'])
df_a_noextra = pd.read_csv(os.path.join(FINAL_DIR,     '01a_differencing_crops_dataset.csv'), index_col='Date', parse_dates=['Date'])

test_index = df_a_noextra[TEST_START:].index

# --- 2. MODEL LOADING ---
print("Loading ARIMA models from disk...")
arima_wheat = ARIMAResults.load(os.path.join(MODELS_DIR, 'arima_baseline_no_extras.pkl'))
arima_corn  = ARIMAResults.load(os.path.join(MODELS_DIR, 'arima_baseline_no_extras_corn.pkl'))
arima_sugar = ARIMAResults.load(os.path.join(MODELS_DIR, 'arima_baseline_no_extras_sugar.pkl'))

# --- 3. FORECASTS ---
print("Generating ARIMA forecasts...")
steps = len(test_index)

wheat_preds = arima_wheat.forecast(steps=steps)
wheat_preds.index = test_index

corn_preds  = arima_corn.forecast(steps=steps)
corn_preds.index = test_index

sugar_preds = arima_sugar.forecast(steps=steps)
sugar_preds.index = test_index

actual_wheat = df_master.loc[test_index, 'Wheat_Close']
actual_corn  = df_master.loc[test_index, 'Corn_Close']
actual_sugar = df_master.loc[test_index, 'Sugar_Close']

# --- 4. METRICS ---
def metrics(actual, pred):
    return dict(rmse=root_mean_squared_error(actual, pred))

m_wheat = metrics(actual_wheat, wheat_preds)
m_corn  = metrics(actual_corn,  corn_preds)
m_sugar = metrics(actual_sugar, sugar_preds)

print("\n=== ARIMA EVALUATION RESULTS (No Walk-Forward Recalculation) ===")
print(f"Wheat — RMSE=${m_wheat['rmse']:.2f}")
print(f"Corn  — RMSE=${m_corn['rmse']:.2f}")
print(f"Sugar — RMSE=${m_sugar['rmse']:.2f}\n")

# --- 5. PLOT ---
print("Generating 3-panel ARIMA forecast plot...")
fig, (ax_wheat, ax_corn, ax_sugar) = plt.subplots(1, 3, figsize=(28, 7))
fig.suptitle(
    'ARIMA Baseline Forecast — Wheat vs Corn vs Sugar\nTest Period (January 2025 – Present) | Saved Model, No Walk-Forward Recalculation',
    fontsize=14, fontweight='bold'
)

for ax, actual, preds, m, commodity, color, unit in [
    (ax_wheat, actual_wheat, wheat_preds, m_wheat, 'Wheat', 'goldenrod',  'USD/Bu'),
    (ax_corn,  actual_corn,  corn_preds,  m_corn,  'Corn',  'forestgreen', 'USD/Bu'),
    (ax_sugar, actual_sugar, sugar_preds, m_sugar, 'Sugar', 'firebrick',   'USD/lb'),
]:
    ax.plot(actual.index, actual,
            label='Actual Closing Price',
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
