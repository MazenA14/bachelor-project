import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMAResults
from sklearn.metrics import root_mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION & DATA LOADING ---
print("Initiating Phase 4g: ARIMA Comparison — Gold vs Silver (No Recalculation)...\n")
PROCESSED_DIR = '../data-extra-variables/02_processed/'
FINAL_DIR     = '../data-extra-variables/03_final/'
MODELS_DIR    = '../models-extra-variables/'
TEST_START    = '2025-01-01'

print("Loading datasets...")
df_master    = pd.read_csv(os.path.join(PROCESSED_DIR, '01_master_metals_dataset.csv'), index_col='Date', parse_dates=['Date'])
df_a_noextra = pd.read_csv(os.path.join(FINAL_DIR,     '01a_differencing_metals_dataset.csv'), index_col='Date', parse_dates=['Date'])

test_index = df_a_noextra[TEST_START:].index

# --- 2. MODEL LOADING ---
print("Loading ARIMA models from disk...")
arima_gold   = ARIMAResults.load(os.path.join(MODELS_DIR, 'arima_baseline_no_extras.pkl'))
arima_silver = ARIMAResults.load(os.path.join(MODELS_DIR, 'arima_baseline_no_extras_silver.pkl'))

# --- 3. FORECASTS ---
print("Generating ARIMA forecasts...")
steps = len(test_index)

gold_preds   = arima_gold.forecast(steps=steps)
gold_preds.index = test_index

silver_preds = arima_silver.forecast(steps=steps)
silver_preds.index = test_index

actual_gold   = df_master.loc[test_index, 'Gold_Close']
actual_silver = df_master.loc[test_index, 'Silver_Close']

# --- 4. METRICS ---
def metrics(actual, pred):
    return dict(
        rmse = root_mean_squared_error(actual, pred),
    )

m_gold   = metrics(actual_gold,   gold_preds)
m_silver = metrics(actual_silver, silver_preds)

print("\n=== ARIMA EVALUATION RESULTS (No Walk-Forward Recalculation) ===")
print(f"Gold   — RMSE=${m_gold['rmse']:.2f}")
print(f"Silver — RMSE=${m_silver['rmse']:.2f}\n")

# --- 5. PLOT ---
print("Generating side-by-side ARIMA forecast plot...")
fig, (ax_gold, ax_silver) = plt.subplots(1, 2, figsize=(20, 7))
fig.suptitle(
    'ARIMA Baseline Forecast — Gold vs Silver\nTest Period (January 2025 – Present) | Saved Model, No Walk-Forward Recalculation',
    fontsize=14, fontweight='bold'
)

for ax, actual, preds, m, metal, color, unit in [
    (ax_gold,   actual_gold,   gold_preds,   m_gold,   'Gold',   'goldenrod', 'USD/oz'),
    (ax_silver, actual_silver, silver_preds, m_silver, 'Silver', 'slategray', 'USD/oz'),
]:
    ax.plot(actual.index, actual, label='Actual Closing Price',
            color='black', linewidth=1.8, linestyle='-')
    ax.plot(preds.index, preds,
            label=f'ARIMA Prediction (RMSE: ${m["rmse"]:.2f})',
            color=color, linewidth=1.5, linestyle='--')
    ax.set_title(f'{metal} Price Forecast (ARIMA — No Recalculation)', fontsize=12, pad=10)
    ax.set_ylabel(f'{metal} Price ({unit})', fontsize=10)
    ax.set_xlabel('Date', fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()
