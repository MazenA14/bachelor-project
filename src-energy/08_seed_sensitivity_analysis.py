# =============================================================================
# SEED SENSITIVITY ANALYSIS — Architecture A Full (XGBoost, Log Returns)
#   Verifies that XGBoost test-set results are not sensitive to random_state.
#   Seeds tested : 0, 7, 42 (original), 123
#   Commodities  : Brent Crude Oil  |  Natural Gas
#   Output       : RMSE on actual prices (test set: 2025-01-01 onwards)
#                  + percentage variation relative to seed-42 baseline
# =============================================================================
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
import warnings

warnings.filterwarnings("ignore")

OUTPUT_FILE = '../results-energy/08_seed_sensitivity_results.txt'
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

class Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, path):
        self.file = open(path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, msg):
        self.stdout.write(msg)
        self.file.write(msg)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()

tee = Tee(OUTPUT_FILE)
sys.stdout = tee

PROCESSED_DIR = '../data-energy/02_processed/'
FINAL_DIR     = '../data-energy/03_final/'
TRAIN_END     = '2023-12-31'
VAL_START     = '2024-01-01'
VAL_END       = '2024-12-31'
TEST_START    = '2025-01-01'
SEEDS         = [0, 7, 42, 123]

print("Loading data...")
df_master = pd.read_csv(
    os.path.join(PROCESSED_DIR, '01_master_energy_dataset.csv'),
    index_col='Date', parse_dates=['Date']
)
df_a = pd.read_csv(
    os.path.join(FINAL_DIR, '01a_engineered_differencing_energy_dataset.csv'),
    index_col='Date', parse_dates=['Date']
)

train = df_a[:TRAIN_END]
val   = df_a[VAL_START:VAL_END]
test  = df_a[TEST_START:]

# Both commodities share the same feature set — only the target column differs
drop_cols = [
    'Brent_Crude_Close', 'Natural_Gas_Close', 'DXY_Close', 'VIX_Close', 'SP500_Close', 'EGP_USD_Close',
    'Brent_Crude_Close_LogReturn', 'Natural_Gas_Close_LogReturn', 'DXY_Close_LogReturn',
    'VIX_Close_LogReturn', 'SP500_Close_LogReturn', 'EGP_USD_Close_LogReturn',
    'US_10Yr_Yield_Diff', 'Egypt_Inflation_YoY', 'CBE_Interest_Rate'
]

X_train = train.drop(columns=drop_cols)
X_val   = val.drop(columns=drop_cols)
X_test  = test.drop(columns=drop_cols)

COMMODITIES = [
    {
        'label':        'Brent Crude Oil',
        'target':       'Brent_Crude_Close_LogReturn',
        'price_col':    'Brent_Crude_Close',
    },
    {
        'label':        'Natural Gas',
        'target':       'Natural_Gas_Close_LogReturn',
        'price_col':    'Natural_Gas_Close',
    },
]

all_results = []

for commodity in COMMODITIES:
    label     = commodity['label']
    target    = commodity['target']
    price_col = commodity['price_col']

    y_train = train[target]
    y_val   = val[target]

    actual_prices    = df_master.loc[test.index, price_col]
    last_train_price = df_master.loc[TRAIN_END, price_col]

    def reverse_log_returns(preds_log, test_index, _last=last_train_price, _col=price_col):
        yesterday = df_master.loc[test_index, _col].shift(1)
        yesterday.iloc[0] = _last
        return yesterday * np.exp(preds_log)

    print(f"\nRunning seed sensitivity for {label} (seeds: {SEEDS})...")
    rows = []
    for seed in SEEDS:
        model = xgb.XGBRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=5,
            random_state=seed, early_stopping_rounds=50
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        pred_prices = reverse_log_returns(model.predict(X_test), X_test.index)
        rmse        = root_mean_squared_error(actual_prices, pred_prices)
        best_round  = model.best_iteration + 1
        rows.append({'Commodity': label, 'Seed': seed, 'RMSE': rmse, 'Best Round': best_round})
        print(f"  Seed {seed:>3d} | RMSE = {rmse:.4f} | Best Round = {best_round}")

    df_rows = pd.DataFrame(rows)
    baseline = df_rows.loc[df_rows['Seed'] == 42, 'RMSE'].values[0]
    df_rows['Delta vs seed-42 (%)'] = ((df_rows['RMSE'] - baseline) / baseline * 100).round(2)
    all_results.append(df_rows)

df_all = pd.concat(all_results, ignore_index=True)

print("\n" + "=" * 70)
print("SEED SENSITIVITY RESULTS — Architecture A Full (XGBoost, Log Returns)")
print("=" * 70)
for label in [c['label'] for c in COMMODITIES]:
    subset = df_all[df_all['Commodity'] == label].drop(columns='Commodity')
    max_var = subset['Delta vs seed-42 (%)'].abs().max()
    print(f"\n{label}")
    print("-" * 55)
    print(subset.to_string(index=False))
    print(f"  Max variation: {max_var:.2f}%  ->  {'ROBUST' if max_var < 5 else 'SENSITIVE'}")
print("=" * 70)

sys.stdout = tee.stdout
tee.close()
print(f"\nResults saved to: {os.path.abspath(OUTPUT_FILE)}")
