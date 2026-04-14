# =============================================================================
# PIPELINE IDENTITY
#   Dataset             : NO EXTRAS — reads 01a_differencing_energy_dataset.csv
#   Features Used       : Global Only (Brent, NatGas, DXY, VIX, SP500, US_10Yr)
#                         NO local Egyptian data (EGP, Inflation, CBE)
#                         NO engineered columns (no lags, rolling, cross-features)
#   Architecture        : A — DIFFERENCING (Log Returns)
#   Target              : Brent_Crude_Close_LogReturn
# =============================================================================
import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import warnings

warnings.filterwarnings("ignore")

print("Initiating Phase 6: Time-Series Hyperparameter Tuning (Architecture A, No Extras, Global Only)...\n")

# --- 1. CONFIGURATION & DIRECTORY SETUP ---
FINAL_DIR = '../data-energy/03_final/'
DATA_FILE = os.path.join(FINAL_DIR, '01a_differencing_energy_dataset.csv')

# --- 2. DATA LOADING & STRICT TIMELINE SPLIT ---
df = pd.read_csv(DATA_FILE, index_col='Date', parse_dates=['Date'])

# Combine Train + Validation sets for cross-validation tuning.
# Strictly exclude the 2025 Test Set to prevent future data leakage.
TRAIN_VAL_END = '2024-12-31'
train_val_data = df[:TRAIN_VAL_END]

# --- 3. TARGET & LEAKAGE CONTROL ---
# Drop raw prices, current-day stationarity columns, and ALL local Egyptian data.
# This tunes only on the global signal (energy + macro).
drop_cols = [
    # Raw prices — data leakage
    'Brent_Crude_Close', 'Natural_Gas_Close', 'DXY_Close', 'VIX_Close', 'SP500_Close', 'EGP_USD_Close',
    # Current-day log returns — target and its peers (leakage)
    'Brent_Crude_Close_LogReturn', 'Natural_Gas_Close_LogReturn', 'DXY_Close_LogReturn',
    'VIX_Close_LogReturn', 'SP500_Close_LogReturn', 'EGP_USD_Close_LogReturn',
    # Rate differences — leakage
    'US_10Yr_Yield_Diff',
    # Local Egyptian features — excluded by design for this global-only tuning
    'Egypt_Inflation_YoY', 'CBE_Interest_Rate'
]

# Safe-drop in case some columns don't exist in the no-extras dataset
cols_to_drop = [c for c in drop_cols if c in train_val_data.columns]

X = train_val_data.drop(columns=cols_to_drop)
y = train_val_data['Brent_Crude_Close_LogReturn']

print(f"Features fed into Grid Search: {len(X.columns)}")
print("Feature list:")
for col in X.columns:
    print(f"  - {col}")

# --- 4. THE HYPERPARAMETER GRID ---
param_grid = {
    'n_estimators': [500, 1000, 1500],
    'learning_rate': [0.01, 0.05],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# --- 5. TIME-SERIES CROSS VALIDATION ---
# TimeSeriesSplit preserves chronological order — no data from the future leaks into the past.
tscv = TimeSeriesSplit(n_splits=5)
xgb_model = xgb.XGBRegressor(random_state=42)

print("\nCommencing Grid Search (training hundreds of models across chronological folds)...")
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

# --- 6. EXECUTION AND EXTRACTION ---
grid_search.fit(X, y)
best_params = grid_search.best_params_
best_score = -grid_search.best_score_

print("\n" + "="*55)
print("OPTIMAL HYPERPARAMETERS FOUND")
print("="*55)
for param, value in best_params.items():
    print(f"  --> {param}: {value}")

print(f"\nAverage Cross-Validation RMSE (Log-Return space): {best_score:.6f}")
print("="*55)

# --- 7. SAVE TO FILE ---
output_path = '../data-energy/best_xgboost_brent_params.txt'
with open(output_path, 'w') as f:
    f.write("Best Hyperparameters for Brent Crude Architecture A\n")
    f.write("Dataset  : No Extras (no lags / rolling / cross-features)\n")
    f.write("Features : Global Only (no EGP / Inflation / CBE)\n")
    f.write("="*55 + "\n")
    for param, value in best_params.items():
        f.write(f"{param}: {value}\n")
    f.write(f"\nCV RMSE (log-return space): {best_score:.6f}\n")

print(f"\nSaved winning configuration to: {output_path}")