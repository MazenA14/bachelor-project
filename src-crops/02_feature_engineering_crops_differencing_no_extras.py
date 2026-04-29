# =============================================================================
# PIPELINE IDENTITY
#   Stationarity Method : DIFFERENCING  (Log Returns: R_t = ln(P_t / P_{t-1}))
#   Extra Features      : NO (Only raw prices + log return columns — no lags/rolling/cross)
#   Output dataset      : 01a_differencing_crops_dataset.csv
#   Used by             : 03_model_training_no_extras.py  (XGBoost Architecture A)
#                         04c / 04d evaluation scripts
# =============================================================================
import os
import pandas as pd
import numpy as np

# --- 1. CONFIGURATION & DIRECTORY SETUP ---
PROCESSED_DIR = '../data-crops/02_processed/'
FINAL_DIR = '../data-crops/03_final/'

# Ensure the final output directory exists
os.makedirs(FINAL_DIR, exist_ok=True)

input_file = os.path.join(PROCESSED_DIR, '01_master_crops_dataset.csv')
output_file = os.path.join(FINAL_DIR, '01a_differencing_crops_dataset.csv')

print("Initiating Differencing Pipeline (No Extra Columns)...")

# --- 2. LOAD THE MASTER DATASET ---
try:
    df = pd.read_csv(input_file, index_col='Date', parse_dates=['Date'])
    print("Master dataset loaded successfully.")
except FileNotFoundError:
    print(f"⚠️ Error: Could not find {input_file}. Run Phase 1 first.")
    exit()

# --- 3. STATIONARITY (LOG RETURNS) ---
print("Calculating Log Returns to achieve stationarity...")
# We only apply log returns to absolute prices/indices.
columns_to_diff = ['Wheat_Close', 'Corn_Close', 'Sugar_Close', 'Brent_Crude_Close', 'DXY_Close', 'EGP_USD_Close']

for col in columns_to_diff:
    # Applying the mathematical formula: R_t = ln(P_t / P_{t-1})
    df[f'{col}_LogReturn'] = np.log(df[col] / df[col].shift(1))

# Egypt_Inflation_YoY and CBE_Interest_Rate are already rates (not prices), so we apply simple differencing instead
df['Egypt_Inflation_YoY_Diff'] = df['Egypt_Inflation_YoY'].diff()
df['CBE_Interest_Rate_Diff'] = df['CBE_Interest_Rate'].diff()

# --- 4. CLEANUP & SAVE ---
print("Executing final data cleanup...")
# The log return / diff calculation creates a NaN for the very first row.
# We drop this initial empty row to keep the dataset clean.
df.dropna(inplace=True)

df.to_csv(output_file)

print(f"\n--- DIFFERENCING PIPELINE (NO EXTRAS) COMPLETE ---")
print(f"Dataset created with {len(df)} rows and {len(df.columns)} columns.")
print(f"No lagged, rolling, or cross-features were added.")
print(f"File saved to: {output_file}")
