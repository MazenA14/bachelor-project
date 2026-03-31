import os
import pandas as pd
import numpy as np

# --- 1. CONFIGURATION & DIRECTORY SETUP ---
PROCESSED_DIR = '../data-extra-variables-local/02_processed/'
FINAL_DIR = '../data-extra-variables-local/03_final/'

# Ensure the final output directory exists
os.makedirs(FINAL_DIR, exist_ok=True)

input_file = os.path.join(PROCESSED_DIR, '01_master_metals_dataset.csv')
output_file = os.path.join(FINAL_DIR, '01a_differencing_metals_dataset.csv')

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
columns_to_diff = ['Gold_Close', 'Silver_Close', 'DXY_Close', 'SP500_Close', 'VIX_Close', 'EGP_USD_Close', 'Local_Gold_24k_EGP']

for col in columns_to_diff:
    # Applying the mathematical formula: R_t = ln(P_t / P_{t-1})
    df[f'{col}_LogReturn'] = np.log(df[col] / df[col].shift(1))

# --- 4. CLEANUP & SAVE ---
print("Executing final data cleanup...")
# The log return calculation creates a NaN for the very first row.
# We drop this initial empty row to keep the dataset clean.
df.dropna(inplace=True)

df.to_csv(output_file)

print(f"\n--- DIFFERENCING PIPELINE (NO EXTRAS) COMPLETE ---")
print(f"Dataset created with {len(df)} rows and {len(df.columns)} columns.")
print(f"No lagged, rolling, or cross-features were added.")
print(f"File saved to: {output_file}")

