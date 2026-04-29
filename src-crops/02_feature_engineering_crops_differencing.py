# =============================================================================
# PIPELINE IDENTITY
#   Stationarity Method : DIFFERENCING  (Log Returns: R_t = ln(P_t / P_{t-1}))
#   Extra Features      : YES (Lag T-1/T-3/T-7, 14-Day Rolling Avg, Cross-Feature)
#   Output dataset      : 01a_engineered_differencing_crops_dataset.csv
#   Used by             : 03_model_training_engineered.py  (XGBoost Architecture A)
#                         04a / 04b evaluation scripts
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
output_file = os.path.join(FINAL_DIR, '01a_engineered_differencing_crops_dataset.csv')

print("Initiating Feature Engineering Protocol (Differencing)...")

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

# --- 4. TIME-SERIES MEMORY (LAGGED FEATURES) ---
print("Generating T-1, T-3, and T-7 Lagged Features...")
# XGBoost doesn't understand the flow of time. We must explicitly give it "yesterday",
# "3 days ago", and "1 week ago" as completely separate columns.
for col in columns_to_diff:
    df[f'{col}_Lag1'] = df[col].shift(1)
    df[f'{col}_Lag3'] = df[col].shift(3)
    df[f'{col}_Lag7'] = df[col].shift(7)

# --- 5. MARKET MOMENTUM (ROLLING AVERAGES) ---
print("Calculating 14-Day Rolling Averages...")

# This smooths out the daily noise and gives the model the broader micro-trend
for col in columns_to_diff:
    df[f'{col}_Roll14'] = df[col].rolling(window=14).mean()

# --- 6. DOMAIN-SPECIFIC CROSS-FEATURES ---
print("Engineering Economic Cross-Features...")
# The Wheat/Corn ratio is a classic agricultural market spread indicator.
# A widening spread can signal shifts in crop substitution and feed demand dynamics.
df['Wheat_Corn_Ratio'] = df['Wheat_Close'] / df['Corn_Close']

# --- 7. CLEANUP & SAVE ---
print("Executing final data cleanup...")
# Shifting (Lag7) and Rolling (Roll14) will inherently create NaNs for the first 14 days
# of our dataset because you cannot calculate a 14-day average on Day 1.
# We drop these initial empty rows to prevent XGBoost from crashing.
df.dropna(inplace=True)

df.to_csv(output_file)

print(f"\n--- FEATURE ENGINEERING COMPLETE ---")
print(f"Engineered Dataset created with {len(df)} rows and {len(df.columns)} features.")
print(f"Ready for Machine Learning ingestion. File saved to: {output_file}")