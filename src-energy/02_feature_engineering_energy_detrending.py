# =============================================================================
# PIPELINE IDENTITY
#   Stationarity Method : DETRENDING  (Linear Regression residuals: y - trend)
#   Extra Features      : YES (Lag T-1/T-3/T-7, 14-Day Rolling Avg, Cross-Feature)
#   Output dataset      : 01b_engineered_detrending_energy_dataset.csv
#   Used by             : 03_model_training_engineered.py  (XGBoost Architecture B)
#                         04a / 04b evaluation scripts
# =============================================================================
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- 1. CONFIGURATION & DIRECTORY SETUP ---
PROCESSED_DIR = '../data-energy/02_processed/'
FINAL_DIR = '../data-energy/03_final/'

# Ensure the final output directory exists
os.makedirs(FINAL_DIR, exist_ok=True)

input_file = os.path.join(PROCESSED_DIR, '01_master_energy_dataset.csv')
output_file = os.path.join(FINAL_DIR, '01b_engineered_detrending_energy_dataset.csv')

print("Initiating Architecture B Feature Engineering (Detrending)...")

# --- 2. LOAD THE MASTER DATASET ---
try:
    df = pd.read_csv(input_file, index_col='Date', parse_dates=['Date'])
    print("Master dataset loaded successfully.")
except FileNotFoundError:
    print(f"⚠️ Error: Could not find {input_file}. Run Phase 1 first.")
    exit()

# --- 3. STATIONARITY (LINEAR DETRENDING) ---
print("Calculating Linear Trends and Extracting Residuals...")
# Detrend price-based columns; US_10Yr_Yield is already a rate and is left as-is
columns_to_detrend = ['Brent_Crude_Close', 'Natural_Gas_Close', 'DXY_Close', 'VIX_Close', 'SP500_Close', 'EGP_USD_Close']

# Create an X-axis representing the flow of time (0, 1, 2, 3... to the end of the dataset)
# We must reshape it to a 2D array because scikit-learn requires it.
time_index = np.arange(len(df)).reshape(-1, 1)

for col in columns_to_detrend:
    # Instantiate the standard linear equation: y = mx + b
    lr_model = LinearRegression()

    # Grab the actual prices (Y-axis)
    y_actual = df[col].values.reshape(-1, 1)

    # Fit the mathematical line of best fit across the entire dataset
    lr_model.fit(time_index, y_actual)

    # Calculate exactly what the line expected the price to be on every single day
    trend_line = lr_model.predict(time_index)

    # Calculate the Residual (Actual Price minus the Expected Trend Price)
    df[f'{col}_Trend'] = trend_line
    df[f'{col}_Residual'] = y_actual - trend_line

# --- 4. TIME-SERIES MEMORY (LAGGED FEATURES) ---
print("Generating T-1, T-3, and T-7 Lagged Features...")
# To ensure a mathematically valid A/B test, we must feed XGBoost the EXACT same
# historical memory context it received in Architecture A.
for col in columns_to_detrend:
    df[f'{col}_Lag1'] = df[col].shift(1)
    df[f'{col}_Lag3'] = df[col].shift(3)
    df[f'{col}_Lag7'] = df[col].shift(7)

# --- 5. MARKET MOMENTUM (ROLLING AVERAGES) ---
print("Calculating 14-Day Rolling Averages...")
for col in columns_to_detrend:
    df[f'{col}_Roll14'] = df[col].rolling(window=14).mean()

# --- 6. DOMAIN-SPECIFIC CROSS-FEATURES ---
print("Engineering Economic Cross-Features...")
# The Brent/Natural Gas ratio is a classic energy market spread indicator.
# A widening spread can signal shifts in fuel substitution and energy demand dynamics.
df['Brent_NatGas_Ratio'] = df['Brent_Crude_Close'] / df['Natural_Gas_Close']

# --- 7. CLEANUP & SAVE ---
print("Executing final data cleanup...")
# Drop the initial NaN rows created by lags/rolling averages
df.dropna(inplace=True)

df.to_csv(output_file)

print(f"\n--- DETRENDING PIPELINE COMPLETE ---")
print(f"Architecture B Dataset created with {len(df)} rows and {len(df.columns)} features.")
print(f"File saved to: {output_file}")