import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- 1. CONFIGURATION & DIRECTORY SETUP ---
PROCESSED_DIR = '../data-extra-variables-local/02_processed/'
FINAL_DIR = '../data-extra-variables-local/03_final/'

# Ensure the final output directory exists
os.makedirs(FINAL_DIR, exist_ok=True)

input_file = os.path.join(PROCESSED_DIR, '01_master_metals_dataset.csv')
output_file = os.path.join(FINAL_DIR, '01b_detrending_metals_dataset.csv')

print("Initiating Detrending Pipeline (No Extra Columns)...")

# --- 2. LOAD THE MASTER DATASET ---
try:
    df = pd.read_csv(input_file, index_col='Date', parse_dates=['Date'])
    print("Master dataset loaded successfully.")
except FileNotFoundError:
    print(f"⚠️ Error: Could not find {input_file}. Run Phase 1 first.")
    exit()

# --- 3. STATIONARITY (LINEAR DETRENDING) ---
print("Calculating Linear Trends and Extracting Residuals...")
columns_to_detrend = ['Gold_Close', 'Silver_Close', 'DXY_Close', 'SP500_Close', 'VIX_Close', 'EGP_USD_Close', 'Local_Gold_24k_EGP']

# Create an X-axis representing the flow of time (0, 1, 2, 3... to the end of the dataset)
# We must reshape it to a 2D array because scikit-learn requires it.
time_index = np.arange(len(df)).reshape(-1, 1)

for col in columns_to_detrend:
    # Instantiate the standard linear equation: y = mx + b
    lr_model = LinearRegression()
    
    # Grab the actual prices (Y-axis)
    y_actual = df[col].values.reshape(-1, 1)
    
    # Fit the mathematical line of best fit across the full available history
    lr_model.fit(time_index, y_actual)
    
    # Calculate exactly what the line expected the price to be on every single day
    trend_line = lr_model.predict(time_index)
    
    # Calculate the Residual (Actual Price minus the Expected Trend Price)
    df[f'{col}_Trend'] = trend_line
    df[f'{col}_Residual'] = y_actual - trend_line

# --- 4. CLEANUP & SAVE ---
print("Executing final data cleanup...")
# Detrending via linear regression does not produce NaNs,
# but we still call dropna for safety in case the source data has any.
df.dropna(inplace=True)

df.to_csv(output_file)

print(f"\n--- DETRENDING PIPELINE (NO EXTRAS) COMPLETE ---")
print(f"Dataset created with {len(df)} rows and {len(df.columns)} columns.")
print(f"No lagged, rolling, or cross-features were added.")
print(f"File saved to: {output_file}")

