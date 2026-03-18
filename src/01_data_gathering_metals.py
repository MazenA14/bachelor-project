import os
import yfinance as yf
from fredapi import Fred
import pandas as pd
from dotenv import load_dotenv

# --- 1. CONFIGURATION & DIRECTORY SETUP ---
load_dotenv()
FRED_API_KEY = os.getenv('FRED_API_KEY')
fred = Fred(api_key=FRED_API_KEY)

# Define strict thesis timeframe
START_DATE = '2014-01-01'
END_DATE = '2026-03-15'

# Define folder paths
RAW_DIR = '../data/01_raw/'
PROCESSED_DIR = '../data/02_processed/'

# Create directories if they don't exist
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("Initiating data gathering protocol...")

# --- 2. FETCH & SAVE GLOBAL TARGETS (yfinance) ---
print("Pulling Yahoo Finance data (Gold, Silver, DXY)...")
tickers = {'Gold': 'GC=F', 'Silver': 'SI=F', 'DXY': 'DX=F'}

# Download 'Close' prices and rename columns safely
yf_data = yf.download(list(tickers.values()), start=START_DATE, end=END_DATE)['Close']
yf_data.rename(columns={'GC=F': 'Gold_Close', 'SI=F': 'Silver_Close', 'DX=F': 'DXY_Close'}, inplace=True)

# THE FIX: Strip timezones from Yahoo Finance so it matches your other data perfectly
yf_data.index = yf_data.index.tz_localize(None)

# Save the raw data immediately to your local folder
yf_data.to_csv(os.path.join(RAW_DIR, '01_yf_global_data.csv'))
print(f"Yahoo Finance data secured and saved to {RAW_DIR}")

# --- 3. FETCH & SAVE GLOBAL MACROECONOMICS (FRED API) ---
print("Pulling US 10-Year Treasury Yields from FRED...")
treasury_10yr = fred.get_series('DGS10', observation_start=START_DATE, observation_end=END_DATE)
treasury_df = pd.DataFrame(treasury_10yr, columns=['US_10Yr_Yield'])
treasury_df.index.name = 'Date'

# Save the raw FRED data
treasury_df.to_csv(os.path.join(RAW_DIR, '01_fred_treasury_10yr.csv'))
print(f"FRED data secured and saved to {RAW_DIR}")

# --- 4. LOAD & CLEAN EGYPTIAN MACROECONOMICS (Local CSVs) ---
print("Loading and cleaning local Egyptian CSVs...")
egp_usd_path = os.path.join(RAW_DIR, '01_USD_EGP_Historical_Data.csv')
egypt_inflation_path = os.path.join(RAW_DIR, '01_Egypt_Inflation_CBE.csv')

try:
    # A. Clean Investing.com EGP/USD Data
    # Only grab Date and Price based on your CSV structure
    egp_usd_df = pd.read_csv(egp_usd_path, usecols=['Date', 'Price'], parse_dates=['Date'])
    
    # Clean the "Price" column (remove commas so python sees it as a float)
    egp_usd_df['Price'] = egp_usd_df['Price'].astype(str).str.replace(',', '').astype(float)
    egp_usd_df.rename(columns={'Price': 'EGP_USD_Close'}, inplace=True)
    egp_usd_df.set_index('Date', inplace=True)

    # B. Clean Central Bank Inflation Data
    inflation_df = pd.read_csv(egypt_inflation_path, usecols=['Date', 'Headline (y/y)'])
    
    # THE FIX: Replace the non-standard "Sept" with standard "Sep"
    inflation_df['Date'] = inflation_df['Date'].str.replace('Sept', 'Sep', regex=False)
    
    # Now convert to dates
    inflation_df['Date'] = pd.to_datetime(inflation_df['Date'], format='%b-%y')
    
    # Clean the "Headline (y/y)" column
    inflation_df['Headline (y/y)'] = inflation_df['Headline (y/y)'].astype(str).str.replace('%', '', regex=False).astype(float)
    inflation_df.rename(columns={'Headline (y/y)': 'Egypt_Inflation_YoY'}, inplace=True)
    inflation_df.set_index('Date', inplace=True)
    
    print("Egyptian local data cleaned successfully!")

except FileNotFoundError:
    print("⚠️ Error: Could not find the local CSVs. Double-check your file names.")
except ValueError as e:
    print(f"⚠️ Column parsing error: Please verify the CSV column names match the script exactly. Details: {e}")

# --- 5. THE MASTER MERGE & FORWARD-FILL ---
print("\nInitiating Master Merge and Date Alignment...")

# 1. Create a perfect, unbroken timeline
master_timeline = pd.date_range(start=START_DATE, end=END_DATE, freq='D', name='Date')
master_df = pd.DataFrame(index=master_timeline)

# 2. Join all DataFrames
master_df = master_df.join(yf_data)
master_df = master_df.join(treasury_df)
master_df = master_df.join(egp_usd_df)
master_df = master_df.join(inflation_df)

# 3. Handle the "Weekend Trap" and "Monthly Mismatch" via Forward Fill
master_df.ffill(inplace=True)

# 4. THE DIAGNOSTIC FIX: Let's see exactly what is missing before we drop anything
print("\n--- DIAGNOSTICS: Blank rows per column ---")
print(master_df.isna().sum())

# 5. THE SAFER DROP: Instead of dropping a row if ANY data is missing,
# let's only drop rows at the very beginning where global markets hadn't opened yet.
master_df.dropna(subset=['Gold_Close'], inplace=True)

# Save the perfectly aligned dataset
final_save_path = os.path.join(PROCESSED_DIR, '01_master_metals_dataset.csv')
master_df.to_csv(final_save_path)

print(f"\n--- PIPELINE COMPLETE ---")
print(f"Master Dataset created with {len(master_df)} daily rows.")
print(f"File saved to: {final_save_path}")