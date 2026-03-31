import os
import yfinance as yf
import pandas as pd

# --- 1. CONFIGURATION & DIRECTORY SETUP ---
START_DATE = '2014-01-01'
END_DATE = '2026-03-15' # Adjust to current date

# Routing everything to the local dataset folders
RAW_DIR = '../data-extra-variables-local/01_raw/'
PROCESSED_DIR = '../data-extra-variables-local/02_processed/'
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("Initiating ADVANCED Data Gathering Protocol (Local EGP Target)...")

# --- 2. FETCH GLOBAL TARGETS & NEW VARIABLES (yfinance) ---
print("Pulling Yahoo Finance data (Gold, Silver, DXY, VIX, S&P500)...")
# Note: Global Gold is now a FEATURE, no longer the Target!
tickers = {'Gold': 'GC=F', 'Silver': 'SI=F', 'DXY': 'DX-Y.NYB', 'VIX': '^VIX', 'SP500': '^GSPC'}

yf_data = yf.download(list(tickers.values()), start=START_DATE, end=END_DATE)['Close']
yf_data.rename(columns={
    'GC=F': 'Gold_Close', 
    'SI=F': 'Silver_Close', 
    'DX-Y.NYB': 'DXY_Close',
    '^VIX': 'VIX_Close',
    '^GSPC': 'SP500_Close'
}, inplace=True)

yf_data.to_csv(os.path.join(RAW_DIR, '01_yf_global_data.csv'))

# --- 3. LOAD GLOBAL MACROECONOMICS (Local CSV) ---
print("Loading US 10-Year Treasury Yields from local CSV...")
treasury_path = os.path.join(RAW_DIR, '01_fred_treasury_10yr.csv')
treasury_df = pd.read_csv(treasury_path, usecols=['Date', 'US_10Yr_Yield'])
treasury_df['Date'] = pd.to_datetime(treasury_df['Date'], format='%Y-%m-%d')
treasury_df['US_10Yr_Yield'] = pd.to_numeric(treasury_df['US_10Yr_Yield'], errors='coerce')
treasury_df.set_index('Date', inplace=True)

# --- 4. LOAD & CLEAN EGYPTIAN MACROECONOMICS (Local CSVs) ---
print("Loading and cleaning local Egyptian CSVs...")
egp_usd_path = os.path.join(RAW_DIR, '01_USD_EGP_Historical_Data.csv')
egypt_inflation_path = os.path.join(RAW_DIR, '01_Egypt_Inflation_CBE.csv')
egypt_interest_path = os.path.join(RAW_DIR, '01_Overnight_deposit_rate.csv')
local_gold_path = os.path.join(RAW_DIR, '01_Offical_Egypt_Gold_Price.csv') # Local 24k gold series

try:
    # A. EGP/USD 
    egp_usd_df = pd.read_csv(egp_usd_path, usecols=['Date', 'Price'])
    egp_usd_df['Price'] = egp_usd_df['Price'].astype(str).str.replace(',', '').astype(float)
    egp_usd_df['Date'] = pd.to_datetime(egp_usd_df['Date'], format='%m/%d/%Y')
    egp_usd_df.rename(columns={'Price': 'EGP_USD_Close'}, inplace=True)
    egp_usd_df.set_index('Date', inplace=True)

    # B. Inflation 
    inflation_df = pd.read_csv(egypt_inflation_path, usecols=['Date', 'Headline (y/y)'])
    inflation_df['Headline (y/y)'] = inflation_df['Headline (y/y)'].astype(str).str.replace('%', '').astype(float)
    inflation_df['Date'] = inflation_df['Date'].astype(str).str.replace('Sept-', 'Sep-', regex=False)
    inflation_df['Date'] = pd.to_datetime(inflation_df['Date'], format='%b-%y')
    inflation_df.rename(columns={'Headline (y/y)': 'Egypt_Inflation_YoY'}, inplace=True)
    inflation_df.set_index('Date', inplace=True)

    # C. CBE Interest Rate 
    interest_df = pd.read_csv(egypt_interest_path, usecols=['Date', 'Overnight Deposit Rate'])
    interest_df['Overnight Deposit Rate'] = interest_df['Overnight Deposit Rate'].astype(str).str.replace('%', '').astype(float)
    interest_df['Date'] = interest_df['Date'].astype(str).str.replace('-Sept-', '-Sep-', regex=False)
    interest_df['Date'] = pd.to_datetime(interest_df['Date'], format='%d-%b-%y')
    interest_df.rename(columns={'Overnight Deposit Rate': 'CBE_Interest_Rate'}, inplace=True)
    interest_df.set_index('Date', inplace=True)

    # D. NEW: Official Egypt gold price dataset
    local_gold_df = pd.read_csv(local_gold_path, usecols=['Date', 'Price'])
    local_gold_df['Price'] = local_gold_df['Price'].astype(str).str.replace(',', '').astype(float)
    local_gold_df['Date'] = pd.to_datetime(local_gold_df['Date'], format='%m/%d/%Y')
    local_gold_df.rename(columns={'Price': 'Local_Gold_24k_EGP'}, inplace=True)
    local_gold_df.set_index('Date', inplace=True)   

except FileNotFoundError as e:
    print(f"⚠️ Error: Could not find file. Details: {e}")

# --- 5. THE MASTER MERGE & FORWARD-FILL ---
print("\nInitiating Master Merge...")
master_timeline = pd.date_range(start=START_DATE, end=END_DATE, freq='D', name='Date')
master_df = pd.DataFrame(index=master_timeline)

master_df = master_df.join(yf_data)
master_df = master_df.join(treasury_df)
master_df = master_df.join(egp_usd_df)
master_df = master_df.join(inflation_df)
master_df = master_df.join(interest_df)
master_df = master_df.join(local_gold_df) # Joining the local 24k target series

# Forward fill pushes weekend stock prices and monthly macro rates forward
master_df.ffill(inplace=True)

# Keep only rows where every joined series is available after forward-fill.
master_df.dropna(inplace=True)

final_save_path = os.path.join(PROCESSED_DIR, '01_master_metals_dataset.csv')
master_df.to_csv(final_save_path)

print(f"\n--- ADVANCED PIPELINE COMPLETE ---")
print(f"Master Dataset created: {master_df.index.min().date()} to {master_df.index.max().date()}")
print(f"Rows: {len(master_df)} | Features: {len(master_df.columns)}")