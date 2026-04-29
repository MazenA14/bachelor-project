import os
import yfinance as yf
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION & DIRECTORY SETUP ---
START_DATE = '2014-01-01'
END_DATE = '2026-03-15' # Adjust to current date

# Routing to a brand new Crops specific folder
RAW_DIR = '../data-crops/01_raw/'
PROCESSED_DIR = '../data-crops/02_processed/'
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("Initiating Data Gathering Protocol (Agriculture/Crops Group)...")

# --- 2. FETCH GLOBAL TARGETS & EXOGENOUS FEATURES (yfinance) ---
print("Pulling Yahoo Finance data (Wheat, Corn, Sugar, Brent Crude, DXY)...")

# ZW=F (Wheat), ZC=F (Corn), SB=F (Sugar), BZ=F (Brent Crude), DX-Y.NYB (DXY)
tickers = {
    'Wheat': 'ZW=F', 
    'Corn': 'ZC=F', 
    'Sugar': 'SB=F', 
    'Brent_Crude': 'BZ=F', 
    'DXY': 'DX-Y.NYB'
}

yf_data = yf.download(list(tickers.values()), start=START_DATE, end=END_DATE)['Close']
yf_data.rename(columns={
    'ZW=F': 'Wheat_Close', 
    'ZC=F': 'Corn_Close', 
    'SB=F': 'Sugar_Close',
    'BZ=F': 'Brent_Crude_Close', 
    'DX-Y.NYB': 'DXY_Close'
}, inplace=True)

# Save each yfinance series individually
yf_series_map = {
    '01_yf_wheat.csv':       'Wheat_Close',
    '02_yf_corn.csv':        'Corn_Close',
    '03_yf_sugar.csv':       'Sugar_Close',
    '04_yf_brent_crude.csv': 'Brent_Crude_Close',
    '05_yf_dxy.csv':         'DXY_Close'
}

for fname, col in yf_series_map.items():
    df_single = yf_data[[col]].copy()
    df_single.index.name = 'Date'
    df_single.to_csv(os.path.join(RAW_DIR, fname))
    print(f"  Saved: {fname}")

# --- 3. LOAD LOCAL EGYPTIAN ECONOMIC DATA ---
print("\nLoading Local Egyptian Economic Variables...")

# Pointing backward to your existing local CSVs so you don't have to download them again
LOCAL_CSV_DIR = '../data-extra-variables-local/01_raw/'
egp_usd_path = os.path.join(LOCAL_CSV_DIR, '01_USD_EGP_Historical_Data.csv')
egypt_inflation_path = os.path.join(LOCAL_CSV_DIR, '01_Egypt_Inflation_CBE.csv')
egypt_interest_path = os.path.join(LOCAL_CSV_DIR, '01_Overnight_deposit_rate.csv')

try:
    # A. EGP/USD Exchange Rate
    egp_usd_df = pd.read_csv(egp_usd_path, usecols=['Date', 'Price'])
    egp_usd_df['Price'] = egp_usd_df['Price'].astype(str).str.replace(',', '').astype(float)
    egp_usd_df['Date'] = pd.to_datetime(egp_usd_df['Date'], format='%m/%d/%Y')
    egp_usd_df.rename(columns={'Price': 'EGP_USD_Close'}, inplace=True)
    egp_usd_df.set_index('Date', inplace=True)
    print("  Loaded: EGP/USD Exchange Rate")

    # B. Egypt Inflation 
    inflation_df = pd.read_csv(egypt_inflation_path, usecols=['Date', 'Headline (y/y)'])
    inflation_df['Headline (y/y)'] = inflation_df['Headline (y/y)'].astype(str).str.replace('%', '').astype(float)
    inflation_df['Date'] = inflation_df['Date'].astype(str).str.replace('Sept-', 'Sep-', regex=False)
    inflation_df['Date'] = pd.to_datetime(inflation_df['Date'], format='%b-%y')
    inflation_df.rename(columns={'Headline (y/y)': 'Egypt_Inflation_YoY'}, inplace=True)
    inflation_df.set_index('Date', inplace=True)
    print("  Loaded: Egypt Inflation YoY")

    # C. CBE Interest Rate
    interest_df = pd.read_csv(egypt_interest_path, usecols=['Date', 'Overnight Deposit Rate'])
    interest_df['Overnight Deposit Rate'] = interest_df['Overnight Deposit Rate'].astype(str).str.replace('%', '').astype(float)
    interest_df['Date'] = interest_df['Date'].astype(str).str.replace('-Sept-', '-Sep-', regex=False)
    interest_df['Date'] = pd.to_datetime(interest_df['Date'], format='%d-%b-%y')
    interest_df.rename(columns={'Overnight Deposit Rate': 'CBE_Interest_Rate'}, inplace=True)
    interest_df.set_index('Date', inplace=True)
    print("  Loaded: CBE Interest Rate")

except FileNotFoundError as e:
    print(f"\n⚠️ Error: Could not find Egyptian CSV files. Details: {e}")
    print("Ensure the local CSVs are present in '../data-extra-variables-local/01_raw/'")
    egp_usd_df, inflation_df, interest_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# --- 4. THE MASTER MERGE & FORWARD-FILL ---
print("\nInitiating Master Merge...")
master_timeline = pd.date_range(start=START_DATE, end=END_DATE, freq='D', name='Date')
master_df = pd.DataFrame(index=master_timeline)

master_df = master_df.join(yf_data)
master_df = master_df.join(egp_usd_df)
master_df = master_df.join(inflation_df)
master_df = master_df.join(interest_df)

master_df.ffill(inplace=True)
master_df.dropna(inplace=True)

final_save_path = os.path.join(PROCESSED_DIR, '01_master_crops_dataset.csv')
master_df.to_csv(final_save_path)

print(f"\n--- CROPS PIPELINE COMPLETE ---")
print(f"Dataset created: {master_df.index.min().date()} to {master_df.index.max().date()}")
print(f"Rows: {len(master_df)} | Features: {len(master_df.columns)}")