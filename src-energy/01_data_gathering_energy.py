import os
import yfinance as yf
from fredapi import Fred
import pandas as pd
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION & DIRECTORY SETUP ---
load_dotenv()
FRED_API_KEY = os.getenv('FRED_API_KEY')
fred = Fred(api_key=FRED_API_KEY)

START_DATE = '2014-01-01'
END_DATE = '2026-03-15' # Adjust to current date

# Routing to the Energy specific folder
RAW_DIR = '../data-energy/01_raw/'
PROCESSED_DIR = '../data-energy/02_processed/'
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("Initiating ADVANCED Data Gathering Protocol (Global Energy Target)...")

# --- 2. FETCH GLOBAL TARGET & ENERGY FEATURES (yfinance) ---
print("Pulling Yahoo Finance data (Brent Crude, NatGas, DXY, VIX, S&P 500)...")

# BZ=F (Brent Crude), NG=F (Natural Gas), ^GSPC (S&P 500)
tickers = {'Brent_Crude': 'BZ=F', 'Natural_Gas': 'NG=F', 'DXY': 'DX-Y.NYB', 'VIX': '^VIX', 'SP500': '^GSPC'}

yf_data = yf.download(list(tickers.values()), start=START_DATE, end=END_DATE)['Close']
yf_data.rename(columns={
    'BZ=F': 'Brent_Crude_Close', 
    'NG=F': 'Natural_Gas_Close', 
    'DX-Y.NYB': 'DXY_Close',
    '^VIX': 'VIX_Close',
    '^GSPC': 'SP500_Close'
}, inplace=True)

yf_data.to_csv(os.path.join(RAW_DIR, '01_yf_global_energy_data.csv'))

# --- 3. FETCH GLOBAL MACROECONOMICS (FRED API) ---
print("Pulling US 10-Year Treasury Yields from FRED...")
treasury_10yr = fred.get_series('DGS10', observation_start=START_DATE, observation_end=END_DATE)
treasury_df = pd.DataFrame(treasury_10yr, columns=['US_10Yr_Yield'])
treasury_df.index.name = 'Date'



# --- 5. THE MASTER MERGE & FORWARD-FILL ---
print("\nInitiating Master Merge...")
master_timeline = pd.date_range(start=START_DATE, end=END_DATE, freq='D', name='Date')
master_df = pd.DataFrame(index=master_timeline)

master_df = master_df.join(yf_data)
master_df = master_df.join(treasury_df)

# Forward fill pushes weekend stock prices and monthly macro rates forward
master_df.ffill(inplace=True)

# Drop any early NA values before all datasets properly align
master_df.dropna(inplace=True)

final_save_path = os.path.join(PROCESSED_DIR, '01_master_energy_dataset.csv')
master_df.to_csv(final_save_path)

print(f"\n--- ADVANCED PIPELINE COMPLETE ---")
print(f"Energy Dataset created: {master_df.index.min().date()} to {master_df.index.max().date()}")
print(f"Rows: {len(master_df)} | Features: {len(master_df.columns)}")