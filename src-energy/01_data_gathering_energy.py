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

# Save each yfinance series individually
yf_series_map = {
    '01_yf_brent_crude.csv':   'Brent_Crude_Close',
    '02_yf_natural_gas.csv':   'Natural_Gas_Close',
    '03_yf_dxy.csv':           'DXY_Close',
    '04_yf_vix.csv':           'VIX_Close',
    '05_yf_sp500.csv':         'SP500_Close',
}
for fname, col in yf_series_map.items():
    df_single = yf_data[[col]].copy()
    df_single.index.name = 'Date'
    df_single.to_csv(os.path.join(RAW_DIR, fname))
    print(f"  Saved: {fname}")

# --- 3. FETCH GLOBAL MACROECONOMICS (FRED API) ---
print("Pulling US 10-Year Treasury Yields from FRED...")
treasury_10yr = fred.get_series('DGS10', observation_start=START_DATE, observation_end=END_DATE)
treasury_df = pd.DataFrame(treasury_10yr, columns=['US_10Yr_Yield'])
treasury_df.index.name = 'Date'
treasury_df.to_csv(os.path.join(RAW_DIR, '06_fred_treasury_10yr.csv'))
print("  Saved: 06_fred_treasury_10yr.csv")

# --- 4. LOAD LOCAL EGYPTIAN ECONOMIC DATA (from pre-downloaded CSVs) ---
print("\nLoading Local Egyptian Economic Variables...")

egp_usd_path = os.path.join(RAW_DIR, '01_USD_EGP_Historical_Data.csv')
egypt_inflation_path = os.path.join(RAW_DIR, '01_Egypt_Inflation_CBE.csv')
egypt_interest_path = os.path.join(RAW_DIR, '01_Overnight_deposit_rate.csv')

# A. EGP/USD Exchange Rate
egp_usd_df = pd.read_csv(egp_usd_path, usecols=['Date', 'Price'])
egp_usd_df['Price'] = egp_usd_df['Price'].astype(str).str.replace(',', '').astype(float)
egp_usd_df['Date'] = pd.to_datetime(egp_usd_df['Date'], format='%m/%d/%Y')
egp_usd_df.rename(columns={'Price': 'EGP_USD_Close'}, inplace=True)
egp_usd_df.set_index('Date', inplace=True)
print("  Loaded: EGP/USD Exchange Rate")

# B. Egypt Inflation (Year-over-Year Headline)
inflation_df = pd.read_csv(egypt_inflation_path, usecols=['Date', 'Headline (y/y)'])
inflation_df['Headline (y/y)'] = inflation_df['Headline (y/y)'].astype(str).str.replace('%', '').astype(float)
inflation_df['Date'] = inflation_df['Date'].astype(str).str.replace('Sept-', 'Sep-', regex=False)
inflation_df['Date'] = pd.to_datetime(inflation_df['Date'], format='%b-%y')
inflation_df.rename(columns={'Headline (y/y)': 'Egypt_Inflation_YoY'}, inplace=True)
inflation_df.set_index('Date', inplace=True)
print("  Loaded: Egypt Inflation YoY")

# C. CBE Overnight Deposit Rate (Interest Rate)
interest_df = pd.read_csv(egypt_interest_path, usecols=['Date', 'Overnight Deposit Rate'])
interest_df['Overnight Deposit Rate'] = interest_df['Overnight Deposit Rate'].astype(str).str.replace('%', '').astype(float)
interest_df['Date'] = interest_df['Date'].astype(str).str.replace('-Sept-', '-Sep-', regex=False)
interest_df['Date'] = pd.to_datetime(interest_df['Date'], format='%d-%b-%y')
interest_df.rename(columns={'Overnight Deposit Rate': 'CBE_Interest_Rate'}, inplace=True)
interest_df.set_index('Date', inplace=True)
print("  Loaded: CBE Interest Rate")

# --- 5. THE MASTER MERGE & FORWARD-FILL ---
print("\nInitiating Master Merge...")
master_timeline = pd.date_range(start=START_DATE, end=END_DATE, freq='D', name='Date')
master_df = pd.DataFrame(index=master_timeline)

master_df = master_df.join(yf_data)
master_df = master_df.join(treasury_df)
master_df = master_df.join(egp_usd_df)
master_df = master_df.join(inflation_df)
master_df = master_df.join(interest_df)

# Forward fill pushes weekend stock prices and monthly macro rates forward
master_df.ffill(inplace=True)

# Drop any early NA values before all datasets properly align
master_df.dropna(inplace=True)

final_save_path = os.path.join(PROCESSED_DIR, '01_master_energy_dataset.csv')
master_df.to_csv(final_save_path)

print(f"\n--- ADVANCED PIPELINE COMPLETE ---")
print(f"Energy Dataset created: {master_df.index.min().date()} to {master_df.index.max().date()}")
print(f"Rows: {len(master_df)} | Features: {len(master_df.columns)}")
print(f"Columns: {list(master_df.columns)}")