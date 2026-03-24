import os
import yfinance as yf
from fredapi import Fred
import pandas as pd
from dotenv import load_dotenv

# --- 1. CONFIGURATION & DIRECTORY SETUP ---
load_dotenv()
FRED_API_KEY = os.getenv('FRED_API_KEY')
fred = Fred(api_key=FRED_API_KEY)

START_DATE = '2014-01-01'
END_DATE = '2026-03-15' # Adjust to current date

RAW_DIR = '../data-extra-variables/01_raw/'
PROCESSED_DIR = '../data-extra-variables/02_processed/'
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("Initiating ADVANCED Data Gathering Protocol...")

# --- 2. FETCH GLOBAL TARGETS & NEW VARIABLES (yfinance) ---
print("Pulling Yahoo Finance data (Gold, Silver, DXY, VIX, S&P500)...")
# ADDED: VIX (^VIX) and S&P 500 (^GSPC)
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


# --- 3. FETCH GLOBAL MACROECONOMICS (FRED API) ---
print("Pulling US 10-Year Treasury Yields from FRED...")
treasury_10yr = fred.get_series('DGS10', observation_start=START_DATE, observation_end=END_DATE)
treasury_df = pd.DataFrame(treasury_10yr, columns=['US_10Yr_Yield'])
treasury_df.index.name = 'Date'


# --- 4. LOAD & CLEAN EGYPTIAN MACROECONOMICS (Local CSVs) ---
print("Loading and cleaning local Egyptian CSVs...")
egp_usd_path = os.path.join(RAW_DIR, '01_USD_EGP_Historical_Data.csv')
egypt_inflation_path = os.path.join(RAW_DIR, '01_Egypt_Inflation_CBE.csv')
# ADDED: New Interest Rate Path
egypt_interest_path = os.path.join(RAW_DIR, '01_Overnight_deposit_rate.csv')

try:
    # A. EGP/USD  — format: MM/DD/YYYY  (e.g. 03/11/2026)
    egp_usd_df = pd.read_csv(egp_usd_path, usecols=['Date', 'Price'])
    egp_usd_df['Price'] = egp_usd_df['Price'].astype(str).str.replace(',', '').astype(float)
    egp_usd_df['Date'] = pd.to_datetime(egp_usd_df['Date'], format='%m/%d/%Y')
    egp_usd_df.rename(columns={'Price': 'EGP_USD_Close'}, inplace=True)
    egp_usd_df.set_index('Date', inplace=True)

    # B. Inflation  — format: Mon-YY  (e.g. Feb-26, Sept-25)
    #    'Sept' is non-standard; normalise to 'Sep' before parsing
    inflation_df = pd.read_csv(egypt_inflation_path, usecols=['Date', 'Headline (y/y)'])
    inflation_df['Headline (y/y)'] = inflation_df['Headline (y/y)'].astype(str).str.replace('%', '').astype(float)
    inflation_df['Date'] = (
        inflation_df['Date'].astype(str)
        .str.replace('Sept-', 'Sep-', regex=False)
    )
    inflation_df['Date'] = pd.to_datetime(inflation_df['Date'], format='%b-%y')
    inflation_df.rename(columns={'Headline (y/y)': 'Egypt_Inflation_YoY'}, inplace=True)
    inflation_df.set_index('Date', inplace=True)

    # C. CBE Interest Rate  — format: DD-Mon-YY  (e.g. 15-Feb-26, 27-Sept-20)
    #    'Sept' is non-standard; normalise to 'Sep' before parsing
    interest_df = pd.read_csv(egypt_interest_path, usecols=['Date', 'Overnight Deposit Rate'])
    interest_df['Overnight Deposit Rate'] = interest_df['Overnight Deposit Rate'].astype(str).str.replace('%', '').astype(float)
    interest_df['Date'] = (
        interest_df['Date'].astype(str)
        .str.replace('-Sept-', '-Sep-', regex=False)
    )
    interest_df['Date'] = pd.to_datetime(interest_df['Date'], format='%d-%b-%y')
    interest_df.rename(columns={'Overnight Deposit Rate': 'CBE_Interest_Rate'}, inplace=True)
    interest_df.set_index('Date', inplace=True)

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
master_df = master_df.join(interest_df) # Joining the new variable

# Forward fill pushes weekend stock prices forward, AND pushes monthly inflation/interest rates forward
master_df.ffill(inplace=True)
master_df.dropna(inplace=True)

final_save_path = os.path.join(PROCESSED_DIR, '01_master_metals_dataset.csv')
master_df.to_csv(final_save_path)

print(f"\n--- ADVANCED PIPELINE COMPLETE ---")
print(f"Master Dataset created with {len(master_df)} rows and {len(master_df.columns)} features.")