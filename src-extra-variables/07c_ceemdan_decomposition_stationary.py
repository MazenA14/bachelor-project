import os
import pandas as pd
import numpy as np
from PyEMD import CEEMDAN
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CEEMDAN-XGBoost Pipeline (STATIONARY)  --  Step 1c: Decomposition
# =============================================================================
# This variant fixes the critical extrapolation flaw found in the pure CEEMDAN
# pipeline. Instead of decomposing the non-stationary absolute Gold Price,
# we decompose the strictly stationary Gold_Close_LogReturn.
#
# By ensuring the input signal is bounded and stationary, the resulting IMFs
# are also strictly stationary. This allows the tree-based XGBoost models
# to learn the frequency dynamics without ever needing to extrapolate into
# unseen price territory.
# =============================================================================


def main():
    # --- 1. CONFIGURATION & DIRECTORY SETUP ---
    PROCESSED_DIR = '../data-extra-variables/02_processed/'
    FINAL_DIR     = '../data-extra-variables/03_final/'
    PLOTS_DIR     = '../plots-extra-variables/ceemdan/'

    os.makedirs(FINAL_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    input_file  = os.path.join(PROCESSED_DIR, '01_master_metals_dataset.csv')
    output_file = os.path.join(FINAL_DIR, '02c_ceemdan_stationary_dataset.csv')

    print("=" * 60)
    print("  CEEMDAN-XGBoost (STATIONARY) -- Phase 1c: Decomposition")
    print("=" * 60)

    # --- 2. LOAD & PREPARE DATA ---
    try:
        df_master = pd.read_csv(input_file, index_col='Date', parse_dates=['Date'])
        print(f"[OK] Master dataset loaded: {len(df_master)} rows")
    except FileNotFoundError:
        print(f"[!]  Error: {input_file} not found. Run 01_data_gathering first.")
        exit()

    print("\nCalculating Log Returns for stationarity...")
    columns_to_diff = ['Gold_Close', 'Silver_Close', 'DXY_Close', 'SP500_Close', 'VIX_Close', 'EGP_USD_Close']
    for col in columns_to_diff:
        df_master[f'{col}_LogReturn'] = np.log(df_master[col] / df_master[col].shift(1))

    # Drop the first day which is now NaN due to the Log Return calculation
    df_master.dropna(inplace=True)
    print(f"     Dropped Day 1 (NaN). Usable rows: {len(df_master)}")

    # THE CRITICAL DIFFERENCE: We decompose the Log Returns, NOT the absolute price
    gold_signal = df_master['Gold_Close_LogReturn'].values

    # --- 3. CEEMDAN DECOMPOSITION ---
    print("\nRunning CEEMDAN decomposition on Gold_Close_LogReturn signal...")
    print("  (This ensures all resulting IMFs are strictly stationary)")

    ceemdan = CEEMDAN(
        trials=100,
        epsilon=0.005,
        ext_EMD=None,
    )
    ceemdan.noise_seed(42)
    ceemdan.processes = 1  # Windows safety

    imfs = ceemdan(gold_signal)
    n_imfs = imfs.shape[0]
    print(f"[OK] Decomposition complete -- extracted {n_imfs} stationary components")

    # --- 4. BUILD IMF DATAFRAME ---
    imf_df = pd.DataFrame(index=df_master.index)

    for i in range(n_imfs):
        label = f'IMF{i+1}' if i < n_imfs - 1 else 'Residue'
        imf_df[label] = imfs[i]

    # Sanity check
    reconstructed = imf_df.sum(axis=1).values
    reconstruction_error = np.max(np.abs(gold_signal - reconstructed))
    print(f"  Reconstruction sanity check -- max abs error: {reconstruction_error:.6e}")

    # --- 5. FEATURE ENGINEERING: IMF TEMPORAL FEATURES ---
    print("\nEngineering temporal features for every IMF component...")
    imf_columns = list(imf_df.columns)

    for col in imf_columns:
        imf_df[f'{col}_Lag1']  = imf_df[col].shift(1)
        imf_df[f'{col}_Lag3']  = imf_df[col].shift(3)
        imf_df[f'{col}_Lag7']  = imf_df[col].shift(7)
        imf_df[f'{col}_Roll14'] = imf_df[col].rolling(window=14).mean()

    # --- 6. FEATURE ENGINEERING: EXOGENOUS MACRO FEATURES ---
    print("Engineering exogenous macro-economic features (using log returns where appropriate)...")

    # Add stationary external variables
    exog_cols = [
        'Silver_Close_LogReturn', 'DXY_Close_LogReturn', 'SP500_Close_LogReturn', 
        'VIX_Close_LogReturn', 'EGP_USD_Close_LogReturn',
        'US_10Yr_Yield', 'Egypt_Inflation_YoY', 'CBE_Interest_Rate'
    ]
    for col in exog_cols:
        imf_df[col] = df_master[col]

    # Add lags for the log returns
    lag_cols = ['Silver_Close_LogReturn', 'DXY_Close_LogReturn', 'SP500_Close_LogReturn', 
                'VIX_Close_LogReturn', 'EGP_USD_Close_LogReturn']
    for col in lag_cols:
        imf_df[f'{col}_Lag1']  = df_master[col].shift(1)
        imf_df[f'{col}_Lag3']  = df_master[col].shift(3)
        imf_df[f'{col}_Lag7']  = df_master[col].shift(7)
        imf_df[f'{col}_Roll14'] = df_master[col].rolling(window=14).mean()

    # Gold/Silver Ratio (absolute is fine as a relative indicator)
    imf_df['Gold_Silver_Ratio'] = df_master['Gold_Close'] / df_master['Silver_Close']

    # Carry the raw Gold_Close and LogReturn for evaluation reversal
    imf_df['Gold_Close'] = df_master['Gold_Close']
    imf_df['Gold_Close_LogReturn'] = df_master['Gold_Close_LogReturn']

    # --- 7. CLEANUP ---
    imf_df.dropna(inplace=True)
    print(f"[OK] Stationary feature matrix built: {len(imf_df)} rows x {len(imf_df.columns)} cols")

    # --- 8. SAVE TO DISK ---
    imf_df.to_csv(output_file)
    print(f"\n[OK] Saved stationary CEEMDAN dataset to: {output_file}")

    # --- 9. VISUALISATION ---
    print("\nGenerating Stationary CEEMDAN decomposition plots...")

    fig, axes = plt.subplots(n_imfs + 1, 1, figsize=(16, 3 * (n_imfs + 1)), sharex=True)

    axes[0].plot(df_master.index, gold_signal, color='black', linewidth=0.8)
    axes[0].set_ylabel('Log Return', fontsize=9)
    axes[0].set_title('CEEMDAN Decomposition of Stationary Gold Log Returns', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    for i in range(n_imfs):
        label = f'IMF{i+1}' if i < n_imfs - 1 else 'Residue'
        color = 'steelblue' if i < 2 else ('orange' if i == 2 else 'green')
        if i == n_imfs - 1:
            color = 'crimson'
        axes[i + 1].plot(df_master.index, imfs[i], color=color, linewidth=0.6)
        axes[i + 1].set_ylabel(label, fontsize=9)
        axes[i + 1].grid(True, alpha=0.3)

    plt.xlabel('Date', fontsize=11)
    plt.tight_layout()
    decomp_path = os.path.join(PLOTS_DIR, 'ceemdan_decomposition_stationary.png')
    plt.savefig(decomp_path, dpi=150)
    plt.show()
    print(f"  Plot saved to: {decomp_path}")

    print("\n" + "=" * 60)
    print("  STATIONARY CEEMDAN DECOMPOSITION COMPLETE")
    print("=" * 60)

if __name__ == '__main__':
    main()
