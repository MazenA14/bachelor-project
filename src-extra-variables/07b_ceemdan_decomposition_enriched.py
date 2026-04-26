import os
import pandas as pd
import numpy as np
from PyEMD import CEEMDAN
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CEEMDAN-XGBoost Pipeline (ENRICHED)  --  Step 1b: Decomposition + Exogenous
# =============================================================================
# Variant B of the CEEMDAN pipeline.  The CEEMDAN decomposition itself is
# identical to variant A (same IMFs), but the output dataset is ENRICHED
# with all external macro-economic & cross-market features:
#
#   - Silver, DXY, S&P500, VIX  (global markets)
#   - US 10-Year Treasury Yield  (monetary policy)
#   - EGP/USD, Egypt Inflation, CBE Interest Rate  (local Egyptian)
#   - Gold/Silver Ratio  (cross-asset indicator)
#
# For every IMF sub-model, XGBoost will now see both the IMF's own temporal
# features AND the broader market context -- matching the information
# richness of Architecture A (Log Returns).
# =============================================================================


def main():
    # --- 1. CONFIGURATION & DIRECTORY SETUP ---
    PROCESSED_DIR = '../data-extra-variables/02_processed/'
    FINAL_DIR     = '../data-extra-variables/03_final/'
    PLOTS_DIR     = '../plots-extra-variables/ceemdan/'

    os.makedirs(FINAL_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    input_file  = os.path.join(PROCESSED_DIR, '01_master_metals_dataset.csv')
    output_file = os.path.join(FINAL_DIR, '02b_ceemdan_enriched_dataset.csv')

    print("=" * 60)
    print("  CEEMDAN-XGBoost (ENRICHED) -- Phase 1b: Decomposition")
    print("=" * 60)

    # --- 2. LOAD THE MASTER DATASET ---
    try:
        df_master = pd.read_csv(input_file, index_col='Date', parse_dates=['Date'])
        print(f"[OK] Master dataset loaded: {len(df_master)} rows, {len(df_master.columns)} cols")
    except FileNotFoundError:
        print(f"[!]  Error: {input_file} not found.  Run 01_data_gathering first.")
        exit()

    gold_signal = df_master['Gold_Close'].values

    # --- 3. CEEMDAN DECOMPOSITION ---
    print("\nRunning CEEMDAN decomposition on Gold_Close signal...")
    print("  (Adding Gaussian white noise at multiple amplitudes and averaging...)")

    ceemdan = CEEMDAN(
        trials=100,          # number of ensemble realisations
        epsilon=0.005,       # noise amplitude scaling factor
        ext_EMD=None,        # use default internal EMD
    )
    ceemdan.noise_seed(42)   # reproducibility
    # Use serial processing to avoid Windows multiprocessing spawn issues
    ceemdan.processes = 1

    imfs = ceemdan(gold_signal)
    n_imfs = imfs.shape[0]
    print(f"[OK] Decomposition complete -- extracted {n_imfs} components (IMFs + residue)")

    # --- 4. BUILD IMF DATAFRAME ---
    imf_df = pd.DataFrame(index=df_master.index)

    for i in range(n_imfs):
        label = f'IMF{i+1}' if i < n_imfs - 1 else 'Residue'
        imf_df[label] = imfs[i]

    # Sanity check: the sum of all IMFs should reconstruct the original signal
    reconstructed = imf_df.sum(axis=1).values
    reconstruction_error = np.max(np.abs(gold_signal - reconstructed))
    print(f"  Reconstruction sanity check -- max abs error: {reconstruction_error:.6e}")

    # --- 5. FEATURE ENGINEERING: IMF TEMPORAL FEATURES ---
    print("\nEngineering temporal features for every IMF component...")

    imf_columns = list(imf_df.columns)

    for col in imf_columns:
        # Lags: T-1, T-3, T-7
        imf_df[f'{col}_Lag1']  = imf_df[col].shift(1)
        imf_df[f'{col}_Lag3']  = imf_df[col].shift(3)
        imf_df[f'{col}_Lag7']  = imf_df[col].shift(7)
        # Rolling 14-day average
        imf_df[f'{col}_Roll14'] = imf_df[col].rolling(window=14).mean()

    # --- 6. FEATURE ENGINEERING: EXOGENOUS MACRO FEATURES ---
    print("Engineering exogenous macro-economic & cross-market features...")

    # External variables to enrich each sub-model with
    exog_cols = ['Silver_Close', 'DXY_Close', 'SP500_Close', 'VIX_Close',
                 'US_10Yr_Yield', 'EGP_USD_Close', 'Egypt_Inflation_YoY',
                 'CBE_Interest_Rate']

    # Add raw values
    for col in exog_cols:
        imf_df[col] = df_master[col]

    # Add lagged features (T-1, T-3, T-7) for price-like variables
    lag_cols = ['Silver_Close', 'DXY_Close', 'SP500_Close', 'VIX_Close', 'EGP_USD_Close']
    for col in lag_cols:
        imf_df[f'{col}_Lag1']  = df_master[col].shift(1)
        imf_df[f'{col}_Lag3']  = df_master[col].shift(3)
        imf_df[f'{col}_Lag7']  = df_master[col].shift(7)

    # Add 14-day rolling averages for price-like variables
    for col in lag_cols:
        imf_df[f'{col}_Roll14'] = df_master[col].rolling(window=14).mean()

    # Gold/Silver Ratio (classic macro indicator)
    imf_df['Gold_Silver_Ratio'] = df_master['Gold_Close'] / df_master['Silver_Close']

    # Carry the raw Gold_Close for evaluation reversal
    imf_df['Gold_Close'] = df_master['Gold_Close']

    # --- 7. CLEANUP ---
    # Drop the initial rows made NaN by lagging / rolling
    imf_df.dropna(inplace=True)

    print(f"[OK] Enriched feature matrix built: {len(imf_df)} rows x {len(imf_df.columns)} cols")
    print(f"     (vs. pure IMF-only variant with ~51 cols)")

    # --- 8. SAVE TO DISK ---
    imf_df.to_csv(output_file)
    print(f"\n[OK] Saved enriched CEEMDAN dataset to: {output_file}")

    # --- 9. VISUALISATION ---
    print("\nGenerating CEEMDAN decomposition plots...")

    fig, axes = plt.subplots(n_imfs + 1, 1, figsize=(16, 3 * (n_imfs + 1)), sharex=True)

    # Original signal on top
    axes[0].plot(df_master.index, gold_signal, color='black', linewidth=0.8)
    axes[0].set_ylabel('Original', fontsize=9)
    axes[0].set_title('CEEMDAN Decomposition of Gold Close Price (Enriched Pipeline)', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Each IMF
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
    decomp_path = os.path.join(PLOTS_DIR, 'ceemdan_decomposition_enriched.png')
    plt.savefig(decomp_path, dpi=150)
    plt.show()
    print(f"  Plot saved to: {decomp_path}")

    print("\n" + "=" * 60)
    print("  ENRICHED CEEMDAN DECOMPOSITION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
