import os
import pandas as pd
import numpy as np
from PyEMD import CEEMDAN
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CEEMDAN-XGBoost Pipeline  --  Step 1: Signal Decomposition
# =============================================================================
# This script applies Complete Ensemble Empirical Mode Decomposition with
# Adaptive Noise (CEEMDAN) to the Gold_Close price series.  The raw signal
# is decomposed into Intrinsic Mode Functions (IMFs) that separate:
#   - IMF1-IMF2  -> high-frequency (short-term volatility / noise)
#   - IMF3       -> mid-frequency  (medium-term trend changes)
#   - IMF4+      -> low-frequency  (long-term macro trend)
#
# The individual IMFs (+ residue) are saved alongside engineered temporal
# features (lags, rolling averages) so that each can be modelled
# independently by XGBoost in the next phase.
# =============================================================================


def main():
    # --- 1. CONFIGURATION & DIRECTORY SETUP ---
    PROCESSED_DIR = '../data-extra-variables/02_processed/'
    FINAL_DIR     = '../data-extra-variables/03_final/'
    PLOTS_DIR     = '../plots-extra-variables/ceemdan/'

    os.makedirs(FINAL_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    input_file  = os.path.join(PROCESSED_DIR, '01_master_metals_dataset.csv')
    output_file = os.path.join(FINAL_DIR, '02_ceemdan_imfs_dataset.csv')

    print("=" * 60)
    print("  CEEMDAN-XGBoost Pipeline -- Phase 1: Signal Decomposition")
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

    # --- 5. FEATURE ENGINEERING FOR EACH IMF ---
    print("\nEngineering temporal features for every IMF component...")

    # We create lagged and rolling features for each component so that XGBoost
    # has the same time-series memory it received in the main pipeline.
    imf_columns = list(imf_df.columns)

    for col in imf_columns:
        # Lags: T-1, T-3, T-7
        imf_df[f'{col}_Lag1']  = imf_df[col].shift(1)
        imf_df[f'{col}_Lag3']  = imf_df[col].shift(3)
        imf_df[f'{col}_Lag7']  = imf_df[col].shift(7)
        # Rolling 14-day average
        imf_df[f'{col}_Roll14'] = imf_df[col].rolling(window=14).mean()

    # Also carry the raw Gold_Close alongside so the evaluation script can
    # reverse predictions back to price space.
    imf_df['Gold_Close'] = df_master['Gold_Close']

    # Drop the initial rows made NaN by lagging / rolling
    imf_df.dropna(inplace=True)

    print(f"[OK] Feature matrix built: {len(imf_df)} rows x {len(imf_df.columns)} cols")

    # --- 6. SAVE TO DISK ---
    imf_df.to_csv(output_file)
    print(f"\n[OK] Saved CEEMDAN dataset to: {output_file}")

    # --- 7. VISUALISATION ---
    print("\nGenerating CEEMDAN decomposition plots...")

    fig, axes = plt.subplots(n_imfs + 1, 1, figsize=(16, 3 * (n_imfs + 1)), sharex=True)

    # Original signal on top
    axes[0].plot(df_master.index, gold_signal, color='black', linewidth=0.8)
    axes[0].set_ylabel('Original', fontsize=9)
    axes[0].set_title('CEEMDAN Decomposition of Gold Close Price', fontsize=14)
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
    decomp_path = os.path.join(PLOTS_DIR, 'ceemdan_decomposition.png')
    plt.savefig(decomp_path, dpi=150)
    plt.show()
    print(f"  Plot saved to: {decomp_path}")

    print("\n" + "=" * 60)
    print("  CEEMDAN DECOMPOSITION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
