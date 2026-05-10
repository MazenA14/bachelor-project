"""
CEEMDAN unified pipeline (src-extra-variables)

Phase 1 — Unified decomposition (single CEEMDAN per track, trials=50):
  - Track 1 (price): Gold_Close -> core_ceemdan_price_imfs.csv
  - Track 2 (stationary): Gold_Close_LogReturn -> core_ceemdan_stationary_imfs.csv

Phase 2 — Fork feature engineering (no repeated CEEMDAN):
  IMF cores for 02 / 02b / 02c all come from Gold_Close_LogReturn CEEMDAN (stationary /
  differencing-consistent). Pure vs enriched vs stationary differs only in exogenous features.
  - 02_ceemdan_imfs_dataset.csv — log-return IMFs + IMF-only features (+ Gold diagnostics)
  - 02b_ceemdan_enriched_dataset.csv — log-return IMFs + level macro columns & lags
  - 02c_ceemdan_stationary_dataset.csv — log-return IMFs + log-return macro blocks

  Price-level CEEMDAN on Gold_Close is still computed for diagnostics/plots and core_ceemdan_price_imfs.csv.

Set CEEMDAN_CAUSAL=1 for rolling-window decomposition (reduces look-ahead bias; slow).
"""
from __future__ import annotations

import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ceemdan_common import (
    CEEMDAN_CAUSAL,
    CEEMDAN_TRIALS,
    FINAL_DIR,
    PLOTS_DIR,
    PROCESSED_DIR,
    build_enriched_dataset,
    build_pure_dataset,
    build_stationary_dataset,
    full_series_ceemdan,
    rolling_ceemdan_series,
)

warnings.filterwarnings("ignore")


def _imf_labels(n_rows: int) -> list[str]:
    return [f"IMF{i + 1}" if i < n_rows - 1 else "Residue" for i in range(n_rows)]


def _imfs_array_to_df(index: pd.DatetimeIndex, imfs: np.ndarray) -> pd.DataFrame:
    labels = _imf_labels(imfs.shape[0])
    data = {labels[i]: imfs[i] for i in range(imfs.shape[0])}
    return pd.DataFrame(data, index=index)


def main() -> None:
    os.makedirs(FINAL_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    input_file = os.path.join(PROCESSED_DIR, "01_master_metals_dataset.csv")
    core_price_path = os.path.join(FINAL_DIR, "core_ceemdan_price_imfs.csv")
    core_stat_path = os.path.join(FINAL_DIR, "core_ceemdan_stationary_imfs.csv")
    out_pure = os.path.join(FINAL_DIR, "02_ceemdan_imfs_dataset.csv")
    out_enriched = os.path.join(FINAL_DIR, "02b_ceemdan_enriched_dataset.csv")
    out_stationary = os.path.join(FINAL_DIR, "02c_ceemdan_stationary_dataset.csv")

    print("=" * 60)
    print("  CEEMDAN unified pipeline — decomposition + feature forks")
    print("=" * 60)

    try:
        df_master = pd.read_csv(input_file, index_col="Date", parse_dates=["Date"])
        print(f"[OK] Master dataset loaded: {len(df_master)} rows")
    except FileNotFoundError:
        print(f"[!] Missing {input_file}. Run 01_data_gathering_metals.py first.")
        sys.exit(1)

    gold_price = df_master["Gold_Close"].values.astype(float)

    # --- Track 1: price-level CEEMDAN ---
    print("\n--- Price-level CEEMDAN (Gold_Close) ---")
    print(f"  trials={CEEMDAN_TRIALS}, causal={CEEMDAN_CAUSAL}")
    if not CEEMDAN_CAUSAL:
        imfs_p, err_p = full_series_ceemdan(gold_price, trials=CEEMDAN_TRIALS)
        print(f"  Reconstruction max abs error (price): {err_p:.6e}")
        if err_p > 1e-3:
            print("  [!] Reconstruction error is non-negligible; consider raising CEEMDAN_TRIALS.")
        df_core_price = _imfs_array_to_df(df_master.index, imfs_p)
    else:
        print("  Running rolling causal CEEMDAN (this can take a long time)...")
        mat_p, n_comp = rolling_ceemdan_series(gold_price, trials=CEEMDAN_TRIALS)
        labels = [f"IMF{i + 1}" if i < n_comp - 1 else "Residue" for i in range(n_comp)]
        df_core_price = pd.DataFrame(mat_p, index=df_master.index, columns=labels)
        recon = mat_p.sum(axis=1)
        err_p = float(np.max(np.abs(gold_price - recon)))
        print(f"  Reconstruction max abs error (causal sum check): {err_p:.6e}")

    df_core_price.to_csv(core_price_path)
    print(f"[OK] Saved price core IMFs: {core_price_path}")

    # --- Track 2: stationary CEEMDAN on log returns ---
    print("\n--- Stationary CEEMDAN (Gold_Close_LogReturn) ---")
    master_lr = df_master.copy()
    cols_lr = ["Gold_Close", "Silver_Close", "DXY_Close", "SP500_Close", "VIX_Close", "EGP_USD_Close"]
    for col in cols_lr:
        master_lr[f"{col}_LogReturn"] = np.log(master_lr[col] / master_lr[col].shift(1))
    master_lr.dropna(inplace=True)
    gold_lr = master_lr["Gold_Close_LogReturn"].values.astype(float)

    if not CEEMDAN_CAUSAL:
        imfs_s, err_s = full_series_ceemdan(gold_lr, trials=CEEMDAN_TRIALS)
        print(f"  Reconstruction max abs error (log-return): {err_s:.6e}")
        df_core_stat = _imfs_array_to_df(master_lr.index, imfs_s)
    else:
        print("  Running rolling causal CEEMDAN on log returns...")
        mat_s, n_comp_s = rolling_ceemdan_series(gold_lr, trials=CEEMDAN_TRIALS)
        labels_s = [f"IMF{i + 1}" if i < n_comp_s - 1 else "Residue" for i in range(n_comp_s)]
        df_core_stat = pd.DataFrame(mat_s, index=master_lr.index, columns=labels_s)
        recon_s = mat_s.sum(axis=1)
        err_s = float(np.max(np.abs(gold_lr - recon_s)))
        print(f"  Reconstruction max abs error (causal log-return): {err_s:.6e}")

    df_core_stat.to_csv(core_stat_path)
    print(f"[OK] Saved stationary core IMFs: {core_stat_path}")

    if not CEEMDAN_CAUSAL:
        print(
            "\n  NOTE: Single-pass CEEMDAN uses the full sample and may leak test-period"
            "\n        information into IMFs via spline end-points. For thesis-grade causal"
            "\n        IMFs, set environment variable CEEMDAN_CAUSAL=1 (rolling window)."
        )

    # --- Phase 2: feature-engineered forks (all IMF tracks = log-return CEEMDAN core) ---
    print("\n--- Building forked datasets (pure / enriched / stationary) ---")
    print("    Using stationary IMF core (Gold_Close_LogReturn) for 02 / 02b / 02c — XGBoost+Ridge")
    print("    operates in differencing / log-return space like Architecture A.")

    df_pure = build_pure_dataset(df_core_stat, df_master)
    df_pure.to_csv(out_pure)
    print(f"[OK] Pure dataset -> {out_pure} ({df_pure.shape[1]} cols)")

    df_enr = build_enriched_dataset(df_core_stat, df_master)
    df_enr.to_csv(out_enriched)
    print(f"[OK] Enriched dataset -> {out_enriched} ({df_enr.shape[1]} cols)")

    df_stat = build_stationary_dataset(df_core_stat, master_lr)
    df_stat.to_csv(out_stationary)
    print(f"[OK] Stationary dataset -> {out_stationary} ({df_stat.shape[1]} cols)")

    # --- Plot: price decomposition ---
    print("\nGenerating CEEMDAN plot (price track)...")
    n_comp = len(df_core_price.columns)
    fig, axes = plt.subplots(n_comp + 1, 1, figsize=(16, 3 * (n_comp + 1)), sharex=True)
    axes[0].plot(df_master.index, gold_price, color="black", linewidth=0.8)
    axes[0].set_ylabel("Original", fontsize=9)
    axes[0].set_title("CEEMDAN Decomposition of Gold Close Price (unified pipeline)", fontsize=14)
    axes[0].grid(True, alpha=0.3)
    for i, col in enumerate(df_core_price.columns):
        color = "steelblue" if i < 2 else ("orange" if i == 2 else "green")
        if col == "Residue":
            color = "crimson"
        axes[i + 1].plot(df_core_price.index, df_core_price[col], color=color, linewidth=0.6)
        axes[i + 1].set_ylabel(col, fontsize=9)
        axes[i + 1].grid(True, alpha=0.3)
    plt.xlabel("Date", fontsize=11)
    plt.tight_layout()
    decomp_path = os.path.join(PLOTS_DIR, "ceemdan_decomposition.png")
    plt.savefig(decomp_path, dpi=150)
    plt.show()
    print(f"  Plot saved to: {decomp_path}")

    print("\n" + "=" * 60)
    print("  UNIFIED CEEMDAN PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
