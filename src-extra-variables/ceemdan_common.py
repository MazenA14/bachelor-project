"""
Shared CEEMDAN-XGBoost utilities for src-extra-variables:
- IMF band assignment (high / mid / low / residue)
- Component-specific feature engineering and column selection
- Gold + silver cross-metal context (aligned with 05c full gold model: Silver_Close_LogReturn
  and engineered gold/silver signals; no ablation / no gold same-day log-return as feature)
- Optional causal (rolling-window) CEEMDAN to reduce look-ahead bias
"""
from __future__ import annotations

import os
import re
import warnings
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PyEMD import CEEMDAN

warnings.filterwarnings("ignore")

Track = Literal["pure", "enriched", "stationary"]

PROCESSED_DIR = "../data-extra-variables/02_processed/"
FINAL_DIR = "../data-extra-variables/03_final/"
MODELS_DIR = "../models-extra-variables/"
PLOTS_DIR = "../plots-extra-variables/ceemdan/"

CEEMDAN_TRIALS = int(os.environ.get("CEEMDAN_TRIALS", "50"))
CEEMDAN_EPSILON = 0.005
# Set CEEMDAN_CAUSAL=1 for rolling-window decomposition (slow, methodology-safe).
CEEMDAN_CAUSAL = os.environ.get("CEEMDAN_CAUSAL", "0").strip().lower() in ("1", "true", "yes")
CEEMDAN_ROLL_WINDOW = int(os.environ.get("CEEMDAN_ROLL_WINDOW", "756"))

TRAIN_END = "2023-12-31"
VAL_START = "2024-01-01"
VAL_END = "2024-12-31"
TEST_START = "2025-01-01"
TRAIN_VAL_END = "2024-12-31"

GOLD_GLOBAL_COLS = ["Gold_Close_ReturnVol14", "Gold_Close_BBWidth20", "Gold_Close_ATRproxy14"]


def add_silver_logreturn_engineering(out: pd.DataFrame, master: pd.DataFrame) -> None:
    """
    Silver log-return block (same spirit as 05c gold full model: Silver_Close_LogReturn as input).
    Fills only missing columns so stationary exog can pre-define Silver_Close_LogReturn* lags.
    """
    s = master["Silver_Close"].reindex(out.index).astype(float)
    if "Silver_Close_LogReturn" not in out.columns:
        out["Silver_Close_LogReturn"] = np.log(s / s.shift(1))
    lr = out["Silver_Close_LogReturn"]
    if "Silver_Close_LogReturn_Lag1" not in out.columns:
        out["Silver_Close_LogReturn_Lag1"] = lr.shift(1)
    if "Silver_Close_LogReturn_Lag3" not in out.columns:
        out["Silver_Close_LogReturn_Lag3"] = lr.shift(3)
    if "Silver_Close_LogReturn_Lag7" not in out.columns:
        out["Silver_Close_LogReturn_Lag7"] = lr.shift(7)
    if "Silver_Close_LogReturn_Roll14" not in out.columns:
        out["Silver_Close_LogReturn_Roll14"] = lr.rolling(14).mean()
    if "Silver_Close_ReturnVol14" not in out.columns:
        out["Silver_Close_ReturnVol14"] = s.pct_change().rolling(14).std()


def metal_context_columns(df: pd.DataFrame) -> List[str]:
    """Gold volatility + silver log-return + cross-metal ratio (05c-style gold/silver inputs)."""
    cols: List[str] = []
    for g in GOLD_GLOBAL_COLS:
        if g in df.columns:
            cols.append(g)
    if "Gold_Silver_Ratio" in df.columns:
        cols.append("Gold_Silver_Ratio")
    for c in df.columns:
        if c == "Silver_Close_LogReturn" or c.startswith("Silver_Close_LogReturn_"):
            cols.append(c)
        if c == "Silver_Close_ReturnVol14":
            cols.append(c)
    return sorted(set(cols))


MASTER_PRICE_EXOG = [
    "Silver_Close",
    "DXY_Close",
    "SP500_Close",
    "VIX_Close",
    "US_10Yr_Yield",
    "EGP_USD_Close",
    "Egypt_Inflation_YoY",
    "CBE_Interest_Rate",
]
MASTER_PRICE_LAG_BASES = ["Silver_Close", "DXY_Close", "SP500_Close", "VIX_Close", "EGP_USD_Close"]

MASTER_STATIONARY_EXOG_LEVEL = ["US_10Yr_Yield", "Egypt_Inflation_YoY", "CBE_Interest_Rate"]
MASTER_STATIONARY_LOGRET = [
    "Silver_Close_LogReturn",
    "DXY_Close_LogReturn",
    "SP500_Close_LogReturn",
    "VIX_Close_LogReturn",
    "EGP_USD_Close_LogReturn",
]

def natural_imf_targets(columns: Sequence[str]) -> List[str]:
    """Return IMF1..IMFn then Residue in numeric order."""

    def sort_key(c: str) -> Tuple[int, int]:
        if c == "Residue":
            return (1, 10**9)
        m = re.match(r"IMF(\d+)$", c)
        if m:
            return (0, int(m.group(1)))
        return (2, hash(c) % 10**6)

    raw = [c for c in columns if (c.startswith("IMF") or c == "Residue") and "_" not in c]
    return sorted(raw, key=sort_key)


def imf_band(component: str, ordered_targets: Sequence[str]) -> str:
    """Classify IMF component into frequency band; Residue is separate."""
    if component == "Residue":
        return "residue"
    imf_only = [t for t in ordered_targets if t != "Residue"]
    n_imf = len(imf_only)
    m = re.match(r"IMF(\d+)$", component)
    if not m or n_imf == 0:
        return "mid"
    idx0 = int(m.group(1)) - 1
    third = max(1, n_imf // 3)
    if idx0 < third:
        return "high"
    if idx0 < 2 * third:
        return "mid"
    return "low"


def _make_ceemdan(trials: int = CEEMDAN_TRIALS) -> CEEMDAN:
    ce = CEEMDAN(trials=trials, epsilon=CEEMDAN_EPSILON, ext_EMD=None)
    ce.noise_seed(42)
    ce.processes = 1
    return ce


def run_ceemdan_on_segment(segment: np.ndarray, trials: int = CEEMDAN_TRIALS) -> np.ndarray:
    """Run CEEMDAN on a 1D array; returns imfs (n_imfs, len(segment))."""
    ce = _make_ceemdan(trials=trials)
    return ce(np.asarray(segment, dtype=float))


def ceemdan_align_last(imfs: np.ndarray, n_target: int) -> np.ndarray:
    """Take last column of imfs and pad/truncate to n_target rows."""
    last = imfs[:, -1].astype(float)
    if last.shape[0] < n_target:
        return np.pad(last, (0, n_target - last.shape[0]), constant_values=np.nan)
    return last[:n_target]


def rolling_ceemdan_series(
    signal: np.ndarray,
    window: int = CEEMDAN_ROLL_WINDOW,
    trials: int = CEEMDAN_TRIALS,
    min_segment: int = 64,
    progress_every: int = 200,
) -> Tuple[np.ndarray, int]:
    """
    Causal CEEMDAN: at each t, decompose signal[max(0,t-w+1):t+1] and keep endpoint values.
    Returns (matrix T x n_imfs_aligned, n_imfs_aligned).
    """
    n = len(signal)
    n_target = None
    out_rows: List[np.ndarray] = []

    for t in range(n):
        start = max(0, t - window + 1)
        seg = signal[start : t + 1]
        if len(seg) < min_segment:
            imfs = run_ceemdan_on_segment(seg, trials=trials)
        else:
            imfs = run_ceemdan_on_segment(seg, trials=trials)

        if n_target is None:
            n_target = imfs.shape[0]
        row = ceemdan_align_last(imfs, n_target)
        out_rows.append(row)

        if progress_every and (t + 1) % progress_every == 0:
            print(f"    causal CEEMDAN progress: {t + 1}/{n} timestamps")

    mat = np.vstack(out_rows)
    return mat, int(n_target or 0)


def full_series_ceemdan(signal: np.ndarray, trials: int = CEEMDAN_TRIALS) -> Tuple[np.ndarray, float]:
    """Single-pass CEEMDAN (fast; includes look-ahead from global spline extrema)."""
    imfs = run_ceemdan_on_segment(signal, trials=trials)
    recon = np.sum(imfs, axis=0)
    err = float(np.max(np.abs(signal - recon)))
    return imfs, err


def add_gold_volatility_columns(df: pd.DataFrame, master: pd.DataFrame) -> None:
    """ATR-like / Bollinger width / return volatility derived from Gold_Close (in-place)."""
    gold = master["Gold_Close"].reindex(df.index)
    ret = gold.pct_change()
    df["Gold_Close_ReturnVol14"] = ret.rolling(14).std()
    ma20 = gold.rolling(20).mean()
    sd20 = gold.rolling(20).std()
    df["Gold_Close_BBWidth20"] = (4.0 * sd20) / ma20.replace(0, np.nan)
    df["Gold_Close_ATRproxy14"] = (gold * ret.abs()).rolling(14).mean()


def add_imf_temporal_for_band(
    out: pd.DataFrame,
    comp: str,
    series: pd.Series,
    band: str,
) -> None:
    if band == "high":
        out[f"{comp}_Lag1"] = series.shift(1)
        out[f"{comp}_Lag2"] = series.shift(2)
        out[f"{comp}_Roll7"] = series.rolling(7).mean()
    elif band == "mid":
        out[f"{comp}_Lag1"] = series.shift(1)
        out[f"{comp}_Lag3"] = series.shift(3)
        out[f"{comp}_Lag7"] = series.shift(7)
        out[f"{comp}_Roll14"] = series.rolling(14).mean()
    else:  # low or residue
        out[f"{comp}_Lag14"] = series.shift(14)
        out[f"{comp}_Lag30"] = series.shift(30)
        out[f"{comp}_Roll30"] = series.rolling(30).mean()


def add_residue_trend_time(out: pd.DataFrame) -> None:
    n = len(out)
    if n <= 1:
        out["Residue_TrendTime"] = 0.0
    else:
        out["Residue_TrendTime"] = np.linspace(0.0, 1.0, n)


def macro_feature_pool(df: pd.DataFrame, ordered_targets: Sequence[str]) -> List[str]:
    """Columns shared across rows that are not raw IMF targets or per-IMF engineered prefixes."""
    raw = set(ordered_targets) | {"Gold_Close", "Gold_Close_LogReturn"}
    out: List[str] = []
    for c in df.columns:
        if c in raw:
            continue
        if c == "Residue_TrendTime":
            continue
        if any(c.startswith(t + "_") for t in ordered_targets):
            continue
        out.append(c)
    return sorted(out)


def mid_band_macro_subset(macro_cols: Sequence[str]) -> List[str]:
    """Mid-frequency IMF models: cross-asset movers including silver (05c-style cross-metal)."""
    keys = ("Silver_Close", "VIX_Close", "DXY_Close", "SP500_Close")
    return sorted([c for c in macro_cols if any(k in c for k in keys)])


def build_pure_dataset(imf_core: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    ordered = natural_imf_targets(imf_core.columns)
    out = imf_core.copy()
    add_gold_volatility_columns(out, master)
    add_silver_logreturn_engineering(out, master)
    gc = master["Gold_Close"].reindex(out.index)
    out["Gold_Silver_Ratio"] = gc / master["Silver_Close"].reindex(out.index)

    for comp in ordered:
        band = imf_band(comp, ordered)
        if band == "residue":
            band = "low"
        add_imf_temporal_for_band(out, comp, out[comp], band)

    add_residue_trend_time(out)
    out["Gold_Close"] = gc
    out["Gold_Close_LogReturn"] = np.log(out["Gold_Close"] / out["Gold_Close"].shift(1))
    out.dropna(inplace=True)
    return out


def _enriched_exog_block(imf_index: pd.DatetimeIndex, master: pd.DataFrame) -> pd.DataFrame:
    block = pd.DataFrame(index=imf_index)
    for col in MASTER_PRICE_EXOG:
        block[col] = master[col].reindex(imf_index)

    for col in MASTER_PRICE_LAG_BASES:
        s = master[col].reindex(imf_index)
        block[f"{col}_Lag1"] = s.shift(1)
        block[f"{col}_Lag3"] = s.shift(3)
        block[f"{col}_Lag7"] = s.shift(7)
        block[f"{col}_Roll14"] = s.rolling(14).mean()

    gc = master["Gold_Close"].reindex(imf_index)
    sv = master["Silver_Close"].reindex(imf_index)
    block["Gold_Silver_Ratio"] = gc / sv
    return block


def build_enriched_dataset(imf_core: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    ordered = natural_imf_targets(imf_core.columns)
    out = imf_core.copy()
    add_gold_volatility_columns(out, master)
    exog = _enriched_exog_block(out.index, master)
    for col in exog.columns:
        out[col] = exog[col]
    add_silver_logreturn_engineering(out, master)

    for comp in ordered:
        band = imf_band(comp, ordered)
        if band == "residue":
            band = "low"
        add_imf_temporal_for_band(out, comp, out[comp], band)

    add_residue_trend_time(out)
    out["Gold_Close"] = master["Gold_Close"].reindex(out.index)
    out["Gold_Close_LogReturn"] = np.log(out["Gold_Close"] / out["Gold_Close"].shift(1))
    out.dropna(inplace=True)
    return out


def _stationary_exog_block(imf_index: pd.DatetimeIndex, master_lr: pd.DataFrame) -> pd.DataFrame:
    block = pd.DataFrame(index=imf_index)
    for col in MASTER_STATIONARY_LOGRET:
        s = master_lr[col].reindex(imf_index)
        block[col] = s
        block[f"{col}_Lag1"] = s.shift(1)
        block[f"{col}_Lag3"] = s.shift(3)
        block[f"{col}_Lag7"] = s.shift(7)
        block[f"{col}_Roll14"] = s.rolling(14).mean()

    for col in MASTER_STATIONARY_EXOG_LEVEL:
        block[col] = master_lr[col].reindex(imf_index)

    gc = master_lr["Gold_Close"].reindex(imf_index)
    sv = master_lr["Silver_Close"].reindex(imf_index)
    block["Gold_Silver_Ratio"] = gc / sv
    return block


def build_stationary_dataset(imf_core: pd.DataFrame, master_with_lr: pd.DataFrame) -> pd.DataFrame:
    ordered = natural_imf_targets(imf_core.columns)
    out = imf_core.copy()
    add_gold_volatility_columns(out, master_with_lr)

    exog = _stationary_exog_block(out.index, master_with_lr)
    for col in exog.columns:
        out[col] = exog[col]
    add_silver_logreturn_engineering(out, master_with_lr)

    for comp in ordered:
        band = imf_band(comp, ordered)
        if band == "residue":
            band = "low"
        add_imf_temporal_for_band(out, comp, out[comp], band)

    add_residue_trend_time(out)
    out["Gold_Close"] = master_with_lr["Gold_Close"].reindex(out.index)
    out["Gold_Close_LogReturn"] = master_with_lr["Gold_Close_LogReturn"].reindex(out.index)
    out.dropna(inplace=True)
    return out


def feature_columns_pure(df: pd.DataFrame, target: str, ordered_targets: Sequence[str]) -> List[str]:
    band = imf_band(target, ordered_targets)
    own = sorted([c for c in df.columns if c.startswith(target + "_")])
    metal = metal_context_columns(df)
    if band == "high":
        return sorted(set(own + metal))
    if band == "mid":
        return sorted(set(own + metal))
    if band == "low":
        return sorted(set(own + metal))
    res = sorted([c for c in df.columns if c.startswith("Residue_")])
    return sorted(set(res + metal))


def feature_columns_enriched(df: pd.DataFrame, target: str, ordered_targets: Sequence[str]) -> List[str]:
    band = imf_band(target, ordered_targets)
    own = sorted([c for c in df.columns if c.startswith(target + "_")])
    macro_all = macro_feature_pool(df, ordered_targets)
    mid_m = mid_band_macro_subset(macro_all)
    metal = metal_context_columns(df)

    if band == "high":
        return sorted(set(own + metal))
    if band == "mid":
        return sorted(set(own + metal + mid_m))
    if band == "low":
        return sorted(set(own + macro_all))
    res = sorted([c for c in df.columns if c.startswith("Residue_")])
    return sorted(set(res + metal))


def feature_columns_stationary(df: pd.DataFrame, target: str, ordered_targets: Sequence[str]) -> List[str]:
    return feature_columns_enriched(df, target, ordered_targets)


def feature_columns_for(
    df: pd.DataFrame,
    target: str,
    track: Track,
    ordered_targets: Optional[Sequence[str]] = None,
) -> List[str]:
    ot = list(ordered_targets) if ordered_targets is not None else natural_imf_targets(df.columns)
    if track == "pure":
        return feature_columns_pure(df, target, ot)
    if track == "enriched":
        return feature_columns_enriched(df, target, ot)
    return feature_columns_stationary(df, target, ot)


def use_ridge_for_residue() -> bool:
    return True
