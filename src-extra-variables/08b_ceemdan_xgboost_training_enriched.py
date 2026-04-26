import os
import pandas as pd
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CEEMDAN-XGBoost Pipeline (ENRICHED)  --  Step 2b: Sub-Model Training
# =============================================================================
# For every IMF component we train an independent XGBoost regressor, but
# unlike the pure variant (08), each sub-model now receives:
#
#   1. Its own IMF temporal features  (Lag1, Lag3, Lag7, Roll14)
#   2. ALL exogenous macro features   (Silver, DXY, VIX, S&P500, etc.)
#
# This gives each frequency-scale model the same market context that
# Architecture A (Log Returns) had access to.
# =============================================================================

# --- 1. CONFIGURATION ---
FINAL_DIR   = '../data-extra-variables/03_final/'
MODELS_DIR  = '../models-extra-variables/'

os.makedirs(MODELS_DIR, exist_ok=True)

input_file = os.path.join(FINAL_DIR, '02b_ceemdan_enriched_dataset.csv')

print("=" * 60)
print("  CEEMDAN-XGBoost (ENRICHED) -- Phase 2b: Sub-Model Training")
print("=" * 60)

# --- 2. LOAD ENRICHED CEEMDAN FEATURE MATRIX ---
try:
    df = pd.read_csv(input_file, index_col='Date', parse_dates=['Date'])
    print(f"[OK] Enriched CEEMDAN dataset loaded: {len(df)} rows x {len(df.columns)} cols")
except FileNotFoundError:
    print(f"[!]  Error: {input_file} not found.  Run 07b_ceemdan_decomposition_enriched first.")
    exit()

# --- 3. IDENTIFY IMF COMPONENT COLUMNS ---
# The raw IMF columns are: IMF1, IMF2, ..., Residue  (no suffix)
imf_targets = [c for c in df.columns
               if (c.startswith('IMF') or c == 'Residue')
               and '_' not in c]

print(f"\nTarget components to model: {imf_targets}")

# --- 4. IDENTIFY SHARED EXOGENOUS FEATURE COLUMNS ---
# These are the external macro features that EVERY sub-model will receive
# in addition to its own IMF temporal features.
exog_features = [c for c in df.columns
                 if not c.startswith('IMF')
                 and c != 'Residue'
                 and not c.startswith('Residue_')
                 and c != 'Gold_Close']

print(f"Shared exogenous features ({len(exog_features)}):")
for f in exog_features:
    print(f"  - {f}")

# --- 5. CHRONOLOGICAL TIMELINE SPLIT ---
TRAIN_END  = '2023-12-31'
VAL_START  = '2024-01-01'
VAL_END    = '2024-12-31'
TEST_START = '2025-01-01'

train = df[:TRAIN_END]
val   = df[VAL_START:VAL_END]
test  = df[TEST_START:]

print(f"\n  Train : {len(train)} rows  ({train.index.min().date()} -> {train.index.max().date()})")
print(f"  Val   : {len(val)}  rows  ({val.index.min().date()} -> {val.index.max().date()})")
print(f"  Test  : {len(test)}  rows  ({test.index.min().date()} -> {test.index.max().date()})")

# --- 6. TRAIN ENRICHED XGBOOST SUB-MODELS ---
print("\n" + "-" * 60)
print("  Training ENRICHED XGBoost sub-models (IMF features + exogenous)...")
print("-" * 60)

for target in imf_targets:
    print(f"\n>> Training enriched sub-model for [{target}]...")

    # Features for this component: its own lags/rolling + all exogenous
    imf_own_features = [c for c in df.columns if c.startswith(target + '_')]
    feature_cols = imf_own_features + exog_features

    if not imf_own_features:
        print(f"  [!]  No IMF features found for {target}, skipping.")
        continue

    print(f"  IMF-own features : {len(imf_own_features)}")
    print(f"  Exogenous features : {len(exog_features)}")
    print(f"  Total features   : {len(feature_cols)}")

    X_train = train[feature_cols]
    y_train = train[target]
    X_val   = val[feature_cols]
    y_val   = val[target]

    # XGBoost with regularisation to prevent overfitting (as per the paper)
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,          # L1 regularisation
        reg_lambda=1.0,         # L2 regularisation
        random_state=42,
        early_stopping_rounds=50,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Save the enriched sub-model with a distinct name
    model_path = os.path.join(MODELS_DIR, f'ceemdan_enriched_xgb_{target.lower()}.json')
    model.save_model(model_path)
    print(f"  [OK] Saved -> {model_path}")

print("\n" + "=" * 60)
print("  ALL ENRICHED CEEMDAN SUB-MODELS TRAINED AND SAVED")
print("=" * 60)
