import os
import pandas as pd
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CEEMDAN-XGBoost Pipeline (STATIONARY)  --  Step 2c: Sub-Model Training
# =============================================================================
# For every stationary IMF component we train an independent XGBoost regressor.
#
# Because the IMFs were derived from Log Returns, they are strictly stationary.
# This aligns perfectly with the core assumption of tree-based models: that
# test data will fall within the bounds of the training data.
#
# Each sub-model receives:
#   1. Its own stationary IMF temporal features (Lag1, Lag3, Lag7, Roll14)
#   2. ALL stationary exogenous macro features (Silver returns, DXY returns, etc.)
# =============================================================================

# --- 1. CONFIGURATION ---
FINAL_DIR   = '../data-extra-variables/03_final/'
MODELS_DIR  = '../models-extra-variables/'

os.makedirs(MODELS_DIR, exist_ok=True)

input_file = os.path.join(FINAL_DIR, '02c_ceemdan_stationary_dataset.csv')

print("=" * 60)
print("  CEEMDAN-XGBoost (STATIONARY) -- Phase 2c: Sub-Model Training")
print("=" * 60)

# --- 2. LOAD STATIONARY CEEMDAN FEATURE MATRIX ---
try:
    df = pd.read_csv(input_file, index_col='Date', parse_dates=['Date'])
    print(f"[OK] Stationary CEEMDAN dataset loaded: {len(df)} rows x {len(df.columns)} cols")
except FileNotFoundError:
    print(f"[!]  Error: {input_file} not found. Run 07c_ceemdan_decomposition_stationary first.")
    exit()

# --- 3. IDENTIFY IMF COMPONENT COLUMNS ---
# The raw IMF columns are: IMF1, IMF2, ..., Residue  (no suffix)
imf_targets = [c for c in df.columns
               if (c.startswith('IMF') or c == 'Residue')
               and '_' not in c]

print(f"\nTarget components to model: {imf_targets}")

# --- 4. IDENTIFY SHARED EXOGENOUS FEATURE COLUMNS ---
exog_features = [c for c in df.columns
                 if not c.startswith('IMF')
                 and c != 'Residue'
                 and not c.startswith('Residue_')
                 and c not in ['Gold_Close', 'Gold_Close_LogReturn']]

print(f"Shared stationary exogenous features ({len(exog_features)}):")
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

# --- 6. TRAIN STATIONARY XGBOOST SUB-MODELS ---
print("\n" + "-" * 60)
print("  Training STATIONARY XGBoost sub-models...")
print("-" * 60)

for target in imf_targets:
    print(f"\n>> Training stationary sub-model for [{target}]...")

    imf_own_features = [c for c in df.columns if c.startswith(target + '_')]
    feature_cols = imf_own_features + exog_features

    if not imf_own_features:
        print(f"  [!]  No IMF features found for {target}, skipping.")
        continue

    X_train = train[feature_cols]
    y_train = train[target]
    X_val   = val[feature_cols]
    y_val   = val[target]

    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=50,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    model_path = os.path.join(MODELS_DIR, f'ceemdan_stationary_xgb_{target.lower()}.json')
    model.save_model(model_path)
    print(f"  [OK] Saved -> {model_path}")

print("\n" + "=" * 60)
print("  ALL STATIONARY CEEMDAN SUB-MODELS TRAINED AND SAVED")
print("=" * 60)
