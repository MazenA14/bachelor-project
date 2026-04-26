import os
import pandas as pd
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CEEMDAN-XGBoost Pipeline  --  Step 2: Sub-Model Training
# =============================================================================
# For every IMF component extracted in Phase 1 we train an independent
# XGBoost regressor.  Each sub-model learns the temporal dynamics of its
# own frequency scale (high / mid / low).
#
# The predictions from all sub-models will later be summed to reconstruct
# the final gold price forecast (multi-scale information fusion).
# =============================================================================

# --- 1. CONFIGURATION ---
FINAL_DIR   = '../data-extra-variables/03_final/'
MODELS_DIR  = '../models-extra-variables/'

os.makedirs(MODELS_DIR, exist_ok=True)

input_file = os.path.join(FINAL_DIR, '02_ceemdan_imfs_dataset.csv')

print("=" * 60)
print("  CEEMDAN-XGBoost Pipeline -- Phase 2: Sub-Model Training")
print("=" * 60)

# --- 2. LOAD CEEMDAN FEATURE MATRIX ---
try:
    df = pd.read_csv(input_file, index_col='Date', parse_dates=['Date'])
    print(f"[OK] CEEMDAN dataset loaded: {len(df)} rows x {len(df.columns)} cols")
except FileNotFoundError:
    print(f"[!]  Error: {input_file} not found.  Run 07_ceemdan_decomposition first.")
    exit()

# --- 3. IDENTIFY IMF COMPONENT COLUMNS ---
# The raw IMF columns are: IMF1, IMF2, ..., Residue  (no suffix)
imf_targets = [c for c in df.columns
               if (c.startswith('IMF') or c == 'Residue')
               and '_' not in c]

print(f"\nTarget components to model: {imf_targets}")

# --- 4. CHRONOLOGICAL TIMELINE SPLIT ---
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

# --- 5. TRAIN INDEPENDENT XGBOOST SUB-MODELS ---
print("\n" + "-" * 60)
print("  Training XGBoost sub-models for each IMF component...")
print("-" * 60)

for target in imf_targets:
    print(f"\n>> Training sub-model for [{target}]...")

    # Features for this component: its lags & rolling average
    feature_cols = [c for c in df.columns
                    if c.startswith(target + '_')]
    
    if not feature_cols:
        print(f"  [!]  No features found for {target}, skipping.")
        continue
    
    print(f"  Features: {feature_cols}")

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

    # Save the sub-model
    model_path = os.path.join(MODELS_DIR, f'ceemdan_xgb_{target.lower()}.json')
    model.save_model(model_path)
    print(f"  [OK] Saved -> {model_path}")

print("\n" + "=" * 60)
print("  ALL CEEMDAN SUB-MODELS TRAINED AND SAVED")
print("=" * 60)
