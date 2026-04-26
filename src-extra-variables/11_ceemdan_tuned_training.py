import os
import pandas as pd
import xgboost as xgb
import json
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CEEMDAN-XGBoost -- Phase 11: Tuned Sub-Model Training
# =============================================================================
# This script trains the XGBoost sub-models using the optimal, IMF-specific
# hyperparameters discovered in Phase 10.
# =============================================================================

def main():
    FINAL_DIR   = '../data-extra-variables/03_final/'
    MODELS_DIR  = '../models-extra-variables/'
    DATA_FILE   = os.path.join(FINAL_DIR, '02c_ceemdan_stationary_dataset.csv')
    PARAMS_FILE = os.path.join(MODELS_DIR, 'ceemdan_tuned_params.json')

    print("=" * 60)
    print("  CEEMDAN-XGBoost -- Phase 11: Tuned Training")
    print("=" * 60)

    # --- 1. LOAD DATA & PARAMS ---
    try:
        df = pd.read_csv(DATA_FILE, index_col='Date', parse_dates=['Date'])
    except FileNotFoundError:
        print(f"[!] Error: {DATA_FILE} not found.")
        exit()

    try:
        with open(PARAMS_FILE, 'r') as f:
            tuned_params = json.load(f)
    except FileNotFoundError:
        print(f"[!] Error: {PARAMS_FILE} not found. Run 10_ceemdan_tuning.py first.")
        exit()

    # --- 2. TIMELINE SPLIT ---
    TRAIN_END  = '2023-12-31'
    VAL_START  = '2024-01-01'
    VAL_END    = '2024-12-31'
    
    train = df[:TRAIN_END]
    val   = df[VAL_START:VAL_END]

    # --- 3. IDENTIFY TARGETS & FEATURES ---
    imf_targets = [c for c in df.columns if (c.startswith('IMF') or c == 'Residue') and '_' not in c]
    exog_features = [c for c in df.columns 
                     if not c.startswith('IMF') and c != 'Residue' and not c.startswith('Residue_')
                     and c not in ['Gold_Close', 'Gold_Close_LogReturn']]

    # --- 4. TRAIN TUNED MODELS ---
    print("\nTraining models with IMF-specific hyperparameters...")

    for target in imf_targets:
        print(f"\n>> Training {target}...")
        
        imf_own_features = [c for c in df.columns if c.startswith(target + '_')]
        feature_cols = imf_own_features + exog_features

        X_train, y_train = train[feature_cols], train[target]
        X_val, y_val     = val[feature_cols], val[target]

        # Get tuned params, fallback to defaults if missing
        params = tuned_params.get(target, {
            'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 5,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0
        })
        params['random_state'] = 42

        print(f"   Params: Depth={params['max_depth']}, LR={params['learning_rate']}, Trees={params['n_estimators']}")

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        model_path = os.path.join(MODELS_DIR, f'ceemdan_tuned_xgb_{target.lower()}.json')
        model.save_model(model_path)
        print(f"   [OK] Saved to: {model_path}")

    print("\n" + "=" * 60)
    print("  ALL TUNED CEEMDAN SUB-MODELS TRAINED AND SAVED")
    print("=" * 60)

if __name__ == '__main__':
    main()
