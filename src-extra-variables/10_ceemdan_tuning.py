import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import json
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CEEMDAN-XGBoost -- IMF-Specific Hyperparameter Tuning
# =============================================================================
# Top-tier forecasting papers tune each IMF independently.
# High-frequency noise IMFs require shallow trees and high regularisation to
# avoid overfitting. Low-frequency trend IMFs require deeper trees.
# This script uses TimeSeries Cross Validation to find the optimal XGBoost
# parameters for EACH extracted stationary IMF.
# =============================================================================

def main():
    print("=" * 60)
    print("  CEEMDAN-XGBoost -- Phase 10: IMF-Specific Tuning")
    print("=" * 60)

    # --- 1. DIRECTORY SETUP & DATA LOADING ---
    FINAL_DIR = '../data-extra-variables/03_final/'
    MODELS_DIR = '../models-extra-variables/'
    DATA_FILE = os.path.join(FINAL_DIR, '02c_ceemdan_stationary_dataset.csv')
    PARAM_OUTPUT = os.path.join(MODELS_DIR, 'ceemdan_tuned_params.json')

    try:
        df = pd.read_csv(DATA_FILE, index_col='Date', parse_dates=['Date'])
        print(f"[OK] Stationary dataset loaded: {len(df)} rows")
    except FileNotFoundError:
        print(f"[!] Error: {DATA_FILE} not found. Run 07c first.")
        exit()

    # --- 2. TIMELINE SPLIT ---
    # Strictly exclude the 2025 Test Set from tuning
    TRAIN_VAL_END = '2024-12-31'
    train_val_data = df[:TRAIN_VAL_END]

    # --- 3. IDENTIFY TARGETS & SHARED FEATURES ---
    imf_targets = [c for c in df.columns if (c.startswith('IMF') or c == 'Residue') and '_' not in c]
    
    exog_features = [c for c in df.columns 
                     if not c.startswith('IMF') and c != 'Residue' and not c.startswith('Residue_')
                     and c not in ['Gold_Close', 'Gold_Close_LogReturn']]

    # --- 4. HYPERPARAMETER GRID ---
    # We use RandomizedSearch to cover a wide space efficiently
    param_grid = {
        'n_estimators': [200, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [2, 3, 5, 7],           # 2 for noise, 7 for complex trends
        'reg_alpha': [0, 0.1, 1.0, 5.0],     # L1 regularisation (Lasso)
        'reg_lambda': [0.1, 1.0, 5.0, 10.0], # L2 regularisation (Ridge)
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    tscv = TimeSeriesSplit(n_splits=3)
    best_params_dict = {}

    print(f"\nCommencing Time-Series Tuning for {len(imf_targets)} IMFs...\n")

    for target in imf_targets:
        print(f">> Tuning {target}...")
        
        # Features: target's own lags + exogenous
        imf_own_features = [c for c in df.columns if c.startswith(target + '_')]
        feature_cols = imf_own_features + exog_features

        X = train_val_data[feature_cols]
        y = train_val_data[target]

        xgb_model = xgb.XGBRegressor(random_state=42)

        # 20 iterations x 3 splits = 60 fits per IMF
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=20,
            scoring='neg_root_mean_squared_error',
            cv=tscv,
            verbose=0,
            n_jobs=-1,
            random_state=42
        )

        random_search.fit(X, y)
        best_p = random_search.best_params_
        best_score = -random_search.best_score_
        
        best_params_dict[target] = best_p
        
        print(f"   Best RMSE: {best_score:.6f}")
        print(f"   Depth: {best_p['max_depth']}, LR: {best_p['learning_rate']}, Trees: {best_p['n_estimators']}")
        print(f"   L1: {best_p['reg_alpha']}, L2: {best_p['reg_lambda']}\n")

    # --- 5. SAVE TUNED PARAMETERS ---
    with open(PARAM_OUTPUT, 'w') as f:
        json.dump(best_params_dict, f, indent=4)

    print("=" * 60)
    print(f"  TUNING COMPLETE. Parameters saved to: {PARAM_OUTPUT}")
    print("=" * 60)

if __name__ == '__main__':
    main()
