import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error, r2_score

PROCESSED_DIR = '../data-extra-variables/02_processed/'
FINAL_DIR     = '../data-extra-variables/03_final/'
MODELS_DIR    = '../models-extra-variables/'

df_master = pd.read_csv(os.path.join(PROCESSED_DIR, '01_master_metals_dataset.csv'), index_col='Date', parse_dates=['Date'])
df_imf = pd.read_csv(os.path.join(FINAL_DIR, '02_ceemdan_imfs_dataset.csv'), index_col='Date', parse_dates=['Date'])

TRAIN_END = '2023-12-31'
VAL_START = '2024-01-01'
VAL_END = '2024-12-31'
TEST_START = '2025-01-01'

train_idx = df_imf[:TRAIN_END].index
val_idx = df_imf[VAL_START:VAL_END].index
test_idx = df_imf[TEST_START:].index

imf_targets = [c for c in df_imf.columns if (c.startswith('IMF') or c == 'Residue') and '_' not in c]

def evaluate_ceemdan(idx, stage_name):
    actual_prices = df_master.loc[idx, 'Gold_Close']
    X_stage = df_imf.loc[idx]
    
    component_preds = {}
    for target in imf_targets:
        model_path = os.path.join(MODELS_DIR, f'ceemdan_xgb_{target.lower()}.json')
        if not os.path.exists(model_path):
            continue
            
        feature_cols = [c for c in df_imf.columns if c.startswith(target + '_')]
        X = X_stage[feature_cols]
        
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        
        preds = model.predict(X)
        component_preds[target] = pd.Series(preds, index=idx)
        
    pred_sum = pd.DataFrame(component_preds).sum(axis=1)
    
    rmse = root_mean_squared_error(actual_prices, pred_sum)
    mape = mean_absolute_percentage_error(actual_prices, pred_sum)
    mae = mean_absolute_error(actual_prices, pred_sum)
    mse = mean_squared_error(actual_prices, pred_sum)
    r2 = r2_score(actual_prices, pred_sum)
    
    print(f"{stage_name:10s} - RMSE: ${rmse:.2f} | MAPE: {mape:.4f} | MAE: {mae:.2f} | MSE: {mse:.2f} | R2: {r2:.4f}")

print("=== CEEMDAN-XGBoost Model ===")
evaluate_ceemdan(train_idx, "Training")
evaluate_ceemdan(val_idx, "Validation")
evaluate_ceemdan(test_idx, "Testing")
