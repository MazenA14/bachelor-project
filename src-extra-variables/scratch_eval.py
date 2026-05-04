import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

print("Initializing cross-metal multi-stage evaluation...\n")

ROOT_DIR = '../data-extra-variables/'
MASTER_FILE = os.path.join(ROOT_DIR, '02_processed', '01_master_metals_dataset.csv')
DIFF_FILE = os.path.join(ROOT_DIR, '03_final', '01a_differencing_metals_dataset.csv')

df_master = pd.read_csv(MASTER_FILE, index_col='Date', parse_dates=['Date'])
df_diff = pd.read_csv(DIFF_FILE, index_col='Date', parse_dates=['Date'])

TRAIN_END = '2023-12-31'
VAL_START = '2024-01-01'
VAL_END = '2024-12-31'
TEST_START = '2025-01-01'

train = df_diff[:TRAIN_END]
val = df_diff[VAL_START:VAL_END]
test = df_diff[TEST_START:]

def evaluate_stage(model, X, target_metal):
    preds = model.predict(X)
    yesterday = df_master.loc[X.index, f'{target_metal}_Close'].shift(1)
    
    # Fill the first day using the day before the test set begins in df_master
    first_idx = df_master.index.get_loc(X.index[0])
    yesterday.iloc[0] = df_master.iloc[first_idx - 1][f'{target_metal}_Close']
    
    price_pred = yesterday * np.exp(preds)
    actual = df_master.loc[X.index, f'{target_metal}_Close']
    
    rmse = root_mean_squared_error(actual, price_pred)
    mape = mean_absolute_percentage_error(actual, price_pred)
    mae = mean_absolute_error(actual, price_pred)
    mse = mean_squared_error(actual, price_pred)
    r2 = r2_score(actual, price_pred)
    
    return rmse, mape, mae, mse, r2

def print_metrics(model, X_train, X_val, X_test, title, target_metal):
    print(f"=== {title} ===")
    for name, X in [("Training", X_train), ("Validation", X_val), ("Testing", X_test)]:
        rmse, mape, mae, mse, r2 = evaluate_stage(model, X, target_metal)
        print(f"{name:10s} - RMSE: ${rmse:.2f} | MAPE: {mape:.4f} | MAE: {mae:.2f} | MSE: {mse:.2f} | R2: {r2:.4f}")
    print()

# ==========================================
# PART A: GOLD
# ==========================================
drop_cols_gold_full = ['Gold_Close', 'Silver_Close', 'DXY_Close', 'EGP_USD_Close',
                       'Gold_Close_LogReturn', 'DXY_Close_LogReturn', 'EGP_USD_Close_LogReturn']
cols_to_drop_gold_full = [c for c in drop_cols_gold_full if c in df_diff.columns]

y_train_gold = train['Gold_Close_LogReturn']
y_val_gold   = val['Gold_Close_LogReturn']

X_train_full_gold = train.drop(columns=cols_to_drop_gold_full)
X_val_full_gold   = val.drop(columns=cols_to_drop_gold_full)
X_test_full_gold  = test.drop(columns=cols_to_drop_gold_full)

silver_keywords = ['Silver', 'Ratio']
gold_only_cols = [c for c in X_train_full_gold.columns if not any(kw in c for kw in silver_keywords)]

X_train_gold_only = X_train_full_gold[gold_only_cols]
X_val_gold_only   = X_val_full_gold[gold_only_cols]
X_test_gold_only  = X_test_full_gold[gold_only_cols]

xgb_gold_full = xgb.XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_gold_full.fit(X_train_full_gold, y_train_gold, eval_set=[(X_val_full_gold, y_val_gold)], verbose=False)

xgb_gold_only = xgb.XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_gold_only.fit(X_train_gold_only, y_train_gold, eval_set=[(X_val_gold_only, y_val_gold)], verbose=False)

print_metrics(xgb_gold_full, X_train_full_gold, X_val_full_gold, X_test_full_gold, "Gold Full Model (with Silver)", "Gold")
print_metrics(xgb_gold_only, X_train_gold_only, X_val_gold_only, X_test_gold_only, "Gold-Only Model (No Silver)", "Gold")

# ==========================================
# PART B: SILVER
# ==========================================
drop_cols_silver_full = ['Gold_Close', 'Silver_Close', 'DXY_Close', 'EGP_USD_Close',
                         'Silver_Close_LogReturn', 'DXY_Close_LogReturn', 'EGP_USD_Close_LogReturn']
cols_to_drop_silver_full = [c for c in drop_cols_silver_full if c in df_diff.columns]

y_train_silver = train['Silver_Close_LogReturn']
y_val_silver   = val['Silver_Close_LogReturn']

X_train_full_silver = train.drop(columns=cols_to_drop_silver_full)
X_val_full_silver   = val.drop(columns=cols_to_drop_silver_full)
X_test_full_silver  = test.drop(columns=cols_to_drop_silver_full)

gold_keywords = ['Gold', 'Ratio']
silver_only_cols = [c for c in X_train_full_silver.columns if not any(kw in c for kw in gold_keywords)]

X_train_silver_only = X_train_full_silver[silver_only_cols]
X_val_silver_only   = X_val_full_silver[silver_only_cols]
X_test_silver_only  = X_test_full_silver[silver_only_cols]

xgb_silver_full = xgb.XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_silver_full.fit(X_train_full_silver, y_train_silver, eval_set=[(X_val_full_silver, y_val_silver)], verbose=False)

xgb_silver_only = xgb.XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_silver_only.fit(X_train_silver_only, y_train_silver, eval_set=[(X_val_silver_only, y_val_silver)], verbose=False)

print_metrics(xgb_silver_full, X_train_full_silver, X_val_full_silver, X_test_full_silver, "Silver Full Model (with Gold Features)", "Silver")
print_metrics(xgb_silver_only, X_train_silver_only, X_val_silver_only, X_test_silver_only, "Silver-Only Model (No Gold Features)", "Silver")
