import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import warnings

warnings.filterwarnings("ignore")

print("Initiating Phase 7: Time-Series Hyperparameter Tuning (Architecture A Only)...\n")

# --- 1. CONFIGURATION & DIRECTORY SETUP ---
FINAL_DIR = '../data-extra-variables/03_final/'
DATA_FILE = os.path.join(FINAL_DIR, '01a_differencing_metals_dataset.csv')

# --- 2. DATA LOADING & STRICT TIMELINE SPLIT ---
df = pd.read_csv(DATA_FILE, index_col='Date', parse_dates=['Date'])

# We combine Train and Validation sets for Cross-Validation tuning.
# We strictly exclude the 2025 Test Set to prevent future data leakage.
TRAIN_VAL_END = '2024-12-31'
train_val_data = df[:TRAIN_VAL_END]

# --- 3. TARGET & LEAKAGE CONTROL (Mirroring your Training Script) ---
# Dropping raw prices and current-day log returns to isolate the T-1 predictors
drop_cols_a = [
    'Gold_Close', 'Silver_Close', 'DXY_Close', 'EGP_USD_Close', 
    'Gold_Close_LogReturn', 'Silver_Close_LogReturn', 'DXY_Close_LogReturn', 'EGP_USD_Close_LogReturn'
]

# Safe-drop to avoid KeyErrors
cols_to_drop = [c for c in drop_cols_a if c in train_val_data.columns]

X = train_val_data.drop(columns=cols_to_drop)
y = train_val_data['Gold_Close_LogReturn']

print(f"Total Features fed into the Grid Search: {len(X.columns)}")
print("Features being tuned:")
for col in X.columns:
    print(f" - {col}")

# --- 4. THE HYPERPARAMETER GRID ---
param_grid = {
    'n_estimators': [500, 1000, 1500],
    'learning_rate': [0.01, 0.05],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# --- 5. TIME-SERIES CROSS VALIDATION ---
tscv = TimeSeriesSplit(n_splits=5)
xgb_model = xgb.XGBRegressor(random_state=42)

print("\nCommencing Grid Search... (Training hundreds of models across chronological folds...)")
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

# --- 6. EXECUTION AND EXTRACTION ---
grid_search.fit(X, y)
best_params = grid_search.best_params_
best_score = -grid_search.best_score_ 

print("\n" + "="*50)
print("🏆 OPTIMAL HYPERPARAMETERS FOUND 🏆")
print("="*50)
for param, value in best_params.items():
    print(f"  --> {param}: {value}")

print(f"\nAverage Cross-Validation RMSE (Log Space): {best_score:.6f}")
print("="*50)

# --- 7. SAVE TO FILE ---
output_path = '../data-extra-variables/best_xgboost_gold_params.txt'
with open(output_path, 'w') as f:
    f.write("Best Hyperparameters for Gold Architecture A (Differencing + Extras):\n")
    for param, value in best_params.items():
        f.write(f"{param}: {value}\n")
print(f"\nSaved winning configuration to: {output_path}")