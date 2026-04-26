import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CEEMDAN-XGBoost -- Phase 12: Tuned Evaluation
# =============================================================================
# Evaluates the performance of the individually tuned sub-models and compares
# it to the un-tuned stationary baseline.
# =============================================================================

def main():
    PROCESSED_DIR = '../data-extra-variables/02_processed/'
    FINAL_DIR     = '../data-extra-variables/03_final/'
    MODELS_DIR    = '../models-extra-variables/'
    PLOTS_DIR     = '../plots-extra-variables/ceemdan/'
    os.makedirs(PLOTS_DIR, exist_ok=True)

    TEST_START    = '2025-01-01'
    TRAIN_VAL_END = '2024-12-31'

    print("=" * 60)
    print("  CEEMDAN-XGBoost -- Phase 12: Tuned Evaluation")
    print("=" * 60)

    # --- 1. LOAD DATA ---
    df_master = pd.read_csv(os.path.join(PROCESSED_DIR, '01_master_metals_dataset.csv'), index_col='Date', parse_dates=['Date'])
    df_imf = pd.read_csv(os.path.join(FINAL_DIR, '02c_ceemdan_stationary_dataset.csv'), index_col='Date', parse_dates=['Date'])
    test = df_imf[TEST_START:]
    actual_prices = df_master.loc[test.index, 'Gold_Close']

    imf_targets = [c for c in df_imf.columns if (c.startswith('IMF') or c == 'Residue') and '_' not in c]
    exog_features = [c for c in df_imf.columns if not c.startswith('IMF') and c != 'Residue' and not c.startswith('Residue_') and c not in ['Gold_Close', 'Gold_Close_LogReturn']]

    # --- 2. PREDICT WITH TUNED MODELS ---
    print("\nGenerating predictions using Tuned Sub-Models...")
    tuned_preds = {}
    for target in imf_targets:
        model_path = os.path.join(MODELS_DIR, f'ceemdan_tuned_xgb_{target.lower()}.json')
        if not os.path.exists(model_path):
            print(f"  [!] Model missing: {model_path}")
            continue

        imf_own_features = [c for c in df_imf.columns if c.startswith(target + '_')]
        feature_cols = imf_own_features + exog_features

        model = xgb.XGBRegressor()
        model.load_model(model_path)
        tuned_preds[target] = pd.Series(model.predict(test[feature_cols]), index=test.index)

    predicted_log_returns = pd.DataFrame(tuned_preds).sum(axis=1)

    yesterday_price = df_master.loc[test.index, 'Gold_Close'].shift(1)
    yesterday_price.iloc[0] = df_master.loc[df_master[:TRAIN_VAL_END].index[-1], 'Gold_Close']
    tuned_price_preds = yesterday_price * np.exp(predicted_log_returns)

    rmse_tuned = root_mean_squared_error(actual_prices, tuned_price_preds)
    r2_tuned = r2_score(actual_prices, tuned_price_preds)

    # --- 3. COMPARE AGAINST UN-TUNED BASELINE ---
    untuned_preds = {}
    for target in imf_targets:
        model_path = os.path.join(MODELS_DIR, f'ceemdan_stationary_xgb_{target.lower()}.json')
        if os.path.exists(model_path):
            imf_own_features = [c for c in df_imf.columns if c.startswith(target + '_')]
            feature_cols = imf_own_features + exog_features
            m = xgb.XGBRegressor()
            m.load_model(model_path)
            untuned_preds[target] = pd.Series(m.predict(test[feature_cols]), index=test.index)
            
    untuned_log = pd.DataFrame(untuned_preds).sum(axis=1)
    untuned_price_preds = yesterday_price * np.exp(untuned_log)
    rmse_untuned = root_mean_squared_error(actual_prices, untuned_price_preds)

    print("\n" + "=" * 60)
    print("  TEST SET RESULTS (2025 -> Present)")
    print("=" * 60)
    print(f"  Un-Tuned CEEMDAN (Blanket Params) : ${rmse_untuned:.2f}")
    print(f"  Tuned CEEMDAN (IMF-Specific)      : ${rmse_tuned:.2f}")
    print(f"  R-Squared (Tuned)                 : {r2_tuned:.6f}")
    print("=" * 60)

    # --- 4. VISUALISATION ---
    plt.figure(figsize=(12, 6))
    models = ['Un-Tuned (Blanket)', 'Tuned (IMF-Specific)']
    scores = [rmse_untuned, rmse_tuned]
    colors = ['darkorange', 'forestgreen']
    
    bars = plt.bar(models, scores, color=colors, edgecolor='black', width=0.4)
    plt.title('Impact of IMF-Specific Hyperparameter Tuning', fontsize=14)
    plt.ylabel('RMSE (USD)', fontsize=12)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'${yval:.2f}', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'ceemdan_tuning_impact.png')
    plt.savefig(plot_path, dpi=150)
    plt.show()

    print(f"\n[OK] Plot saved to: {plot_path}")

if __name__ == '__main__':
    main()
