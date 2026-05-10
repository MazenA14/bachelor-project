import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

from ceemdan_common import (
    FINAL_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    TEST_START,
    TRAIN_VAL_END,
    feature_columns_for,
    natural_imf_targets,
    use_ridge_for_residue,
)

warnings.filterwarnings("ignore")


def main() -> None:
    PROCESSED_DIR = "../data-extra-variables/02_processed/"
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  CEEMDAN-XGBoost — Phase 12: Tuned evaluation")
    print("=" * 60)

    df_master = pd.read_csv(
        os.path.join(PROCESSED_DIR, "01_master_metals_dataset.csv"),
        index_col="Date",
        parse_dates=["Date"],
    )
    df_imf = pd.read_csv(
        os.path.join(FINAL_DIR, "02c_ceemdan_stationary_dataset.csv"),
        index_col="Date",
        parse_dates=["Date"],
    )
    test = df_imf[TEST_START:]
    actual_prices = df_master.loc[test.index, "Gold_Close"]
    last_val_date = df_master[:TRAIN_VAL_END].index[-1]

    ordered = natural_imf_targets(df_imf.columns)
    imf_targets = list(ordered)

    # --- Tuned predictions ---
    tuned_preds = {}
    for target in imf_targets:
        feat = feature_columns_for(df_imf, target, "stationary", ordered)

        if target == "Residue" and use_ridge_for_residue():
            rp = os.path.join(MODELS_DIR, "ceemdan_tuned_ridge_residue.joblib")
            if not os.path.exists(rp):
                print(f"  [!] Missing {rp}")
                continue
            ridge = joblib.load(rp)
            tuned_preds[target] = pd.Series(ridge.predict(test[feat]), index=test.index)
            continue

        mp = os.path.join(MODELS_DIR, f"ceemdan_tuned_xgb_{target.lower()}.json")
        if not os.path.exists(mp):
            print(f"  [!] Missing {mp}")
            continue
        m = xgb.XGBRegressor()
        m.load_model(mp)
        tuned_preds[target] = pd.Series(m.predict(test[feat]), index=test.index)

    predicted_log_returns = pd.DataFrame(tuned_preds).sum(axis=1)
    yesterday_price = df_master.loc[test.index, "Gold_Close"].shift(1)
    yesterday_price.iloc[0] = df_master.loc[last_val_date, "Gold_Close"]
    tuned_price_preds = yesterday_price * np.exp(predicted_log_returns)

    rmse_tuned = root_mean_squared_error(actual_prices, tuned_price_preds)
    mape_tuned = mean_absolute_percentage_error(actual_prices, tuned_price_preds)
    mae_tuned = mean_absolute_error(actual_prices, tuned_price_preds)
    mse_tuned = mean_squared_error(actual_prices, tuned_price_preds)
    r2_tuned = r2_score(actual_prices, tuned_price_preds)

    # --- Untuned baseline ---
    untuned_preds = {}
    for target in imf_targets:
        feat = feature_columns_for(df_imf, target, "stationary", ordered)
        if target == "Residue" and use_ridge_for_residue():
            rp = os.path.join(MODELS_DIR, "ceemdan_stationary_ridge_residue.joblib")
            if os.path.exists(rp):
                ridge = joblib.load(rp)
                untuned_preds[target] = pd.Series(ridge.predict(test[feat]), index=test.index)
            continue
        mp = os.path.join(MODELS_DIR, f"ceemdan_stationary_xgb_{target.lower()}.json")
        if os.path.exists(mp):
            m = xgb.XGBRegressor()
            m.load_model(mp)
            untuned_preds[target] = pd.Series(m.predict(test[feat]), index=test.index)

    untuned_log = pd.DataFrame(untuned_preds).sum(axis=1)
    untuned_price_preds = yesterday_price * np.exp(untuned_log)
    rmse_untuned = root_mean_squared_error(actual_prices, untuned_price_preds)
    mape_untuned = mean_absolute_percentage_error(actual_prices, untuned_price_preds)
    mae_untuned = mean_absolute_error(actual_prices, untuned_price_preds)
    mse_untuned = mean_squared_error(actual_prices, untuned_price_preds)
    r2_untuned = r2_score(actual_prices, untuned_price_preds)

    print("\n" + "=" * 60)
    print("  TEST SET RESULTS (2025 -> Present)")
    print("=" * 60)
    print(
        f"  Un-Tuned CEEMDAN : ${rmse_untuned:.2f} | MAPE: {mape_untuned:.4f} | "
        f"MAE: {mae_untuned:.2f} | MSE: {mse_untuned:.2f} | R2: {r2_untuned:.6f}"
    )
    print(
        f"  Tuned CEEMDAN    : ${rmse_tuned:.2f} | MAPE: {mape_tuned:.4f} | "
        f"MAE: {mae_tuned:.2f} | MSE: {mse_tuned:.2f} | R2: {r2_tuned:.6f}"
    )
    print("=" * 60)

    plt.figure(figsize=(12, 6))
    models = ["Un-Tuned (blanket params)", "Tuned (Optuna TPE)"]
    scores = [rmse_untuned, rmse_tuned]
    colors = ["darkorange", "forestgreen"]
    bars = plt.bar(models, scores, color=colors, edgecolor="black", width=0.4)
    plt.title("Impact of IMF-Specific Hyperparameter Tuning (Stationary Track)", fontsize=14)
    plt.ylabel("RMSE (USD)", fontsize=12)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.2, f"${yval:.2f}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "ceemdan_tuning_impact.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"\n[OK] Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
