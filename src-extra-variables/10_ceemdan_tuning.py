"""
CEEMDAN stationary track — IMF-specific hyperparameter search with Optuna (TPE).

Focus: learning_rate, max_depth, min_child_weight (plus supporting XGBoost knobs).
Residue is excluded (Ridge handles the slow trend in 08c / 11).

Requires: optuna, xgboost, scikit-learn
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from ceemdan_common import (
    FINAL_DIR,
    MODELS_DIR,
    TRAIN_VAL_END,
    feature_columns_for,
    natural_imf_targets,
    use_ridge_for_residue,
)

warnings.filterwarnings("ignore")

N_TRIALS_PER_IMF = int(os.environ.get("OPTUNA_TRIALS_PER_IMF", "40"))


def main() -> None:
    print("=" * 60)
    print("  CEEMDAN-XGBoost — Phase 10: Optuna TPE tuning (stationary track)")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)
    data_file = os.path.join(FINAL_DIR, "02c_ceemdan_stationary_dataset.csv")
    param_output = os.path.join(MODELS_DIR, "ceemdan_tuned_params.json")

    try:
        df = pd.read_csv(data_file, index_col="Date", parse_dates=["Date"])
        print(f"[OK] Stationary dataset loaded: {len(df)} rows")
    except FileNotFoundError:
        print(f"[!] Missing {data_file}. Run 07_ceemdan_decomposition.py first.")
        sys.exit(1)

    train_val = df[:TRAIN_VAL_END]
    ordered = natural_imf_targets(df.columns)
    imf_targets = list(ordered)

    best_params_dict: dict = {}

    for target in imf_targets:
        if target == "Residue" and use_ridge_for_residue():
            print(f"\n>> Skipping Optuna for [{target}] (Ridge residue model).")
            continue

        feature_cols = feature_columns_for(df, target, "stationary", ordered)
        X_tv = train_val[feature_cols].values
        y_tv = train_val[target].values

        tscv = TimeSeriesSplit(n_splits=3)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 400, 1200, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 9),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 12.0),
                "random_state": 42,
                "n_jobs": -1,
            }

            fold_rmses = []
            for train_idx, val_idx in tscv.split(X_tv):
                X_tr, X_va = X_tv[train_idx], X_tv[val_idx]
                y_tr, y_va = y_tv[train_idx], y_tv[val_idx]
                model = xgb.XGBRegressor(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                pred = model.predict(X_va)
                fold_rmses.append(float(np.sqrt(mean_squared_error(y_va, pred))))

            return float(np.mean(fold_rmses))

        print(f"\n>> Optuna TPE for [{target}] ({len(feature_cols)} features, {N_TRIALS_PER_IMF} trials)...")
        sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=N_TRIALS_PER_IMF, show_progress_bar=False)

        bp = study.best_params
        best_params_dict[target] = {
            "n_estimators": int(bp["n_estimators"]),
            "learning_rate": float(bp["learning_rate"]),
            "max_depth": int(bp["max_depth"]),
            "min_child_weight": float(bp["min_child_weight"]),
            "subsample": float(bp["subsample"]),
            "colsample_bytree": float(bp["colsample_bytree"]),
            "reg_alpha": float(bp["reg_alpha"]),
            "reg_lambda": float(bp["reg_lambda"]),
        }
        print(f"   Best mean CV RMSE: {study.best_value:.6f}")
        print(
            f"   depth={best_params_dict[target]['max_depth']}, "
            f"lr={best_params_dict[target]['learning_rate']:.4f}, "
            f"mcw={best_params_dict[target]['min_child_weight']:.2f}"
        )

    with open(param_output, "w") as f:
        json.dump(best_params_dict, f, indent=4)

    print("\n" + "=" * 60)
    print(f"  TUNING COMPLETE. Parameters saved to: {param_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
