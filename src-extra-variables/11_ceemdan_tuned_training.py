import json
import os
import warnings

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge

from ceemdan_common import (
    FINAL_DIR,
    MODELS_DIR,
    TRAIN_END,
    VAL_END,
    VAL_START,
    feature_columns_for,
    natural_imf_targets,
    use_ridge_for_residue,
)

warnings.filterwarnings("ignore")


def main() -> None:
    DATA_FILE = os.path.join(FINAL_DIR, "02c_ceemdan_stationary_dataset.csv")
    PARAMS_FILE = os.path.join(MODELS_DIR, "ceemdan_tuned_params.json")

    print("=" * 60)
    print("  CEEMDAN-XGBoost — Phase 11: Tuned training (stationary)")
    print("=" * 60)

    try:
        df = pd.read_csv(DATA_FILE, index_col="Date", parse_dates=["Date"])
    except FileNotFoundError:
        print(f"[!] Missing {DATA_FILE}.")
        exit(1)

    try:
        with open(PARAMS_FILE, "r") as f:
            tuned_params = json.load(f)
    except FileNotFoundError:
        print(f"[!] Missing {PARAMS_FILE}. Run 10_ceemdan_tuning.py first.")
        exit(1)

    train = df[:TRAIN_END]
    val = df[VAL_START:VAL_END]

    ordered = natural_imf_targets(df.columns)
    imf_targets = list(ordered)

    print("\nTraining tuned stationary sub-models...")

    for target in imf_targets:
        feature_cols = feature_columns_for(df, target, "stationary", ordered)

        if target == "Residue" and use_ridge_for_residue():
            ridge = Ridge(alpha=1.0, random_state=42)
            ridge.fit(train[feature_cols], train[target])
            path = os.path.join(MODELS_DIR, "ceemdan_tuned_ridge_residue.joblib")
            joblib.dump(ridge, path)
            print(f"\n>> [{target}] Ridge -> {path}")
            continue

        params = tuned_params.get(target)
        if not params:
            params = {
                "n_estimators": 1000,
                "learning_rate": 0.05,
                "max_depth": 5,
                "min_child_weight": 1.0,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
            }
        params = dict(params)
        params["random_state"] = 42

        print(f"\n>> [{target}] depth={params.get('max_depth')} lr={params.get('learning_rate')}")

        model = xgb.XGBRegressor(**params)
        model.fit(
            train[feature_cols],
            train[target],
            eval_set=[(val[feature_cols], val[target])],
            verbose=False,
        )
        out_path = os.path.join(MODELS_DIR, f"ceemdan_tuned_xgb_{target.lower()}.json")
        model.save_model(out_path)
        print(f"   [OK] {out_path}")

    print("\n" + "=" * 60)
    print("  TUNED CEEMDAN SUB-MODELS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
