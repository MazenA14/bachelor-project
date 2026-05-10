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

input_file = os.path.join(FINAL_DIR, "02b_ceemdan_enriched_dataset.csv")


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 60)
    print("  CEEMDAN-XGBoost — Phase 2b: Sub-model training (enriched)")
    print("=" * 60)

    try:
        df = pd.read_csv(input_file, index_col="Date", parse_dates=["Date"])
        print(f"[OK] Enriched dataset loaded: {len(df)} rows x {len(df.columns)} cols")
    except FileNotFoundError:
        print(f"[!] Missing {input_file}. Run 07_ceemdan_decomposition.py first.")
        exit(1)

    ordered = natural_imf_targets(df.columns)
    imf_targets = list(ordered)

    train = df[:TRAIN_END]
    val = df[VAL_START:VAL_END]

    print(f"\n  Train : {len(train)} rows")
    print(f"  Val   : {len(val)} rows")

    print("\n" + "-" * 60)
    print("  Training enriched sub-models (band-specific macro exposure)")
    print("-" * 60)

    for target in imf_targets:
        feature_cols = feature_columns_for(df, target, "enriched", ordered)
        if not feature_cols:
            print(f"\n>> [{target}] skipped — no features.")
            continue

        print(f"\n>> [{target}] features={len(feature_cols)}")

        if target == "Residue" and use_ridge_for_residue():
            ridge = Ridge(alpha=1.0, random_state=42)
            ridge.fit(train[feature_cols], train[target])
            path = os.path.join(MODELS_DIR, "ceemdan_enriched_ridge_residue.joblib")
            joblib.dump(ridge, path)
            print(f"  [OK] Ridge residue saved -> {path}")
            continue

        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            early_stopping_rounds=50,
        )
        model.fit(train[feature_cols], train[target], eval_set=[(val[feature_cols], val[target])], verbose=False)
        out_path = os.path.join(MODELS_DIR, f"ceemdan_enriched_xgb_{target.lower()}.json")
        model.save_model(out_path)
        print(f"  [OK] Saved -> {out_path}")

    print("\n" + "=" * 60)
    print("  ENRICHED CEEMDAN SUB-MODELS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
