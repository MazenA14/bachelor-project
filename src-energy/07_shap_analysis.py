# =============================================================================
# PIPELINE IDENTITY
#   Phase               : 07 — SHAP Explainability Analysis
#   Models Analyzed     : XGBoost A (Differencing) & XGBoost B (Detrending)
#                         Both "Engineered" and "No-Extras" variants
#   Outputs             : SHAP summary bar plots, beeswarm plots,
#                         dependence plots, force plots, feature ranking CSV
#   Saves to            : ../plots-energy/shap/
# =============================================================================
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')          # Non-interactive backend — saves to disk only
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

print("=" * 65)
print("  Phase 7: SHAP Explainability Analysis (Energy Pipeline)")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# 1. CONFIGURATION & DIRECTORY SETUP
# ─────────────────────────────────────────────────────────────────
PROCESSED_DIR = '../data-energy/02_processed/'
FINAL_DIR     = '../data-energy/03_final/'
MODELS_DIR    = '../models-energy/'
SHAP_DIR      = '../plots-energy/shap/'

os.makedirs(SHAP_DIR, exist_ok=True)

# Timeline split (must match training scripts exactly)
TRAIN_END  = '2023-12-31'
VAL_START  = '2024-01-01'
VAL_END    = '2024-12-31'
TEST_START = '2025-01-01'

def split_data(df):
    train = df[:TRAIN_END]
    val   = df[VAL_START:VAL_END]
    test  = df[TEST_START:]
    return train, val, test

# ─────────────────────────────────────────────────────────────────
# 2. DEFINE MODEL CONFIGURATIONS
# ─────────────────────────────────────────────────────────────────
# Each entry fully describes one model variant so we can loop cleanly.
MODEL_CONFIGS = {
    "XGBoost_A_Engineered": {
        "model_file": "xgboost_a.json",
        "data_file":  os.path.join(FINAL_DIR, '01a_engineered_differencing_energy_dataset.csv'),
        "target_col": "Brent_Crude_Close_LogReturn",
        "drop_cols": [
            'Brent_Crude_Close', 'Natural_Gas_Close', 'DXY_Close', 'VIX_Close', 'SP500_Close', 'EGP_USD_Close',
            'Brent_Crude_Close_LogReturn', 'Natural_Gas_Close_LogReturn', 'DXY_Close_LogReturn',
            'VIX_Close_LogReturn', 'SP500_Close_LogReturn', 'EGP_USD_Close_LogReturn',
            'US_10Yr_Yield_Diff',
            'Egypt_Inflation_YoY', 'CBE_Interest_Rate'
        ],
        "label": "Architecture A — Differencing (Engineered)"
    },
    "XGBoost_B_Engineered": {
        "model_file": "xgboost_b.json",
        "data_file":  os.path.join(FINAL_DIR, '01b_engineered_detrending_energy_dataset.csv'),
        "target_col": "Brent_Crude_Close_Residual",
        "drop_cols": [
            'Brent_Crude_Close', 'Natural_Gas_Close', 'DXY_Close', 'VIX_Close', 'SP500_Close', 'EGP_USD_Close',
            'Brent_Crude_Close_Trend', 'Brent_Crude_Close_Residual',
            'Natural_Gas_Close_Trend', 'Natural_Gas_Close_Residual',
            'DXY_Close_Trend', 'DXY_Close_Residual',
            'VIX_Close_Trend', 'VIX_Close_Residual',
            'SP500_Close_Trend', 'SP500_Close_Residual',
            'EGP_USD_Close_Trend', 'EGP_USD_Close_Residual',
            'Egypt_Inflation_YoY', 'CBE_Interest_Rate'
        ],
        "label": "Architecture B — Detrending (Engineered)"
    },
    "XGBoost_A_NoExtras": {
        "model_file": "xgboost_a_no_extras.json",
        "data_file":  os.path.join(FINAL_DIR, '01a_differencing_energy_dataset.csv'),
        "target_col": "Brent_Crude_Close_LogReturn",
        "drop_cols": [
            'Brent_Crude_Close', 'Natural_Gas_Close', 'DXY_Close', 'VIX_Close', 'SP500_Close', 'EGP_USD_Close',
            'Brent_Crude_Close_LogReturn', 'Natural_Gas_Close_LogReturn', 'DXY_Close_LogReturn',
            'VIX_Close_LogReturn', 'SP500_Close_LogReturn', 'EGP_USD_Close_LogReturn',
            'US_10Yr_Yield_Diff',
            'Egypt_Inflation_YoY', 'CBE_Interest_Rate'
        ],
        "label": "Architecture A — Differencing (No Extras)"
    },
    "XGBoost_B_NoExtras": {
        "model_file": "xgboost_b_no_extras.json",
        "data_file":  os.path.join(FINAL_DIR, '01b_detrending_energy_dataset.csv'),
        "target_col": "Brent_Crude_Close_Residual",
        "drop_cols": [
            'Brent_Crude_Close', 'Natural_Gas_Close', 'DXY_Close', 'VIX_Close', 'SP500_Close', 'EGP_USD_Close',
            'Brent_Crude_Close_Trend', 'Brent_Crude_Close_Residual',
            'Natural_Gas_Close_Trend', 'Natural_Gas_Close_Residual',
            'DXY_Close_Trend', 'DXY_Close_Residual',
            'VIX_Close_Trend', 'VIX_Close_Residual',
            'SP500_Close_Trend', 'SP500_Close_Residual',
            'EGP_USD_Close_Trend', 'EGP_USD_Close_Residual',
            'Egypt_Inflation_YoY', 'CBE_Interest_Rate'
        ],
        "label": "Architecture B — Detrending (No Extras)"
    }
}

# ─────────────────────────────────────────────────────────────────
# 3. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def load_model_and_data(config):
    """Load a saved XGBoost model and prepare its test-set features."""
    # Load model
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(MODELS_DIR, config["model_file"]))

    # Load dataset
    df = pd.read_csv(config["data_file"], index_col='Date', parse_dates=['Date'])

    # Split
    train, val, test = split_data(df)

    # Safe-drop: only drop columns that actually exist in this dataset
    cols_to_drop = [c for c in config["drop_cols"] if c in df.columns]
    X_test  = test.drop(columns=cols_to_drop)
    X_train = train.drop(columns=cols_to_drop)

    return model, X_train, X_test


def generate_shap_plots(model, X_background, X_explain, model_key, label):
    """
    Run TreeExplainer, then produce:
      1) Bar plot       — mean |SHAP| per feature (global importance)
      2) Beeswarm plot  — SHAP value distribution per feature
      3) Dependence     — top 3 features scatter plot
      4) Force plot     — single prediction explanation (first test day)
      5) Feature ranking CSV
    """
    prefix = os.path.join(SHAP_DIR, model_key)

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")

    # --- TreeExplainer (fast, exact for tree models) ---
    print("  Computing SHAP values via TreeExplainer...")
    explainer = shap.TreeExplainer(model, data=X_background)
    shap_values = explainer(X_explain)

    # ── Plot 1: Summary Bar ──────────────────────────────────────
    print("  Generating SHAP Summary Bar Plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_explain, plot_type="bar",
                      max_display=20, show=False)
    plt.title(f"SHAP Feature Importance (Mean |SHAP|)\n{label}", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{prefix}_summary_bar.png", dpi=200)
    plt.close()

    # ── Plot 2: Beeswarm ─────────────────────────────────────────
    print("  Generating SHAP Beeswarm Plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_explain, plot_type="dot",
                      max_display=20, show=False)
    plt.title(f"SHAP Beeswarm Plot\n{label}", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{prefix}_beeswarm.png", dpi=200)
    plt.close()

    # ── Plot 3: Dependence (Top 3 features) ──────────────────────
    print("  Generating Dependence Plots for Top 3 Features...")
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:3]
    top_features = [X_explain.columns[i] for i in top_indices]

    for rank, feat in enumerate(top_features, 1):
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feat, shap_values.values, X_explain,
                             show=False)
        plt.title(f"SHAP Dependence — {feat}\n{label}", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{prefix}_dependence_top{rank}_{feat}.png", dpi=200)
        plt.close()

    # ── Plot 4: Force Plot (First Test Day) ──────────────────────
    print("  Generating Force Plot (first test observation)...")
    force_html = shap.force_plot(
        explainer.expected_value,
        shap_values.values[0, :],
        X_explain.iloc[0, :],
        matplotlib=False
    )
    shap.save_html(f"{prefix}_force_plot.html", force_html)

    # ── Also produce a matplotlib version of the force plot ──────
    plt.figure(figsize=(20, 4))
    shap.force_plot(
        explainer.expected_value,
        shap_values.values[0, :],
        X_explain.iloc[0, :],
        matplotlib=True,
        show=False
    )
    plt.title(f"Force Plot — First Test Day ({X_explain.index[0].date()})\n{label}", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{prefix}_force_plot.png", dpi=200, bbox_inches='tight')
    plt.close()

    # ── Save Feature Ranking CSV ─────────────────────────────────
    print("  Saving Feature Importance Ranking...")
    ranking_df = pd.DataFrame({
        'Feature': X_explain.columns,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
    ranking_df.index += 1
    ranking_df.index.name = 'Rank'
    ranking_df.to_csv(f"{prefix}_feature_ranking.csv")

    print(f"\n  Top 10 Features by Mean |SHAP|:")
    print(ranking_df.head(10).to_string(index=True))

    return ranking_df


# ─────────────────────────────────────────────────────────────────
# 4. MAIN LOOP — ANALYZE ALL MODELS
# ─────────────────────────────────────────────────────────────────
all_rankings = {}

for model_key, config in MODEL_CONFIGS.items():
    model_path = os.path.join(MODELS_DIR, config["model_file"])

    # Gracefully skip models that haven't been trained yet
    if not os.path.exists(model_path):
        print(f"\n⚠️  Skipping {config['label']} — model file not found: {config['model_file']}")
        continue

    data_path = config["data_file"]
    if not os.path.exists(data_path):
        print(f"\n⚠️  Skipping {config['label']} — dataset not found: {data_path}")
        continue

    model, X_train, X_test = load_model_and_data(config)

    # Use a subsample of training data as SHAP background (for speed)
    # TreeExplainer uses this to compute E[f(x)] — 200 samples is standard
    bg_size = min(200, len(X_train))
    X_background = shap.sample(X_train, bg_size, random_state=42)

    ranking = generate_shap_plots(model, X_background, X_test,
                                  model_key, config["label"])
    all_rankings[model_key] = ranking

# ─────────────────────────────────────────────────────────────────
# 5. CROSS-MODEL COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────
if len(all_rankings) > 1:
    print(f"\n{'='*65}")
    print("  CROSS-MODEL SHAP COMPARISON — Top Features")
    print(f"{'='*65}")

    # Build a comparison DataFrame: each column is a model's mean |SHAP| per feature
    comparison_frames = []
    for model_key, ranking in all_rankings.items():
        renamed = ranking.set_index('Feature')['Mean_Abs_SHAP'].rename(model_key)
        comparison_frames.append(renamed)

    comparison_df = pd.concat(comparison_frames, axis=1).fillna(0)
    comparison_df['Avg_Across_Models'] = comparison_df.mean(axis=1)
    comparison_df = comparison_df.sort_values('Avg_Across_Models', ascending=False)

    comparison_path = os.path.join(SHAP_DIR, 'cross_model_comparison.csv')
    comparison_df.to_csv(comparison_path)
    print(comparison_df.head(15).to_string())
    print(f"\n  Saved to: {comparison_path}")

    # ── Grouped Bar Chart ────────────────────────────────────────
    top_n = min(12, len(comparison_df))
    plot_df = comparison_df.drop(columns='Avg_Across_Models').head(top_n)

    fig, ax = plt.subplots(figsize=(14, 8))
    plot_df.plot(kind='barh', ax=ax)
    ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
    ax.set_title('Cross-Model SHAP Feature Importance Comparison\n(Energy Pipeline — Brent Crude)', fontsize=14)
    ax.legend(title='Model Variant', fontsize=9, title_fontsize=10)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_DIR, 'cross_model_comparison_bar.png'), dpi=200)
    plt.close()

print(f"\n{'='*65}")
print(f"  SHAP ANALYSIS COMPLETE — All outputs saved to: {SHAP_DIR}")
print(f"{'='*65}")
