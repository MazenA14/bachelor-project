"""
Generates two ablation study figures:
  Figure 4.17 — % RMSE change (Global → Hybrid) per commodity
  Figure 4.18 — F-Score feature importance: Global-only vs Hybrid for Gold
"""
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
#  FIGURE 4.17  —  % ΔRMSE per commodity
# ──────────────────────────────────────────────
df = pd.read_csv('./ablation_global_vs_hybrid_table.csv')

pivot = df.pivot(index='Commodity', columns='Feature Set', values='RMSE')
pivot['delta_pct'] = (pivot['Hybrid'] - pivot['Global-only']) / pivot['Global-only'] * 100
pivot = pivot.sort_values('delta_pct')

commodities = pivot.index.tolist()
deltas      = pivot['delta_pct'].tolist()
colors      = ['#2ecc71' if d < 0 else '#e74c3c' for d in deltas]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(commodities, deltas, color=colors, edgecolor='white', height=0.55)

ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('∆RMSE (%)', fontsize=12)
ax.set_title(
    '% Change in RMSE (Global-only → Hybrid) per Commodity\n'
    'Negative = improvement; Positive = degradation',
    fontsize=12, pad=12,
)

for bar, val in zip(bars, deltas):
    x_pos = val + 0.005 if val >= 0 else val - 0.005
    ha    = 'left'       if val >= 0 else 'right'
    ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
            f'{val:+.3f}%', va='center', ha=ha, fontsize=9, fontweight='bold')

improve_patch = mpatches.Patch(color='#2ecc71', label='Improvement (Hybrid better)')
degrade_patch = mpatches.Patch(color='#e74c3c', label='Degradation (Hybrid worse)')
ax.legend(handles=[improve_patch, degrade_patch], fontsize=10, loc='lower right')

ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('./figure_4_17_delta_rmse.png', dpi=150, bbox_inches='tight')
print("Saved: figure_4_17_delta_rmse.png")
plt.show()


# ──────────────────────────────────────────────
#  FIGURE 4.18  —  F-Score importance: Gold
# ──────────────────────────────────────────────
print("\nTraining Gold models for F-Score importance...")

LOCAL_KEYWORDS = ['EGP', 'Inflation', 'CBE']
TRAIN_END      = '2023-12-31'
VAL_START      = '2024-01-01'
VAL_END        = '2024-12-31'
TEST_START     = '2025-01-01'

ROOT_EV   = './data-extra-variables/'
diff_ev   = pd.read_csv(ROOT_EV + '03_final/01a_differencing_metals_dataset.csv',
                        index_col='Date', parse_dates=['Date'])

tr = diff_ev[:TRAIN_END]
va = diff_ev[VAL_START:VAL_END]

# Same drop list as 05_ablation_global_vs_hybrid_table.py
GOLD_DROP = ['Gold_Close', 'Silver_Close', 'DXY_Close', 'EGP_USD_Close',
             'DXY_Close_LogReturn', 'EGP_USD_Close_LogReturn',
             'Gold_Close_LogReturn', 'Silver_Close_LogReturn']

y_tr = tr['Gold_Close_LogReturn']
y_va = va['Gold_Close_LogReturn']

X_tr_h = tr.drop(columns=[c for c in GOLD_DROP if c in tr.columns])
X_va_h = va.drop(columns=[c for c in GOLD_DROP if c in va.columns])

global_cols = [c for c in X_tr_h.columns if not any(kw in c for kw in LOCAL_KEYWORDS)]
X_tr_g = X_tr_h[global_cols]
X_va_g = X_va_h[global_cols]

EV_PARAMS = dict(n_estimators=500, learning_rate=0.01, max_depth=3,
                 subsample=0.8, colsample_bytree=0.8, random_state=42)

m_hybrid = xgb.XGBRegressor(**EV_PARAMS)
m_hybrid.fit(X_tr_h, y_tr, eval_set=[(X_va_h, y_va)], verbose=False)
print("  Hybrid model trained.")

m_global = xgb.XGBRegressor(**EV_PARAMS)
m_global.fit(X_tr_g, y_tr, eval_set=[(X_va_g, y_va)], verbose=False)
print("  Global model trained.")

# F-Score = number of times each feature is used in a split
fscore_h = pd.Series(m_hybrid.get_booster().get_fscore()).sort_values(ascending=False).head(10)
fscore_g = pd.Series(m_global.get_booster().get_fscore()).sort_values(ascending=False).head(10)

# Clean feature names while preserving LogRet / Level distinction
def clean(name):
    if '_Close_LogReturn' in name:
        base = name.replace('_Close_LogReturn', '')
        return base.replace('_', ' ') + ' (LogRet)'
    name = name.replace('_Close', '').replace('_LogReturn', ' (LogRet)')
    return name.replace('_', ' ')

fscore_h.index = [clean(n) for n in fscore_h.index]
fscore_g.index = [clean(n) for n in fscore_g.index]


def plot_fscore_panel(ax, fscore, color, title):
    """Plot a horizontal bar chart with safe label placement."""
    labels = fscore.index[::-1].tolist()   # lowest rank at bottom
    values = fscore.values[::-1].tolist()
    n      = len(labels)
    y_pos  = list(range(n))

    ax.barh(y_pos, values, color=color, edgecolor='white', height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_ylim(-0.6, n - 0.4)              # tight, no overflow

    max_val = max(values) if values else 1
    ax.set_xlim(0, max_val * 1.18)          # room for outside labels

    for i, val in enumerate(values):
        inside = val > max_val * 0.55       # wide enough to hold text inside
        if inside:
            ax.text(val * 0.97, i, str(int(val)),
                    va='center', ha='right', fontsize=9,
                    fontweight='bold', color='white')
        else:
            ax.text(val + max_val * 0.02, i, str(int(val)),
                    va='center', ha='left', fontsize=9,
                    fontweight='bold', color='black')

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('F-Score (split count)', fontsize=10)
    ax.grid(axis='x', alpha=0.3)


n_g = len(fscore_g)
n_h = len(fscore_h)
row_height = 0.7                             # inches per bar row
fig_h = max(5, max(n_g, n_h) * row_height + 2)

fig, (ax_g, ax_h) = plt.subplots(1, 2, figsize=(14, fig_h))
fig.suptitle(
    'Figure 4.18: F-Score Feature Importance — Gold (GC=F)\n'
    'Top 10 features: Global-only vs Hybrid XGBoost models',
    fontsize=12,
)

plot_fscore_panel(ax_g, fscore_g, 'steelblue', 'Global-only Model')
plot_fscore_panel(ax_h, fscore_h, 'darkblue',  'Hybrid Model (Global + Local Egyptian)')

plt.tight_layout()
plt.savefig('./figure_4_18_gold_fscore_importance.png', dpi=150, bbox_inches='tight')
print("Saved: figure_4_18_gold_fscore_importance.png")
plt.show()
