import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from sklearn.linear_model import LinearRegression

BASE = r"C:\Users\Mazen\OneDrive - GUC\GUC\Years\Year 4\Semester 8\bachelor-project"

nat_gas = pd.read_csv(os.path.join(BASE, "data-energy", "01_raw", "02_yf_natural_gas.csv"), parse_dates=["Date"], index_col="Date")
brent   = pd.read_csv(os.path.join(BASE, "data-energy", "01_raw", "01_yf_brent_crude.csv"), parse_dates=["Date"], index_col="Date")
corn    = pd.read_csv(os.path.join(BASE, "data-crops",  "01_raw", "02_yf_corn.csv"),         parse_dates=["Date"], index_col="Date")
global_data = pd.read_csv(os.path.join(BASE, "data-extra-variables", "01_raw", "01_yf_global_data.csv"), parse_dates=["Date"], index_col="Date")
gold = global_data[["Gold_Close"]].rename(columns={"Gold_Close": "Gold_Close"})

TRAIN_END = "2023-12-31"
VAL_END   = "2024-12-31"

commodities = [
    (nat_gas, "Natural_Gas_Close", "Natural Gas",  "royalblue",   "USD/MMBtu"),
    (brent,   "Brent_Crude_Close",  "Brent Crude",   "darkorange",  "USD/Barrel"),
    (corn,    "Corn_Close",        "Corn",          "goldenrod",   "USX/Bushel"),
    (gold,    "Gold_Close",        "Gold",          "crimson",     "USD/oz"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()

for ax, (df, col, label, color, unit) in zip(axes, commodities):
    series = df[col]

    train = series[:TRAIN_END]
    val   = series[TRAIN_END:VAL_END]
    test  = series[VAL_END:]

    ax.plot(train.index, train.values, color=color, lw=1.2, label="Training")
    ax.plot(val.index,   val.values,   color=color, lw=1.2, linestyle="--", label="Validation")
    ax.plot(test.index,  test.values,  color=color, lw=1.2, linestyle=":",  label="Test")

    # OLS trend line (Architecture B): fitted on integer time index across full dataset
    valid = series.dropna()
    time_idx = np.arange(len(valid)).reshape(-1, 1)
    lr = LinearRegression().fit(time_idx, valid.values)
    trend = lr.predict(time_idx)
    ax.plot(valid.index, trend, color="black", lw=1.5, linestyle="-", label="OLS Trend")

    ax.axvline(pd.Timestamp(TRAIN_END), color="gray", lw=0.8, linestyle="--")
    ax.axvline(pd.Timestamp(VAL_END),   color="gray", lw=0.8, linestyle=":")

    ax.set_title(label, fontsize=13, fontweight="bold")
    ax.set_ylabel(unit, fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

fig.suptitle("Commodity Prices (2014–2026)", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()

out_path = os.path.join(BASE, "commodity_prices.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.show()
