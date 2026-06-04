import os
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

datasets = {
    "metals": os.path.join(BASE, "data-extra-variables", "02_processed", "01_master_metals_dataset.csv"),
    "energy": os.path.join(BASE, "data-energy",          "02_processed", "01_master_energy_dataset.csv"),
    "crops":  os.path.join(BASE, "data-crops",           "02_processed", "01_master_crops_dataset.csv"),
}

commodity_map = {
    "metals": ["Gold_Close", "Silver_Close"],
    "energy": ["Brent_Crude_Close", "Natural_Gas_Close"],
    "crops":  ["Wheat_Close", "Corn_Close", "Sugar_Close"],
}

labels = {
    "Gold_Close":        "Gold",
    "Silver_Close":      "Silver",
    "Brent_Crude_Close": "Brent Crude",
    "Natural_Gas_Close": "Natural Gas",
    "Wheat_Close":       "Wheat",
    "Corn_Close":        "Corn",
    "Sugar_Close":       "Sugar",
}


def adf(series):
    clean = series.dropna()
    result = adfuller(clean)
    return result[0], result[1]


def arch_a(series):
    """Arch A: log returns (differencing)."""
    return np.log(series / series.shift(1))


def arch_b(series):
    """Arch B: linear detrend residuals."""
    time_index = np.arange(len(series)).reshape(-1, 1)
    y = series.values.reshape(-1, 1)
    lr = LinearRegression().fit(time_index, y)
    residuals = y.flatten() - lr.predict(time_index).flatten()
    return pd.Series(residuals, index=series.index)


rows = []
for key, path in datasets.items():
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    for col in commodity_map[key]:
        raw        = df[col]
        transformed_a = arch_a(raw)
        transformed_b = arch_b(raw)

        stat_raw, p_raw   = adf(raw)
        stat_a,   p_a     = adf(transformed_a)
        stat_b,   p_b     = adf(transformed_b)

        rows.append({
            "Commodity":          labels[col],
            "Raw (ADF Stat)":     round(stat_raw, 4),
            "Raw (p-value)":      round(p_raw, 4),
            "Arch A (ADF Stat)":  round(stat_a, 4),
            "Arch A (p-value)":   round(p_a, 4),
            "Arch B (ADF Stat)":  round(stat_b, 4),
            "Arch B (p-value)":   round(p_b, 4),
        })

result_df = pd.DataFrame(rows).set_index("Commodity")
print(result_df.to_string())
