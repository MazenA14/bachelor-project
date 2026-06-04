import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

datasets = {
    "metals": os.path.join(BASE, "data-extra-variables", "02_processed", "01_master_metals_dataset.csv"),
    "energy": os.path.join(BASE, "data-energy", "02_processed", "01_master_energy_dataset.csv"),
    "crops":  os.path.join(BASE, "data-crops",  "02_processed", "01_master_crops_dataset.csv"),
}

commodity_map = {
    "metals": ["Gold_Close", "Silver_Close"],
    "energy": ["Brent_Crude_Close", "Natural_Gas_Close"],
    "crops":  ["Wheat_Close", "Corn_Close", "Sugar_Close"],
}

frames = []
for key, path in datasets.items():
    df = pd.read_csv(path, parse_dates=["Date"])
    cols = commodity_map[key]
    frames.append(df[cols])

df_all = pd.concat(frames, axis=1)

stats = df_all.describe().T
stats["Skewness"] = df_all.skew()
stats["Kurtosis"] = df_all.kurtosis()

final = stats[["mean", "std", "min", "max", "Skewness", "Kurtosis"]]
final.index = ["Gold", "Silver", "Brent Crude", "Natural Gas", "Wheat", "Corn", "Sugar"]
final.index.name = "Commodity"

print(final.round(4).to_string())
