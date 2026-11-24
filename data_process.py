
import os
import platform
import glob

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    system_name = platform.system()
    if system_name == "Linux":
        BASE_PATH = "/home/mauro/Downloads/Radar Rionowcast/processed_parts/"
    else:
        BASE_PATH = "D:\\Drives\\LNCC\\DEXL\\Radar Rionowcast\\processed_parts"

    all_files = glob.glob(os.path.join(BASE_PATH, "*.parquet"))
    OUTPUT_PATH = "./output_data/data_nc/raw/"
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    fill_na = False
    year = 2022

    # Group files by first 10 chars (YYYY-MM-DD)
    groups = []
    for f in all_files:
        date_str = os.path.basename(f)[:10]
        if date_str not in groups:
            groups.append(date_str)
    print(f"Found {len(groups)} groups")

    # group = groups[0]
    group_size = len(groups)
    for i, group in enumerate(groups):
        print(f"{i+1}/{group_size} Processing group: {group}")
        df = pd.DataFrame()

        for file in all_files:
            if group in file:
                _df = pd.read_parquet(file)
                df = pd.concat([df, _df], ignore_index=True)

        df.sort_values(by="time", inplace=True)
        df.reset_index(drop=True, inplace=True)

        altitude_interval = 500
        df["altitude"] = (df["altitude"] / altitude_interval).round() * altitude_interval
        df = (
            df.set_index("time")
            .groupby(["lat", "lon", "altitude"])
            .resample("10min")
            .agg({
                "dbzh": "mean"
            })
            .reset_index()
        )

        times = np.sort(df["time"].unique())
        lats = np.sort(df["lat"].unique())
        lons = np.sort(df["lon"].unique())
        altitudes = np.sort(df["altitude"].unique())

        data_array = np.full((len(times), len(lats), len(lons), len(altitudes)), np.nan)

        # Use pandas Categorical for vectorized indexing
        time_indices = pd.Categorical(df["time"], categories=times).codes
        lat_indices = pd.Categorical(df["lat"], categories=lats).codes  
        lon_indices = pd.Categorical(df["lon"], categories=lons).codes
        alt_indices = pd.Categorical(df["altitude"], categories=altitudes).codes

        data_array[time_indices, lat_indices, lon_indices, alt_indices] = df["dbzh"].values

        ds = xr.Dataset(
            {
                "dbzh": (("time", "lat", "lon", "altitude"), data_array)
            },
            coords={
                "time": times,
                "lat": lats,
                "lon": lons,
                "altitude": altitudes
            }
        )

        output_filename = f"data_{group}.nc"
        if fill_na:
            ds = ds.fillna(0)
            output_filename = f"data_{group}_filled.nc"

        ds.to_netcdf(os.path.join(OUTPUT_PATH, output_filename))
