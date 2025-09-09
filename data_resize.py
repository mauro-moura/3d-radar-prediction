import os
import xarray as xr
import numpy as np

if __name__ == "__main__":
    is_train = True
    exec_name = 'train' if is_train else 'test'
    DATA_PATH = os.path.join("output_data", f"normalized_data_{exec_name}.nc")

    # Open as Dataset
    ds = xr.open_dataset(DATA_PATH)
    # Select specific altitudes given by specialists
    ds = ds.sel(altitude=[500, 1000, 1500, 3000, 5000, 7500])

    # Extract the "dbzh" DataArray
    arr = ds["dbzh"]
    print(f"arr.shape: {arr.shape}")  # (time, lat, lon, altitude)

    # Ensure lat/lon sorted for interpolation
    if not np.all(np.diff(arr.lat.values) > 0):
        arr = arr.sortby("lat")
    if not np.all(np.diff(arr.lon.values) > 0):
        arr = arr.sortby("lon")

    # Resize only lat/lon to 32x32
    target_shape = (32, 32)
    resized_arr = arr.interp(
        lat=np.linspace(arr.lat.min().item(), arr.lat.max().item(), target_shape[0]),
        lon=np.linspace(arr.lon.min().item(), arr.lon.max().item(), target_shape[1]),
        method="linear"
    )

    print(f"resized_arr.shape: {resized_arr.shape}")

    # Save back as Dataset with the same var name
    resized_ds = xr.Dataset({"dbzh": resized_arr})
    out_path = os.path.join("output_data", f"resized_data_{exec_name}.nc")
    resized_ds.to_netcdf(out_path)
