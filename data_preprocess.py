import os
import glob
import joblib

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":
    data_ext = "nc"
    window_size = 11
    scaler_name = 'scaler.save'
    is_train = True
    exec_name = 'train' if is_train else 'test'
    all_files = glob.glob(os.path.join('output_data', 'data_nc', exec_name, f'data_{data_ext}', f'*.{data_ext}'))
    all_files.sort()
    print(f"Found {len(all_files)} files with extension {data_ext}.")

    arr = None
    for data_name in all_files:
        if arr is None:
            arr = xr.open_dataarray(data_name)
            continue
        _arr = xr.open_dataarray(data_name)
        print(f"Before: arr.shape: {arr.shape}, _arr.shape: {_arr.shape}")
        arr = xr.concat((arr, _arr), dim='time')

    # Create windows of 11 timestamps where all values are different from zero
    valid_windows = []
    for i in range(len(arr) - window_size + 1):
        window = arr[i:i + window_size].time
        if np.all(window != 0) and np.all(np.diff(window) != 0):
            valid_windows.append(window)
    valid_windows = np.array(valid_windows)

    # Get all unique timestamps from valid_windows
    unique_timestamps = np.unique(valid_windows)

    final_arr = arr.sel(time=unique_timestamps)

    final_arr_flatten = final_arr.values.reshape(-1)

    if is_train:
        # Process the data to normalize between 0.1 and 0.9
        scaler = MinMaxScaler(feature_range=(0.1, 0.9))
        data_normalized = scaler.fit_transform(final_arr_flatten.reshape(-1, 1)).reshape(final_arr_flatten.shape)
        joblib.dump(scaler, scaler_name)
    else:
        scaler = joblib.load(scaler_name)
        data_normalized = scaler.transform(final_arr_flatten.reshape(-1, 1)).reshape(final_arr_flatten.shape)

    data_normalized = data_normalized.reshape(final_arr.shape)

    # Get back to a xr.Dataset
    data_normalized = xr.Dataset(
        {
            "dbzh": (("time", "lat", "lon", "altitude"), data_normalized)
        },
        coords=final_arr.coords
    )
    print(f"data_normalized['dbzh'].shape: {data_normalized['dbzh'].shape}")
    print(f"data_normalized['dbzh'].min(): {data_normalized['dbzh'].min().item()}, data_normalized['dbzh'].max(): {data_normalized['dbzh'].max().item()}")

    data_normalized.to_netcdf(
        os.path.join('output_data', f'normalized_data_{exec_name}.{data_ext}'),
        engine="netcdf4"
    )
