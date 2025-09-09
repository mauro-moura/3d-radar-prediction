
import numpy as np

def prepare_data(arr, seq_len, pred_len=1, cfg=None):
    timestamps = arr.shape[0]
    data_points = 1 # e.g., 1 variable/feature
    altitude = arr.shape[-1]
    latitude = arr.shape[1]
    longitude = arr.shape[2]

    arr = arr.reshape(timestamps, data_points, altitude, latitude, longitude)

    if cfg.get("architecture") == "DLinear":
        # Flatten spatial dims -> cada ponto espacial vira um feature
        features = data_points * altitude * latitude * longitude
        arr = arr.reshape(timestamps, features)  # (timestamps, features)

    X, y = [], []

    for i in range(timestamps - seq_len - pred_len + 1):
        seq_x = arr[i:i+seq_len]
        # seq_y = arr[i+seq_len:i+seq_len+pred_len]
        seq_y = arr[i+seq_len]

        if cfg.get("architecture") != "DLinear":
            seq_x = np.transpose(seq_x, (0, 2, 3, 4, 1))  # (seq_len, alt, lat, lon, channels)
            seq_y = np.transpose(seq_y, (1, 2, 3, 0))    # (alt, lat, lon, channels)
        
        X.append(seq_x)
        y.append(seq_y)

    X = np.array(X)
    y = np.array(y)

    return X, y

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
