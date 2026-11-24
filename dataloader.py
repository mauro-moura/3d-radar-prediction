
def prepare_data(arr, seq_len):
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

    for i in range(timestamps - seq_len):
        seq_x = arr[i:i+seq_len]
        seq_y = arr[i+seq_len]
        
        X.append(seq_x)
        y.append(seq_y)

    X = np.array(X)
    y = np.array(y)

    return X, y

ds = xr.open_dataset(cfg.get("data_path"))
arr = ds[cfg.get("target_var")].values

X, y = prepare_data(arr, seq_len)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
