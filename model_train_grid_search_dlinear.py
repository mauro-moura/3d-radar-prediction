
import os
import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import KFold, ParameterGrid

from models.conv_lstm import build_model as ConvLSTMArch
from models.dlinear import build_dlinear as DLinearArch

cfg = {
    "data_path": "output_data/train_data.nc",
    "target_var": "dbzh",
    "architecture": "DLinear",  # "ConvLSTM" or "DLinear"
    "seq_len": 10,
    "pred_len": 1,
    "epochs": 1,
    "batch_size": 32,
    "output_path": "outputs/",
    "exp_name": "test_run"
}

def prepare_data(arr, seq_len, pred_len=1):
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

def build_model(params=None):
    model = None
    if cfg.get("architecture") == "ConvLSTM":
        model = ConvLSTMArch(X_train.shape[1:], filters=params["filters"])
    elif cfg.get("architecture") == "DLinear":
        model = DLinearArch(seq_len=cfg["seq_len"], features=features, pred_len=cfg["pred_len"])
    return model

if __name__ == "__main__":
    ds = xr.open_dataset(cfg.get("data_path"))
    arr = ds[cfg.get("target_var")].values

    param_grid = {
        "kernel_size": [15, 25],
        "batch_size": [16, 32],
        "epochs": [50, 100]
    }

    seed = 42
    grid = list(ParameterGrid(param_grid))
    print(f"Total combinations: {len(grid)}")
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    timestamps = arr.shape[0]
    data_points = 1
    altitude = arr.shape[-1]
    latitude = arr.shape[1]
    longitude = arr.shape[2]
    seq_len = cfg.get("seq_len")

    print(f"Selected model: {cfg.get('architecture')}")
    print(f"Params: cfg={cfg}")

    X, y = prepare_data(arr, cfg["seq_len"], cfg["pred_len"])
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    results = []
    model_id = 0
    for params in grid:
        print("Testing params:", params)
        fold_mse = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            seq_len, features = X_train.shape[1], X_train.shape[2]
            
            # Add callbacks if needed (e.g., EarlyStopping, ModelCheckpoint)
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
            ]
            if not os.path.exists(cfg.get("output_path")):
                os.makedirs(cfg.get("output_path"))

            model = build_model(params)
            
            model.optimizer.learning_rate = 1e-3
            
            # Train
            hist = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                callbacks=callbacks
            )

            model.save(os.path.join(cfg.get("output_path"), f"{cfg.get('architecture')}_{model_id}_{cfg.get('exp_name')}.h5"))
            np.save(os.path.join(cfg.get("output_path"), f"{cfg.get('architecture')}_{model_id}_hist_{cfg.get('exp_name')}.npy"), hist.history)
            model_id += 1

            # Evaluate on validation fold
            model_metric = model.evaluate(X_val, y_val)
            fold_mse.append(model_metric)
        
        results.append({"params": params, "mse": np.mean(fold_mse)})

    print("Grid search completed.")
    print("Results:", results)
    # Sort best results
    results = sorted(results, key=lambda x: x["mse"])
    print("Best params:", results[0])

    # Train final model with best params
    best_params = results[0]["params"]
    model = build_model(best_params)
    hist = model.fit(X, y, epochs=best_params["epochs"], batch_size=best_params["batch_size"], validation_split=0.15)
    model.save(os.path.join(cfg.get("output_path"), f"{cfg.get('architecture')}_final_model_{cfg.get('exp_name')}.h5"))
    np.save(os.path.join(cfg.get("output_path"), f"{cfg.get('architecture')}_final_model_hist_{cfg.get('exp_name')}.npy"), hist.history)
    print("Final model trained and saved.")
