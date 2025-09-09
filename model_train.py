
import os
import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from models.conv_lstm import build_model as ConvLSTMArch
from models.dlinear import build_dlinear as DLinearArch

cfg = {
    "data_path": "output_data/resized_data.nc",
    "target_var": "dbzh",
    "architecture": "DLinear",  # "ConvLSTM" or "DLinear"
    "seq_len": 10,
    "epochs": 1,
    "batch_size": 32,
    "output_path": "outputs/",
    "exp_name": "test_run"
}

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

if __name__ == "__main__":
    ds = xr.open_dataset(cfg.get("data_path"))
    arr = ds[cfg.get("target_var")].values

    timestamps = arr.shape[0]
    data_points = 1
    altitude = arr.shape[-1]
    latitude = arr.shape[1]
    longitude = arr.shape[2]
    seed = 42
    seq_len = cfg.get("seq_len")

    print(f"Selected model: {cfg.get('architecture')}")
    print(f"Params: cfg={cfg}")

    X, y = prepare_data(arr, seq_len)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    print("Train shapes:", X_train.shape, y_train.shape)
    print("Test shapes:", X_test.shape, y_test.shape)
    seq_len, features = X.shape[1], X.shape[2]

    # Add callbacks if needed (e.g., EarlyStopping, ModelCheckpoint)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    ]
    if not os.path.exists(cfg.get("output_path")):
        os.makedirs(cfg.get("output_path"))
    
    if cfg.get("architecture") == "ConvLSTM":
        model = ConvLSTMArch(X.shape[1:])
    elif cfg.get("architecture") == "DLinear":
        model = DLinearArch(seq_len, features)

    model.summary()
    
    # Treinar modelo
    hist = model.fit(
        X_train, y_train,
        epochs=cfg.get("epochs", 5), batch_size=cfg.get("batch_size", 32),
        validation_split=0.2,
        callbacks=callbacks
    )
    
    # Save hist and model
    model.save(os.path.join(cfg.get("output_path"), f"{cfg.get('architecture')}_model_{cfg.get('exp_name')}.h5"))
    np.save(os.path.join(cfg.get("output_path"), f"{cfg.get('architecture')}_hist_{cfg.get('exp_name')}.npy"), hist.history)

    pred = model.predict(X_test)
    pred = pred.reshape(y_test.shape)
    # Save preds
    np.save(os.path.join(cfg.get("output_path"), f"{cfg.get('architecture')}_preds_{cfg.get('exp_name')}.npy"), pred)

    test_mse = mse(y_test, pred)
    print("Test MSE:", test_mse)
