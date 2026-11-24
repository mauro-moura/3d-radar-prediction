
import numpy as np

import tensorflow as tf

import keras
from keras import Input, Model, ops
from keras import layers

class SeriesDecomposition(layers.Layer):
    def __init__(self, kernel_size=25, **kwargs):
        super(SeriesDecomposition, self).__init__(**kwargs)
        self.avg_pool = layers.AveragePooling1D(pool_size=kernel_size, strides=1, padding='same')

    def call(self, x):
        # Trend component (moving average)
        trend = self.avg_pool(x)
        # Seasonal component (residual)
        seasonal = x - trend
        return seasonal, trend

@keras.saving.register_keras_serializable(package="radar")
class DLinear(Model):
    def __init__(self, seq_len, pred_len, features, kernel_size=25, **kwargs):
        super(DLinear, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        self.kernel_size = kernel_size
        self.decomposition = SeriesDecomposition(kernel_size)

        # Linear mappings: map from seq_len -> pred_len per feature
        self.linear_seasonal = layers.Dense(pred_len)
        self.linear_trend = layers.Dense(pred_len)

    def call(self, x):
        """
        x: (batch, seq_len, features)
        """
        seasonal, trend = self.decomposition(x)

        # Transpose to (batch, features, seq_len)
        seasonal = tf.transpose(seasonal, [0, 2, 1])
        trend = tf.transpose(trend, [0, 2, 1])

        # Apply Dense across seq_len -> pred_len
        seasonal_out = self.linear_seasonal(seasonal)  # (batch, features, pred_len)
        trend_out = self.linear_trend(trend)

        out = seasonal_out + trend_out  # (batch, features, pred_len)

        # Return as (batch, pred_len, features)
        return tf.transpose(out, [0, 2, 1])

    def get_config(self):
        config = super().get_config()
        config.update({
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "features": self.features,
            "kernel_size": self.kernel_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            seq_len=config["seq_len"],
            pred_len=config["pred_len"],
            features=config["features"],
            **{k: v for k, v in config.items() if k not in ["seq_len", "pred_len", "features"]}
        )

def build_dlinear(seq_len, features, pred_len=1, kernel_size=25):
    inputs = Input(shape=(seq_len, features))
    # BUGFIX: before, kernel_size was passed positionally into the `features`
    # slot, so every DLinear trained with the default kernel_size=25.
    dlinear_model = DLinear(seq_len=seq_len, pred_len=pred_len, features=features, kernel_size=kernel_size)
    outputs = dlinear_model(inputs)
    model = Model(inputs, outputs, name="DLinear")
    model.compile(optimizer="adam", loss="mse")
    return model

# def build_dlinear(seq_len, features, dense_units=[128, 128]):
#     inputs = Input(shape=(seq_len, features))

#     # último passo temporal (time = -1)
#     last_t = layers.Lambda(lambda x: x[:, -1, :])(inputs)

#     # média no tempo (equivalente ao reduce_mean(axis=1))
#     mean_t = layers.GlobalAveragePooling1D()(inputs)

#     # projeções lineares (tendência e residual)
#     trend = layers.Dense(dense_units[0], use_bias=False)(last_t)
#     resid = layers.Dense(dense_units[1], use_bias=False)(mean_t)

#     outputs = layers.Add()([trend, resid])

#     model = Model(inputs, outputs, name="DLinearSimple")
#     model.compile(optimizer="adam", loss="mse")
#     return model

if __name__ == "__main__":
    # Exemplo de uso:
    # Exemplo: (timestamps, data_points, altitude, latitude, longitude)
    timestamps = 100
    data_points = 3
    altitude = 5
    latitude = 16
    longitude = 16

    arr = np.random.rand(timestamps, data_points, altitude, latitude, longitude)

    # Flatten spatial dims -> cada ponto espacial vira um feature
    features = data_points * altitude * latitude * longitude
    arr_flat = arr.reshape(timestamps, features)  # (timestamps, features)

    # Criar janelas (inputs e targets)
    seq_len = 10
    X, y = [], []

    for i in range(timestamps - seq_len):
        X.append(arr_flat[i:i+seq_len])   # (seq_len, features)
        y.append(arr_flat[i+seq_len])     # (features,)

    X = np.array(X)  # (samples, seq_len, features)
    y = np.array(y)  # (samples, features)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    seq_len, features = X.shape[1], X.shape[2]
    model = build_dlinear(seq_len, features)
    model.summary()

