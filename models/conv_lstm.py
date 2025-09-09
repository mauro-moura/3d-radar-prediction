
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM3D, BatchNormalization, Conv3D

def build_model(input_shape=(None, None, None, None, 1), filters=[16, 8]):
    time, depth, height, width, channels = input_shape

    model = Sequential([
        ConvLSTM3D(
            filters=filters[0],
            kernel_size=(3,3,3),
            activation="tanh",
            padding="same",
            return_sequences=True,
            dropout=0.2,
            input_shape=(time, depth, height, width, channels)
        ),
        BatchNormalization(),
        ConvLSTM3D(
            filters=filters[1],
            kernel_size=(3,3,3),
            activation="tanh",
            padding="same",
            return_sequences=False,
            dropout=0.2
        ),
        BatchNormalization(),
        # Final prediction: one frame (depth, height, width, channels)
        Conv3D(
            filters=channels,
            kernel_size=(3,3,3),
            activation="linear",
            padding="same"
        )
    ])

    model.compile(optimizer="adam", loss="mse")
    
    return model
