from tensorflow import keras
from tensorflow.keras import layers


def build_simple_cnn(input_shape: tuple[int, int, int], num_classes: int) -> keras.Model:
    """Create a small CNN for PlantVillage-style image classification."""
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Rescaling(1.0 / 255.0),
            layers.Conv2D(32, kernel_size=5, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=2),
            layers.Dropout(0.25),
            layers.Conv2D(64, kernel_size=5, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=2),
            layers.Dropout(0.25),
            layers.Conv2D(128, kernel_size=3, padding="same", activation="relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="simple_cnn",
    )
