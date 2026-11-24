"""CNN model builder for fruit quality classification."""
from tensorflow.keras import Input, layers, models, optimizers


def build_cnn_model(
    input_shape=(224, 224, 3),
    num_classes=3,
    filters=(32, 64, 128),
    dense_units=128,
    dropout=0.5,
    learning_rate=1e-3,
    augment=True,
):
    """
    Construye una CNN configurable para clasificación de calidad de frutas.

    Args:
        input_shape: (alto, ancho, 3). Debe coincidir con CNN_IMG_SIZE de config.py (224,224).
        num_classes: 3 (Bad, Regular, Good).
        filters: tupla de filtros por bloque convolucional.
        dense_units: unidades en la capa densa antes de salida.
        dropout: tasa de dropout.
        learning_rate: learning rate del optimizador Adam.
        augment: si True, añade capas de data augmentation (solo train).

    Returns:
        Modelo Keras compilado.
    """
    inputs = Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)  # normalización interna [0,1]

    if augment:  # solo se activan durante entrenamiento
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomRotation(0.05)(x)
        x = layers.RandomZoom(0.1)(x)

    for f in filters:
        x = layers.Conv2D(f, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="Custom_CNN")
    model.compile(
        optimizer=optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model