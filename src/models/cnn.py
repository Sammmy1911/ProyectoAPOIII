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

    inputs = Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs) 

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