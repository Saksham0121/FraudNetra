import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from data_pipeline import prepare_dataset
from config import TRAIN_PATH, MODEL_PATH

print("Preparing dataset...")

X_train, y_train = prepare_dataset(TRAIN_PATH)

# train only on normal transactions
X_normal = X_train[y_train == 0]

# sample only 200k rows for training
X_normal = X_normal[np.random.choice(X_normal.shape[0], 200000, replace=False)]

print("Training on normal transactions:", X_normal.shape)


# autoencoder architecture
input_dim = X_normal.shape[1]

input_layer = Input(shape=(input_dim,))

encoded = Dense(64, activation="relu")(input_layer)
encoded = Dense(32, activation="relu")(encoded)
encoded = Dense(16, activation="relu")(encoded)

decoded = Dense(32, activation="relu")(encoded)
decoded = Dense(64, activation="relu")(decoded)
decoded = Dense(input_dim, activation="linear")(decoded)
autoencoder = Model(inputs=input_layer, outputs=decoded)

autoencoder.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse"
)

print("Starting training...")

autoencoder.fit(
    X_normal,
    X_normal,
    epochs=10,
    batch_size=256,
    validation_split=0.2,
    shuffle=True
)

print("Saving model...")
autoencoder.save(MODEL_PATH)
print("Training complete.")