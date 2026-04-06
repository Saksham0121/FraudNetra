import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

from data_pipeline import prepare_dataset
from config import TRAIN_PATH, MODEL_PATH, THRESHOLD_PATH, FEATURE_COLUMNS_PATH, SCALER_PATH

print("Preparing dataset...")
X_full, y_full, feature_columns = prepare_dataset(TRAIN_PATH)

# save feature column structure First so api.py can use it
joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)

print("Splitting dataset into Train, Validation, and Test...")
# 1. First Split off the 15% Hold-out Test Set
X_temp, X_test, y_temp, y_test = train_test_split(
    X_full, y_full, test_size=0.15, stratify=y_full, random_state=42
)

# 2. Split remaining 85% into Train (70% total) and Val (15% total)
# 15 / 85 ≈ 0.1764 test_size for the temp dataset
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1764, stratify=y_temp, random_state=42
)

# 3. Filter exactly for only normal data for autoencoder training 
X_train_normal = X_train[y_train == 0]
X_val_normal = X_val[y_val == 0]

# sample only 200k rows for training to keep training times fast
if X_train_normal.shape[0] > 200000:
    X_train_normal = X_train_normal.sample(n=200000, random_state=42)

print("Training on normal transactions:", X_train_normal.shape)

print("Fitting scaler strictly on Training Set to prevent Data Leakage...")
scaler = StandardScaler()
# Fit and Transform ONLY the Training Normal Data
X_train_normal_scaled = scaler.fit_transform(X_train_normal)

# Transform (DO NOT FIT) the validation and test datasets based on the training mean/std
X_val_normal_scaled = scaler.transform(X_val_normal)
X_test_scaled = scaler.transform(X_test)

# Save scaler back out for inference
joblib.dump(scaler, SCALER_PATH)


# autoencoder architecture
input_dim = X_train_normal_scaled.shape[1]

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
    X_train_normal_scaled,
    X_train_normal_scaled,
    epochs=10,
    batch_size=256,
    validation_data=(X_val_normal_scaled, X_val_normal_scaled),
    shuffle=True
)

print("Calculating anomaly threshold using distinct Validation set...")

reconstructed_val = autoencoder.predict(X_val_normal_scaled)

errors_val = np.mean((X_val_normal_scaled - reconstructed_val) ** 2, axis=1)

# Statistical threshold calculation
threshold = np.mean(errors_val) + 3 * np.std(errors_val)

print("Threshold:", threshold)

joblib.dump(threshold, THRESHOLD_PATH)

print("=========================================")
print("EVALUATING ON UNSEEN HOLD-OUT TEST SET")
print("=========================================")

reconstructed_test = autoencoder.predict(X_test_scaled)
errors_test = np.mean((X_test_scaled - reconstructed_test) ** 2, axis=1)

y_pred = errors_test > threshold

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Genuine", "Fraud"]))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Saving model...")
autoencoder.save(MODEL_PATH)

print("Training complete.")