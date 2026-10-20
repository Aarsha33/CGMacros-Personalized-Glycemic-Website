from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "glucose_trajectory_personalized.csv"
)

MODEL_DIR = PROJECT_ROOT / "model"

MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("TRAINING GLUCOSE TRAJECTORY ANN")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print(f"Dataset rows: {len(df):,}")
print(f"Participants: {df['Participant_ID'].nunique()}")
print(
    f"Meals: "
    f"{df[['Participant_ID', 'Meal_Timestamp']].drop_duplicates().shape[0]}"
)


# ============================================================
# FEATURES
# ============================================================

numeric_features = [
    "HbA1c",
    "Fasting_Glucose",
    "Carbs",
    "Protein",
    "Meal_Hour_Sin",
    "Meal_Hour_Cos",
    "Time_After_Meal"
]

categorical_features = [
    "Meal_Type"
]

target = "Glucose"

X = df[
    numeric_features + categorical_features
].copy()

y = df[target].astype(float)

groups = df["Participant_ID"]


# ============================================================
# PARTICIPANT-AWARE TRAIN/VALIDATION SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)

X_train = X.iloc[train_idx].copy()
X_val = X.iloc[val_idx].copy()

y_train = y.iloc[train_idx].copy()
y_val = y.iloc[val_idx].copy()

train_groups = groups.iloc[train_idx]
val_groups = groups.iloc[val_idx]

print("\nTraining rows:", len(X_train))
print("Validation rows:", len(X_val))
print(
    "Training participants:",
    train_groups.nunique()
)
print(
    "Validation participants:",
    val_groups.nunique()
)
print(
    "Participant overlap:",
    len(
        set(train_groups)
        & set(val_groups)
    )
)


# ============================================================
# PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_val_processed = preprocessor.transform(
    X_val
)

X_train_processed = np.asarray(
    X_train_processed
).astype("float32")

X_val_processed = np.asarray(
    X_val_processed
).astype("float32")

print(
    "\nProcessed training shape:",
    X_train_processed.shape
)

print(
    "Processed validation shape:",
    X_val_processed.shape
)


# ============================================================
# BUILD ANN
# ============================================================

tf.random.set_seed(42)
np.random.seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(X_train_processed.shape[1],)
    ),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(
        16,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        1,
        activation="linear"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=["mae"]
)


# ============================================================
# EARLY STOPPING
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=30,
    restore_best_weights=True
)


# ============================================================
# TRAIN
# ============================================================

print("\nStarting ANN training...")

history = model.fit(
    X_train_processed,
    y_train.values,
    validation_data=(
        X_val_processed,
        y_val.values
    ),
    epochs=300,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

predictions = model.predict(
    X_val_processed,
    verbose=0
).flatten()

predictions = np.maximum(
    predictions,
    0
)


# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(
    y_val,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_val,
        predictions
    )
)

r2 = r2_score(
    y_val,
    predictions
)


print("\n" + "=" * 70)
print("GLUCOSE TRAJECTORY ANN RESULTS")
print("=" * 70)

print(
    f"MAE :  {mae:.2f} mg/dL"
)

print(
    f"RMSE:  {rmse:.2f} mg/dL"
)

print(
    f"R²  :  {r2:.4f}"
)

print(
    f"Epochs trained: {len(history.history['loss'])}"
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = (
    MODEL_DIR
    / "glucose_trajectory_ann.keras"
)

model.save(model_path)


# ============================================================
# SAVE PREPROCESSOR
# ============================================================

import joblib

preprocessor_path = (
    MODEL_DIR
    / "glucose_trajectory_preprocessor.joblib"
)

joblib.dump(
    preprocessor,
    preprocessor_path
)


# ============================================================
# SAVE FEATURE CONFIGURATION
# ============================================================

feature_config = {
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "target": target
}

with open(
    MODEL_DIR
    / "glucose_trajectory_feature_config.json",
    "w"
) as f:

    json.dump(
        feature_config,
        f,
        indent=2
    )


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "MAE_mg_dL": float(mae),
    "RMSE_mg_dL": float(rmse),
    "R2": float(r2),
    "epochs": len(history.history["loss"]),
    "train_rows": len(X_train),
    "validation_rows": len(X_val),
    "train_participants": int(
        train_groups.nunique()
    ),
    "validation_participants": int(
        val_groups.nunique()
    )
}

with open(
    MODEL_DIR
    / "glucose_trajectory_metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=2
    )


# ============================================================
# SAVE VALIDATION PREDICTIONS
# ============================================================

validation_results = df.iloc[
    val_idx
].copy()

validation_results[
    "Predicted_Glucose"
] = predictions

validation_results[
    "Residual"
] = (
    validation_results["Glucose"]
    - validation_results["Predicted_Glucose"]
)

validation_results.to_csv(
    MODEL_DIR
    / "glucose_trajectory_validation_predictions.csv",
    index=False
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    MODEL_DIR
    / "glucose_trajectory_training_history.csv",
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("MODEL FILES SAVED")
print("=" * 70)

print(model_path)
print(preprocessor_path)

print("\nTraining complete.")
print("=" * 70)