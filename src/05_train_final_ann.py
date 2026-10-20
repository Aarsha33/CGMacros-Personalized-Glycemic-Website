# ============================================================
# 05_train_final_ann.py
# Final ANN Training - CGMacros
# ============================================================

import os
import json
import random
import warnings
import joblib

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. PATHS
# ============================================================

DATA_PATH = "data/meal_personalized_dataset.csv"
OUTPUT_DIR = "model"

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 70)
print("CGMacros FINAL ANN TRAINING")
print("=" * 70)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Rows: {len(df):,}")
print(f"Participants: {df['Participant_ID'].nunique()}")


# ============================================================
# 4. CREATE MEAL TIME FEATURES
# ============================================================

print("\nCreating meal-time features...")

df["Meal_Timestamp"] = pd.to_datetime(
    df["Meal_Timestamp"],
    errors="coerce"
)

df["Meal_Hour"] = (
    df["Meal_Timestamp"].dt.hour
    + df["Meal_Timestamp"].dt.minute / 60.0
)

df["Meal_Hour_Sin"] = np.sin(
    2 * np.pi * df["Meal_Hour"] / 24
)

df["Meal_Hour_Cos"] = np.cos(
    2 * np.pi * df["Meal_Hour"] / 24
)


# ============================================================
# 5. FINAL FEATURE SET
# ============================================================

FEATURES = [
    "HbA1c",
    "Fasting_Glucose",
    "Carbs",
    "Protein",
    "Meal_Hour_Sin",
    "Meal_Hour_Cos",
    "Meal_Type",
]

TARGET = "iAUC"

GROUP = "Participant_ID"

X = df[FEATURES].copy()
y = df[TARGET].astype(float)
groups = df[GROUP]


print("\nFinal features:")

for feature in FEATURES:
    print(f"  - {feature}")

print(f"\nTarget: {TARGET}")


# ============================================================
# 6. PARTICIPANT-AWARE TRAIN/VALIDATION SPLIT
# ============================================================

print("\nCreating participant-aware split...")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED
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

groups_train = groups.iloc[train_idx]
groups_val = groups.iloc[val_idx]


print(
    f"\nTraining samples:   {len(X_train):,}"
)

print(
    f"Validation samples: {len(X_val):,}"
)

print(
    f"Training participants:   "
    f"{groups_train.nunique()}"
)

print(
    f"Validation participants: "
    f"{groups_val.nunique()}"
)


# ============================================================
# 7. VERIFY NO PARTICIPANT OVERLAP
# ============================================================

overlap = set(
    groups_train.unique()
).intersection(
    set(groups_val.unique())
)

print(
    f"\nParticipant overlap: {len(overlap)}"
)

if len(overlap) > 0:

    raise ValueError(
        "ERROR: Participant leakage detected!"
    )

print(
    "Participant separation verified."
)


# ============================================================
# 8. TRAINING-ONLY OUTLIER CLIPPING
# ============================================================

print("\nCalculating training-only clipping values...")

clipping_values = {}

for column in ["Carbs", "Protein"]:

    upper = X_train[column].quantile(0.99)

    clipping_values[column] = float(upper)

    X_train[column] = X_train[column].clip(
        upper=upper
    )

    X_val[column] = X_val[column].clip(
        upper=upper
    )

    print(
        f"{column} 99th percentile: "
        f"{upper:.4f}"
    )


# ============================================================
# 9. PREPROCESSING
# ============================================================

print("\nCreating preprocessing pipeline...")

categorical_features = [
    "Meal_Type"
]

numerical_features = [
    "HbA1c",
    "Fasting_Glucose",
    "Carbs",
    "Protein",
    "Meal_Hour_Sin",
    "Meal_Hour_Cos"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ],
    remainder="drop"
)


# IMPORTANT:
# Fit preprocessing ONLY on training data.

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_val_processed = preprocessor.transform(
    X_val
)


X_train_processed = np.asarray(
    X_train_processed
).astype(np.float32)

X_val_processed = np.asarray(
    X_val_processed
).astype(np.float32)


print(
    f"Processed training shape: "
    f"{X_train_processed.shape}"
)

print(
    f"Processed validation shape: "
    f"{X_val_processed.shape}"
)


# ============================================================
# 10. BUILD FINAL ANN
# ============================================================

print("\nBuilding ANN...")

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(X_train_processed.shape[1],)
    ),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.20
    ),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.20
    ),

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
# 11. COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="mse",

    metrics=["mae"]
)


print("\nModel summary:")

model.summary()


# ============================================================
# 12. EARLY STOPPING
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=30,

    restore_best_weights=True
)


# ============================================================
# 13. TRAIN
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING FINAL ANN")
print("=" * 70)

history = model.fit(

    X_train_processed,

    y_train.values,

    validation_data=(
        X_val_processed,
        y_val.values
    ),

    epochs=300,

    batch_size=32,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# ============================================================
# 14. VALIDATION PREDICTIONS
# ============================================================

print("\nGenerating validation predictions...")

predictions = model.predict(
    X_val_processed,
    verbose=0
).flatten()


# ============================================================
# 15. VALIDATION METRICS
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


print("\n")
print("=" * 70)
print("FINAL VALIDATION RESULTS")
print("=" * 70)

print(
    f"MAE:  {mae:.2f}"
)

print(
    f"RMSE: {rmse:.2f}"
)

print(
    f"R²:   {r2:.4f}"
)

print(
    f"Epochs trained: "
    f"{len(history.history['loss'])}"
)


# ============================================================
# 16. RESIDUAL ANALYSIS
# ============================================================

residuals = (
    y_val.values
    - predictions
)

print("\n")
print("=" * 70)
print("RESIDUAL SUMMARY")
print("=" * 70)

print(
    f"Mean residual:   "
    f"{np.mean(residuals):.2f}"
)

print(
    f"Median residual: "
    f"{np.median(residuals):.2f}"
)

print(
    f"Std residual:    "
    f"{np.std(residuals):.2f}"
)


# ============================================================
# 17. SAVE VALIDATION PREDICTIONS
# ============================================================

validation_predictions = pd.DataFrame({

    "Participant_ID":
        groups_val.values,

    "Meal_Timestamp":
        df.iloc[val_idx]["Meal_Timestamp"].values,

    "Meal_Type":
        df.iloc[val_idx]["Meal_Type"].values,

    "Actual_iAUC":
        y_val.values,

    "Predicted_iAUC":
        predictions,

    "Residual":
        residuals
})


validation_predictions.to_csv(
    f"{OUTPUT_DIR}/final_validation_predictions.csv",
    index=False
)


# ============================================================
# 18. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    f"{OUTPUT_DIR}/final_training_history.csv",
    index=False
)


# ============================================================
# 19. SAVE MODEL
# ============================================================

MODEL_PATH = (
    f"{OUTPUT_DIR}/cgmacros_ann.keras"
)

model.save(
    MODEL_PATH
)

print(
    f"\nModel saved to: {MODEL_PATH}"
)


# ============================================================
# 20. SAVE PREPROCESSOR
# ============================================================

PREPROCESSOR_PATH = (
    f"{OUTPUT_DIR}/preprocessor.joblib"
)

joblib.dump(
    preprocessor,
    PREPROCESSOR_PATH
)

print(
    f"Preprocessor saved to: "
    f"{PREPROCESSOR_PATH}"
)


# ============================================================
# 21. SAVE CLIPPING VALUES
# ============================================================

CLIPPING_PATH = (
    f"{OUTPUT_DIR}/clipping_values.json"
)

with open(
    CLIPPING_PATH,
    "w"
) as f:

    json.dump(
        clipping_values,
        f,
        indent=4
    )


print(
    f"Clipping values saved to: "
    f"{CLIPPING_PATH}"
)


# ============================================================
# 22. SAVE FEATURE CONFIGURATION
# ============================================================

feature_config = {

    "features": FEATURES,

    "numerical_features":
        numerical_features,

    "categorical_features":
        categorical_features,

    "target":
        TARGET,

    "group":
        GROUP,

    "meal_time_features": [
        "Meal_Hour_Sin",
        "Meal_Hour_Cos"
    ],

    "model_architecture": {
        "Dense_1": 64,
        "Dense_2": 32,
        "Dense_3": 16,
        "Dropout": 0.20,
        "Output": 1
    },

    "training": {
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 32,
        "max_epochs": 300,
        "early_stopping_patience": 30
    },

    "validation": {
        "method":
            "GroupShuffleSplit",
        "test_size": 0.20,
        "random_state": SEED
    }
}


FEATURE_CONFIG_PATH = (
    f"{OUTPUT_DIR}/final_feature_config.json"
)

with open(
    FEATURE_CONFIG_PATH,
    "w"
) as f:

    json.dump(
        feature_config,
        f,
        indent=4
    )


print(
    f"Feature configuration saved to: "
    f"{FEATURE_CONFIG_PATH}"
)


# ============================================================
# 23. SAVE FINAL METRICS
# ============================================================

metrics = {

    "MAE": float(mae),

    "RMSE": float(rmse),

    "R2": float(r2),

    "epochs_trained":
        int(len(history.history["loss"])),

    "training_samples":
        int(len(X_train)),

    "validation_samples":
        int(len(X_val)),

    "training_participants":
        int(groups_train.nunique()),

    "validation_participants":
        int(groups_val.nunique()),

    "participant_overlap":
        int(len(overlap))
}


METRICS_PATH = (
    f"{OUTPUT_DIR}/final_metrics.json"
)

with open(
    METRICS_PATH,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


print(
    f"Metrics saved to: "
    f"{METRICS_PATH}"
)


# ============================================================
# 24. FINAL CHECK
# ============================================================

print("\n")
print("=" * 70)
print("FINAL ANN TRAINING COMPLETE")
print("=" * 70)

print("\nFiles created:")

print("  model/cgmacros_ann.keras")
print("  model/preprocessor.joblib")
print("  model/clipping_values.json")
print("  model/final_feature_config.json")
print("  model/final_metrics.json")
print("  model/final_validation_predictions.csv")
print("  model/final_training_history.csv")

print("\nThe model is ready for the prediction engine.")