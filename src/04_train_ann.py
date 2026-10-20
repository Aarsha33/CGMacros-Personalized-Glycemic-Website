# ============================================================
# 04_train_ann.py
# ANN Model Comparison for CGMacros
# ============================================================

import os
import json
import random
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GroupKFold
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
print("CGMacros ANN MODEL TRAINING")
print("=" * 70)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Participants: {df['Participant_ID'].nunique()}")

print("\nAvailable columns:")
print(df.columns.tolist())


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

# Convert time of day into circular features.
# This avoids treating 23:00 and 00:00 as very far apart.

df["Meal_Hour_Sin"] = np.sin(
    2 * np.pi * df["Meal_Hour"] / 24
)

df["Meal_Hour_Cos"] = np.cos(
    2 * np.pi * df["Meal_Hour"] / 24
)


# ============================================================
# 5. TARGET
# ============================================================

TARGET = "iAUC"

y = df[TARGET].astype(float)

groups = df["Participant_ID"]


# ============================================================
# 6. FEATURE SETS
# ============================================================

MODEL_A_FEATURES = [
    "Age",
    "Gender",
    "BMI",
    "HbA1c",
    "Fasting_Glucose",
    "Carbs",
    "Protein",
    "Meal_Hour_Sin",
    "Meal_Hour_Cos",
    "Meal_Type",
]

MODEL_B_FEATURES = [
    "HbA1c",
    "Fasting_Glucose",
    "Carbs",
    "Protein",
    "Meal_Hour_Sin",
    "Meal_Hour_Cos",
    "Meal_Type",
]


# ============================================================
# 7. ANN ARCHITECTURE
# ============================================================

def build_ann(input_dim):

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),

        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.20),

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.20),

        tf.keras.layers.Dense(
            16,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            1,
            activation="linear"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"]
    )

    return model


# ============================================================
# 8. PREPROCESSING FUNCTION
# ============================================================

def create_preprocessor(features):

    categorical_features = []

    if "Gender" in features:
        categorical_features.append("Gender")

    if "Meal_Type" in features:
        categorical_features.append("Meal_Type")

    numerical_features = [
        feature
        for feature in features
        if feature not in categorical_features
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

    return preprocessor


# ============================================================
# 9. TRAINING FUNCTION
# ============================================================

def evaluate_model(model_name, features):

    print("\n")
    print("=" * 70)
    print(f"TRAINING {model_name}")
    print("=" * 70)

    X = df[features].copy()

    n_splits = 5

    group_kfold = GroupKFold(
        n_splits=n_splits
    )

    fold_results = []
    all_predictions = []

    for fold, (train_idx, val_idx) in enumerate(
        group_kfold.split(
            X,
            y,
            groups=groups
        ),
        start=1
    ):

        print("\n" + "-" * 70)
        print(f"FOLD {fold}/{n_splits}")
        print("-" * 70)

        X_train = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()

        y_train = y.iloc[train_idx].copy()
        y_val = y.iloc[val_idx].copy()

        groups_train = groups.iloc[train_idx]
        groups_val = groups.iloc[val_idx]

        print(
            f"Training samples:   {len(X_train):,}"
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

        # ----------------------------------------------------
        # TRAINING-ONLY CLIPPING
        # ----------------------------------------------------

        clipping_values = {}

        for column in ["Carbs", "Protein"]:

            if column in X_train.columns:

                upper = X_train[column].quantile(
                    0.99
                )

                clipping_values[column] = float(
                    upper
                )

                X_train[column] = X_train[column].clip(
                    upper=upper
                )

                X_val[column] = X_val[column].clip(
                    upper=upper
                )

        # ----------------------------------------------------
        # PREPROCESSOR
        # ----------------------------------------------------

        preprocessor = create_preprocessor(
            features
        )

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
            f"Processed input features: "
            f"{X_train_processed.shape[1]}"
        )

        # ----------------------------------------------------
        # BUILD ANN
        # ----------------------------------------------------

        model = build_ann(
            X_train_processed.shape[1]
        )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=30,
            restore_best_weights=True
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

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
            verbose=0
        )

        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        predictions = model.predict(
            X_val_processed,
            verbose=0
        ).flatten()

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # SAVE FOLD RESULTS
        # ----------------------------------------------------

        fold_results.append({
            "Model": model_name,
            "Fold": fold,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Train_Samples": len(X_train),
            "Validation_Samples": len(X_val),
            "Train_Participants": groups_train.nunique(),
            "Validation_Participants": groups_val.nunique()
        })

        fold_predictions = pd.DataFrame({
            "Model": model_name,
            "Fold": fold,
            "Participant_ID": groups_val.values,
            "Actual_iAUC": y_val.values,
            "Predicted_iAUC": predictions
        })

        all_predictions.append(
            fold_predictions
        )

    # ========================================================
    # COMBINE FOLD RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        fold_results
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    # ========================================================
    # OVERALL OUT-OF-FOLD METRICS
    # ========================================================

    overall_mae = mean_absolute_error(
        predictions_df["Actual_iAUC"],
        predictions_df["Predicted_iAUC"]
    )

    overall_rmse = np.sqrt(
        mean_squared_error(
            predictions_df["Actual_iAUC"],
            predictions_df["Predicted_iAUC"]
        )
    )

    overall_r2 = r2_score(
        predictions_df["Actual_iAUC"],
        predictions_df["Predicted_iAUC"]
    )

    print("\n")
    print("=" * 70)
    print(f"{model_name} OVERALL OUT-OF-FOLD RESULTS")
    print("=" * 70)

    print(
        f"OOF MAE:  {overall_mae:.2f}"
    )

    print(
        f"OOF RMSE: {overall_rmse:.2f}"
    )

    print(
        f"OOF R²:   {overall_r2:.4f}"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df.to_csv(
        f"{OUTPUT_DIR}/{model_name}_fold_results.csv",
        index=False
    )

    predictions_df.to_csv(
        f"{OUTPUT_DIR}/{model_name}_oof_predictions.csv",
        index=False
    )

    return {
        "Model": model_name,
        "OOF_MAE": overall_mae,
        "OOF_RMSE": overall_rmse,
        "OOF_R2": overall_r2
    }


# ============================================================
# 10. RUN MODEL A
# ============================================================

model_a_result = evaluate_model(
    "Model_A_Personalized",
    MODEL_A_FEATURES
)


# ============================================================
# 11. RUN MODEL B
# ============================================================

model_b_result = evaluate_model(
    "Model_B_Reduced",
    MODEL_B_FEATURES
)


# ============================================================
# 12. COMPARE MODELS
# ============================================================

comparison = pd.DataFrame([
    model_a_result,
    model_b_result
])

print("\n")
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# 13. SAVE COMPARISON
# ============================================================

comparison.to_csv(
    f"{OUTPUT_DIR}/ann_model_comparison.csv",
    index=False
)


# ============================================================
# 14. SAVE FEATURE CONFIGURATION
# ============================================================

feature_config = {
    "Model_A_Personalized": MODEL_A_FEATURES,
    "Model_B_Reduced": MODEL_B_FEATURES,
    "Target": TARGET
}

with open(
    f"{OUTPUT_DIR}/feature_config.json",
    "w"
) as f:

    json.dump(
        feature_config,
        f,
        indent=4
    )


# ============================================================
# 15. FINISHED
# ============================================================

print("\n")
print("=" * 70)
print("ANN MODEL COMPARISON COMPLETE")
print("=" * 70)

print("\nSaved files:")

print(
    "model/Model_A_Personalized_fold_results.csv"
)

print(
    "model/Model_A_Personalized_oof_predictions.csv"
)

print(
    "model/Model_B_Reduced_fold_results.csv"
)

print(
    "model/Model_B_Reduced_oof_predictions.csv"
)

print(
    "model/ann_model_comparison.csv"
)

print(
    "model/feature_config.json"
)