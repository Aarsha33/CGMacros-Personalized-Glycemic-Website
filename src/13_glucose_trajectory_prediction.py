from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "model"

MODEL_FILE = (
    MODEL_DIR / "glucose_trajectory_ann.keras"
)

PREPROCESSOR_FILE = (
    MODEL_DIR / "glucose_trajectory_preprocessor.joblib"
)

CONFIG_FILE = (
    MODEL_DIR / "glucose_trajectory_feature_config.json"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = tf.keras.models.load_model(
    MODEL_FILE
)

preprocessor = joblib.load(
    PREPROCESSOR_FILE
)

with open(CONFIG_FILE, "r") as f:
    feature_config = json.load(f)


# ============================================================
# VALID OPTIONS
# ============================================================

VALID_MEAL_TYPES = [
    "breakfast",
    "lunch",
    "dinner",
    "snack"
]

TIME_POINTS = [
    0,
    30,
    60,
    90,
    120
]


# ============================================================
# HELPER: MEAL TIME
# ============================================================

def meal_time_to_decimal(meal_time):
    """
    Convert HH:MM meal time into decimal hours.
    """

    if hasattr(meal_time, "hour"):
        return (
            meal_time.hour
            + meal_time.minute / 60.0
        )

    time_string = str(meal_time)

    parts = time_string.split(":")

    hour = int(parts[0])
    minute = int(parts[1])

    return hour + minute / 60.0


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict_glucose_trajectory(
    hba1c,
    fasting_glucose,
    carbs,
    protein,
    meal_time,
    meal_type
):
    """
    Predict glucose at:

    0, 30, 60, 90 and 120 minutes
    after a meal.
    """

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    hba1c = float(hba1c)
    fasting_glucose = float(fasting_glucose)
    carbs = float(carbs)
    protein = float(protein)

    if hba1c <= 0:
        raise ValueError(
            "HbA1c must be greater than 0."
        )

    if fasting_glucose <= 0:
        raise ValueError(
            "Fasting glucose must be greater than 0."
        )

    if carbs < 0:
        raise ValueError(
            "Carbohydrates cannot be negative."
        )

    if protein < 0:
        raise ValueError(
            "Protein cannot be negative."
        )

    meal_type = str(
        meal_type
    ).strip().lower()

    if meal_type not in VALID_MEAL_TYPES:
        raise ValueError(
            f"Meal type must be one of: "
            f"{VALID_MEAL_TYPES}"
        )

    # --------------------------------------------------------
    # Convert meal time
    # --------------------------------------------------------

    decimal_hour = meal_time_to_decimal(
        meal_time
    )

    # --------------------------------------------------------
    # Cyclical meal-time features
    # --------------------------------------------------------

    meal_hour_sin = np.sin(
        2 * np.pi * decimal_hour / 24
    )

    meal_hour_cos = np.cos(
        2 * np.pi * decimal_hour / 24
    )

    # --------------------------------------------------------
    # Create one row for every time point
    # --------------------------------------------------------

    rows = []

    for time_after_meal in TIME_POINTS:

        rows.append({
            "HbA1c": hba1c,
            "Fasting_Glucose": fasting_glucose,
            "Carbs": carbs,
            "Protein": protein,
            "Meal_Hour_Sin": meal_hour_sin,
            "Meal_Hour_Cos": meal_hour_cos,
            "Time_After_Meal": time_after_meal,
            "Meal_Type": meal_type
        })

    input_df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    processed = preprocessor.transform(
        input_df
    )

    processed = np.asarray(
        processed
    ).astype("float32")

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = model.predict(
        processed,
        verbose=0
    ).flatten()

    # Prevent impossible negative glucose
    predictions = np.maximum(
        predictions,
        0
    )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    result = pd.DataFrame({
        "Time_After_Meal": TIME_POINTS,
        "Predicted_Glucose": predictions
    })

    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    result = predict_glucose_trajectory(
        hba1c=5.4,
        fasting_glucose=91,
        carbs=50,
        protein=20,
        meal_time="12:30",
        meal_type="lunch"
    )

    print("=" * 70)
    print("GLUCOSE TRAJECTORY PREDICTION TEST")
    print("=" * 70)

    print(result.to_string(index=False))

    print("=" * 70)