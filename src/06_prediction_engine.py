"""
CGMacros Prediction Engine

Loads the trained ANN model and predicts post-meal glycemic response
(iAUC) for a new meal input.

This file uses the EXACT preprocessing steps used during final ANN training.
"""

from pathlib import Path
from datetime import datetime, time

import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "model" / "cgmacros_ann.keras"
PREPROCESSOR_PATH = PROJECT_ROOT / "model" / "preprocessor.joblib"
CLIPPING_PATH = PROJECT_ROOT / "model" / "clipping_values.json"
FEATURE_CONFIG_PATH = PROJECT_ROOT / "model" / "final_feature_config.json"


# ============================================================
# LOAD MODEL FILES
# ============================================================

print("Loading CGMacros prediction engine...")

model = tf.keras.models.load_model(MODEL_PATH)

preprocessor = joblib.load(PREPROCESSOR_PATH)

with open(CLIPPING_PATH, "r") as f:
    clipping_values = json.load(f)

with open(FEATURE_CONFIG_PATH, "r") as f:
    feature_config = json.load(f)

print("Model loaded successfully.")
print("Preprocessor loaded successfully.")
print("Clipping values loaded successfully.")
print("Feature configuration loaded successfully.")


# ============================================================
# VALID MEAL TYPES
# ============================================================

VALID_MEAL_TYPES = {
    "breakfast": "breakfast",
    "lunch": "lunch",
    "dinner": "dinner",
    "snack": "snack",
    "snacks": "snack",
}


# ============================================================
# NORMALIZE MEAL TYPE
# ============================================================

def normalize_meal_type(meal_type):
    """
    Convert different user spellings/capitalizations
    into the standard meal types used during training.
    """

    if meal_type is None:
        raise ValueError("Meal type cannot be empty.")

    meal_type_clean = str(meal_type).strip().lower()

    if meal_type_clean not in VALID_MEAL_TYPES:
        valid = ", ".join(
            ["breakfast", "lunch", "dinner", "snack"]
        )

        raise ValueError(
            f"Invalid meal type '{meal_type}'. "
            f"Please use one of: {valid}."
        )

    return VALID_MEAL_TYPES[meal_type_clean]


# ============================================================
# CONVERT MEAL TIME TO DECIMAL HOUR
# ============================================================

def meal_time_to_decimal_hour(meal_time):
    """
    Accepts:

        "12:30"

        datetime.time

        datetime.datetime

    Returns:

        Decimal hour.
        Example:
        12:30 -> 12.5
    """

    # --------------------------------------------------------
    # String input
    # --------------------------------------------------------

    if isinstance(meal_time, str):

        meal_time = meal_time.strip()

        # Try 24-hour format
        try:

            parsed_time = datetime.strptime(
                meal_time,
                "%H:%M"
            ).time()

        except ValueError:

            # Try AM/PM format
            try:

                parsed_time = datetime.strptime(
                    meal_time,
                    "%I:%M %p"
                ).time()

            except ValueError:

                raise ValueError(
                    "Meal time must be in HH:MM format, "
                    "for example 12:30 or 18:45."
                )

        return (
            parsed_time.hour
            + parsed_time.minute / 60.0
        )

    # --------------------------------------------------------
    # datetime.time input
    # --------------------------------------------------------

    if isinstance(meal_time, time):

        return (
            meal_time.hour
            + meal_time.minute / 60.0
        )

    # --------------------------------------------------------
    # datetime.datetime input
    # --------------------------------------------------------

    if isinstance(meal_time, datetime):

        return (
            meal_time.hour
            + meal_time.minute / 60.0
        )

    raise ValueError(
        "Unsupported meal time format. "
        "Use 'HH:MM', datetime.time, or datetime.datetime."
    )


# ============================================================
# VALIDATE NUMERIC INPUT
# ============================================================

def validate_numeric(
    value,
    name,
    minimum=None
):
    """
    Validate a numeric user input.
    """

    try:

        value = float(value)

    except (TypeError, ValueError):

        raise ValueError(
            f"{name} must be a numeric value."
        )

    if not np.isfinite(value):

        raise ValueError(
            f"{name} must be a finite number."
        )

    if minimum is not None and value < minimum:

        raise ValueError(
            f"{name} must be greater than or equal to "
            f"{minimum}."
        )

    return value


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict_glycemic_response(
    hba1c,
    fasting_glucose,
    carbs,
    protein,
    meal_time,
    meal_type,
):
    """
    Predict post-meal glycemic response (iAUC).

    Parameters
    ----------
    hba1c : float
        HbA1c value.

    fasting_glucose : float
        Fasting glucose value.

    carbs : float
        Meal carbohydrate amount in grams.

    protein : float
        Meal protein amount in grams.

    meal_time : str / datetime.time / datetime.datetime
        Meal time.

    meal_type : str
        Breakfast, lunch, dinner, or snack.

    Returns
    -------
    float
        Predicted iAUC.
    """

    # ========================================================
    # 1. VALIDATE NUMERICAL INPUTS
    # ========================================================

    hba1c = validate_numeric(
        hba1c,
        "HbA1c",
        minimum=0
    )

    fasting_glucose = validate_numeric(
        fasting_glucose,
        "Fasting glucose",
        minimum=0
    )

    carbs = validate_numeric(
        carbs,
        "Carbohydrates",
        minimum=0
    )

    protein = validate_numeric(
        protein,
        "Protein",
        minimum=0
    )

    # ========================================================
    # 2. NORMALIZE MEAL TYPE
    # ========================================================

    meal_type = normalize_meal_type(
        meal_type
    )

    # ========================================================
    # 3. CONVERT MEAL TIME
    # ========================================================

    decimal_hour = meal_time_to_decimal_hour(
        meal_time
    )

    # ========================================================
    # 4. CREATE CYCLIC MEAL-TIME FEATURES
    #
    # Same method used during ANN training.
    # ========================================================

    meal_hour_sin = np.sin(
        2 * np.pi * decimal_hour / 24
    )

    meal_hour_cos = np.cos(
        2 * np.pi * decimal_hour / 24
    )

    # ========================================================
    # 5. APPLY TRAINING-TIME CLIPPING
    #
    # IMPORTANT:
    #
    # The clipping_values.json file contains:
    #
    # {
    #     "Carbs": 160.08,
    #     "Protein": 88.0
    # }
    #
    # These values were calculated from the training data.
    # ========================================================

    carbs_limit = float(
        clipping_values["Carbs"]
    )

    protein_limit = float(
        clipping_values["Protein"]
    )

    # Keep original values for possible future debugging
    carbs_original = carbs
    protein_original = protein

    # Apply clipping
    carbs = min(
        carbs,
        carbs_limit
    )

    protein = min(
        protein,
        protein_limit
    )

    # ========================================================
    # 6. BUILD INPUT DATAFRAME
    #
    # These column names must match the training data.
    # ========================================================

    input_data = pd.DataFrame(
        [
            {
                "HbA1c": hba1c,
                "Fasting_Glucose": fasting_glucose,
                "Carbs": carbs,
                "Protein": protein,
                "Meal_Hour_Sin": meal_hour_sin,
                "Meal_Hour_Cos": meal_hour_cos,
                "Meal_Type": meal_type,
            }
        ]
    )

    # ========================================================
    # 7. APPLY EXACT SAVED PREPROCESSING
    # ========================================================

    processed_input = preprocessor.transform(
        input_data
    )

    # ========================================================
    # 8. ANN PREDICTION
    # ========================================================

    prediction = model.predict(
        processed_input,
        verbose=0
    )

    predicted_iauc = float(
        prediction[0][0]
    )

    # ========================================================
    # 9. PREVENT NEGATIVE PREDICTION
    # ========================================================

    predicted_iauc = max(
        0.0,
        predicted_iauc
    )

    return predicted_iauc


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("CGMacross ANN PREDICTION ENGINE TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Example test input
    #
    # This is ONLY a software test.
    # It is NOT a medical recommendation.
    # --------------------------------------------------------

    test_hba1c = 5.4

    test_fasting_glucose = 91

    test_carbs = 50

    test_protein = 20

    test_meal_time = "12:30"

    test_meal_type = "lunch"

    # --------------------------------------------------------
    # Run prediction
    # --------------------------------------------------------

    prediction = predict_glycemic_response(
        hba1c=test_hba1c,
        fasting_glucose=test_fasting_glucose,
        carbs=test_carbs,
        protein=test_protein,
        meal_time=test_meal_time,
        meal_type=test_meal_type,
    )

    # ========================================================
    # DISPLAY TEST INPUT
    # ========================================================

    print()
    print("TEST INPUT")
    print("-" * 60)

    print(
        f"HbA1c:             {test_hba1c}"
    )

    print(
        f"Fasting glucose:   {test_fasting_glucose}"
    )

    print(
        f"Carbohydrates:     {test_carbs} g"
    )

    print(
        f"Protein:           {test_protein} g"
    )

    print(
        f"Meal time:         {test_meal_time}"
    )

    print(
        f"Meal type:         {test_meal_type}"
    )

    # ========================================================
    # DISPLAY PREDICTION
    # ========================================================

    print()
    print("-" * 60)
    print(
        "PREDICTED POST-MEAL GLYCEMIC RESPONSE"
    )
    print("-" * 60)

    print(
        f"Predicted iAUC: {prediction:.2f}"
    )

    print()
    print(
        "Prediction engine test completed successfully."
    )

    print("=" * 60)