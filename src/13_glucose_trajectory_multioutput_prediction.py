from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "glucose_trajectory_multioutput_ann.keras"
)

PREPROCESSOR_PATH = (
    BASE_DIR
    / "model"
    / "glucose_trajectory_multioutput_preprocessor.joblib"
)

CONFIG_PATH = (
    BASE_DIR
    / "model"
    / "glucose_trajectory_multioutput_feature_config.json"
)


model = tf.keras.models.load_model(MODEL_PATH)

preprocessor = joblib.load(PREPROCESSOR_PATH)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)


TIME_POINTS = config["time_points_minutes"]


def predict_glucose_trajectory(
    hba1c,
    fasting_glucose,
    carbs,
    protein,
    meal_time,
    meal_type,
):
    """
    Predict glucose at 0, 30, 60, 90 and 120 minutes
    after a meal using the multi-output ANN.
    """

    hba1c = float(hba1c)
    fasting_glucose = float(fasting_glucose)
    carbs = float(carbs)
    protein = float(protein)

    meal_time = str(meal_time)

    hour, minute = meal_time.split(":")

    hour = int(hour)
    minute = int(minute)

    decimal_hour = hour + minute / 60.0

    angle = 2 * np.pi * decimal_hour / 24.0

    meal_hour_sin = np.sin(angle)
    meal_hour_cos = np.cos(angle)

    input_df = pd.DataFrame(
        [
            {
                "HbA1c": hba1c,
                "Fasting_Glucose": fasting_glucose,
                "Carbs": carbs,
                "Protein": protein,
                "Meal_Hour_Sin": meal_hour_sin,
                "Meal_Hour_Cos": meal_hour_cos,
                "Meal_Type": meal_type.lower().strip(),
            }
        ]
    )

    processed = preprocessor.transform(input_df)

    prediction = model.predict(
        processed,
        verbose=0
    )[0]

    prediction = np.maximum(
        prediction,
        0
    )

    return {
        int(time_point): float(value)
        for time_point, value
        in zip(TIME_POINTS, prediction)
    }


if __name__ == "__main__":

    result = predict_glucose_trajectory(
        hba1c=6.0,
        fasting_glucose=110,
        carbs=77.8,
        protein=29.7,
        meal_time="12:30",
        meal_type="lunch",
    )

    print("=" * 60)
    print("MULTI-OUTPUT GLUCOSE TRAJECTORY PREDICTION")
    print("=" * 60)

    for time_point, glucose in result.items():
        print(
            f"{time_point:>3} min : "
            f"{glucose:.2f} mg/dL"
        )
