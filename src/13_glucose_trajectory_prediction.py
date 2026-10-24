from pathlib import Path
import importlib.util


# Load the new multi-output trajectory predictor
BASE_DIR = Path(__file__).resolve().parent

module_path = BASE_DIR / "13_glucose_trajectory_multioutput_prediction.py"

spec = importlib.util.spec_from_file_location(
    "multioutput_trajectory",
    module_path
)

trajectory_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trajectory_module)


def predict_glucose_trajectory(
    hba1c,
    fasting_glucose,
    carbs,
    protein,
    meal_time,
    meal_type
):
    """
    Returns the trajectory in the format expected by app.py.
    """

    predictions = trajectory_module.predict_glucose_trajectory(
        hba1c=hba1c,
        fasting_glucose=fasting_glucose,
        carbs=carbs,
        protein=protein,
        meal_time=meal_time,
        meal_type=meal_type
    )

    times = [0, 30, 60, 90, 120]

    return {
        "Time_After_Meal": times,
        "Predicted_Glucose": [
            float(predictions[t])
            for t in times
        ]
    }
