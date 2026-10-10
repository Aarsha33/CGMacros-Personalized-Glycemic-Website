"""
CGMacros What-If Simulation Engine

Compares a user's original meal prediction with
a modified meal prediction.

Supported changes:
- Food quantity
- Food type
- Meal time
"""

from pathlib import Path
import sys
import importlib.util


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# =========================================================
# LOAD 08_meal_prediction.py
# =========================================================

def load_module(module_name, file_path):
    """Load a Python module from a numbered filename."""

    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module from: {file_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


meal_prediction_module = load_module(
    "meal_prediction",
    SRC_DIR / "08_meal_prediction.py"
)

predict_meal = meal_prediction_module.predict_meal


# =========================================================
# PERCENTAGE CHANGE
# =========================================================

def calculate_percentage_change(before, after):
    """
    Calculate percentage change from the original
    prediction to the modified prediction.
    """

    if before == 0:
        return None

    return ((after - before) / before) * 100


# =========================================================
# SIMPLE INTERPRETATION
# =========================================================

def get_interpretation(before, after):
    """
    Convert the numerical comparison into
    simple user-friendly language.
    """

    if after < before:
        return (
            "The model predicts a lower post-meal "
            "response after this change."
        )

    elif after > before:
        return (
            "The model predicts a higher post-meal "
            "response after this change."
        )

    else:
        return (
            "The model predicts no change in the "
            "post-meal response."
        )


# =========================================================
# WHAT-IF SIMULATION
# =========================================================

def run_what_if(
    original_food,
    original_quantity,
    original_hba1c,
    original_fasting_glucose,
    original_meal_time,
    original_meal_type,
    new_food=None,
    new_quantity=None,
    new_meal_time=None,
):
    """
    Compare the original meal with a modified meal.

    The user's HbA1c, fasting glucose and meal type
    remain unchanged.

    Parameters
    ----------
    original_food : str
        Original food item.

    original_quantity : float
        Original quantity in grams.

    original_hba1c : float
        User HbA1c.

    original_fasting_glucose : float
        User fasting glucose.

    original_meal_time : str
        Original meal time, for example "12:30".

    original_meal_type : str
        Original meal type.

    new_food : str, optional
        Modified food item.

    new_quantity : float, optional
        Modified quantity in grams.

    new_meal_time : str, optional
        Modified meal time.

    Returns
    -------
    dict
        Comparison results.
    """

    # -----------------------------------------------------
    # Keep original values if no change is supplied
    # -----------------------------------------------------

    if new_food is None:
        new_food = original_food

    if new_quantity is None:
        new_quantity = original_quantity

    if new_meal_time is None:
        new_meal_time = original_meal_time

    # =====================================================
    # ORIGINAL MEAL
    # =====================================================

    before_result = predict_meal(
        food_name=original_food,
        quantity_grams=original_quantity,
        hba1c=original_hba1c,
        fasting_glucose=original_fasting_glucose,
        meal_time=original_meal_time,
        meal_type=original_meal_type,
    )

    # =====================================================
    # MODIFIED MEAL
    # =====================================================

    after_result = predict_meal(
        food_name=new_food,
        quantity_grams=new_quantity,
        hba1c=original_hba1c,
        fasting_glucose=original_fasting_glucose,
        meal_time=new_meal_time,
        meal_type=original_meal_type,
    )

    # =====================================================
    # EXTRACT PREDICTIONS
    # =====================================================

    # IMPORTANT:
    # 08_meal_prediction.py returns "Predicted_iAUC"
    # with capital letters.

    before_prediction = float(
        before_result["Predicted_iAUC"]
    )

    after_prediction = float(
        after_result["Predicted_iAUC"]
    )

    # =====================================================
    # CALCULATE CHANGES
    # =====================================================

    absolute_change = (
        after_prediction - before_prediction
    )

    percentage_change = calculate_percentage_change(
        before_prediction,
        after_prediction
    )

    interpretation = get_interpretation(
        before_prediction,
        after_prediction
    )

    # =====================================================
    # RETURN RESULTS
    # =====================================================

    return {

        # Predictions
        "before": before_prediction,
        "after": after_prediction,

        # Changes
        "absolute_change": absolute_change,
        "percentage_change": percentage_change,

        # Interpretation
        "interpretation": interpretation,

        # Original meal
        "original_food": original_food,
        "original_quantity": original_quantity,
        "original_meal_time": original_meal_time,

        # Modified meal
        "new_food": new_food,
        "new_quantity": new_quantity,
        "new_meal_time": new_meal_time,

        # Nutrition
        # IMPORTANT:
        # 08_meal_prediction.py returns
        # "Carbs_g" and "Protein_g".

        "before_carbs": before_result["Carbs_g"],
        "before_protein": before_result["Protein_g"],

        "after_carbs": after_result["Carbs_g"],
        "after_protein": after_result["Protein_g"],
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CGMacros What-If Simulation Test")
    print("=" * 60)

    # -----------------------------------------------------
    # ORIGINAL MEAL
    # -----------------------------------------------------

    result = run_what_if(
        original_food="Rice, raw, milled",
        original_quantity=100,

        original_hba1c=5.4,
        original_fasting_glucose=91,

        original_meal_time="12:30",
        original_meal_type="lunch",

        # -------------------------------------------------
        # WHAT-IF CHANGE
        # Increase rice from 100 g to 150 g
        # -------------------------------------------------

        new_quantity=150,
    )

    # =====================================================
    # DISPLAY ORIGINAL MEAL
    # =====================================================

    print()
    print("ORIGINAL MEAL")
    print("-" * 60)

    print(
        f"Food:              {result['original_food']}"
    )

    print(
        f"Quantity:           "
        f"{result['original_quantity']} g"
    )

    print(
        f"Meal time:          "
        f"{result['original_meal_time']}"
    )

    print(
        f"Carbohydrates:     "
        f"{result['before_carbs']:.2f} g"
    )

    print(
        f"Protein:            "
        f"{result['before_protein']:.2f} g"
    )

    print(
        f"Predicted iAUC:     "
        f"{result['before']:.2f}"
    )

    # =====================================================
    # DISPLAY WHAT-IF MEAL
    # =====================================================

    print()
    print("WHAT-IF MEAL")
    print("-" * 60)

    print(
        f"Food:              {result['new_food']}"
    )

    print(
        f"Quantity:           "
        f"{result['new_quantity']} g"
    )

    print(
        f"Meal time:          "
        f"{result['new_meal_time']}"
    )

    print(
        f"Carbohydrates:     "
        f"{result['after_carbs']:.2f} g"
    )

    print(
        f"Protein:            "
        f"{result['after_protein']:.2f} g"
    )

    print(
        f"Predicted iAUC:     "
        f"{result['after']:.2f}"
    )

    # =====================================================
    # COMPARISON
    # =====================================================

    print()
    print("COMPARISON")
    print("-" * 60)

    print(
        f"Before:             "
        f"{result['before']:.2f}"
    )

    print(
        f"After:              "
        f"{result['after']:.2f}"
    )

    print(
        f"Absolute change:    "
        f"{result['absolute_change']:.2f}"
    )

    if result["percentage_change"] is not None:

        print(
            f"Percentage change:  "
            f"{result['percentage_change']:.2f}%"
        )

    # =====================================================
    # INTERPRETATION
    # =====================================================

    print()
    print("INTERPRETATION")
    print("-" * 60)

    print(result["interpretation"])

    print()
    print("=" * 60)