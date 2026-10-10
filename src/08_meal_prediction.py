"""
CGMacros Meal Prediction

Connects:

    Food database
          ↓
    Quantity-based nutrient calculation
          ↓
    CGMacros ANN prediction engine
"""

from pathlib import Path
import sys
from importlib.util import spec_from_file_location, module_from_spec


# ============================================================
# MODULE PATH
# ============================================================

SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# LOAD NUMBERED PYTHON MODULE
# ============================================================

def load_module(module_name, filename):
    """Load a Python module whose filename starts with a number."""

    module_path = SRC_DIR / filename

    if not module_path.exists():
        raise FileNotFoundError(
            f"Module not found:\n{module_path}"
        )

    spec = spec_from_file_location(
        module_name,
        module_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module:\n{module_path}"
        )

    module = module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


# ============================================================
# LOAD EXISTING CGMacros MODULES
# ============================================================

food_database_module = load_module(
    "cgmacros_food_database",
    "07_food_database.py"
)

prediction_engine_module = load_module(
    "cgmacros_prediction_engine",
    "06_prediction_engine.py"
)


# ============================================================
# GET FUNCTIONS
# ============================================================

load_food_database = (
    food_database_module.load_food_database
)

calculate_nutrients = (
    food_database_module.calculate_nutrients
)

predict_glycemic_response = (
    prediction_engine_module.predict_glycemic_response
)


# ============================================================
# MEAL PREDICTION FUNCTION
# ============================================================

def predict_meal(
    food_name,
    quantity_grams,
    hba1c,
    fasting_glucose,
    meal_time,
    meal_type,
    food_df=None
):
    """
    Calculate nutrients from the selected food and quantity,
    then send the calculated nutrients to the CGMacros ANN.

    Parameters
    ----------
    food_name : str
        Food name from the food database.

    quantity_grams : float
        Quantity consumed in grams.

    hba1c : float
        User HbA1c value.

    fasting_glucose : float
        User fasting glucose value.

    meal_time : str
        Meal time, for example "12:30".

    meal_type : str
        breakfast, lunch, dinner, or snack.

    food_df : pandas.DataFrame, optional
        Loaded food database.

    Returns
    -------
    dict
        Food, nutrient and prediction information.
    """

    # --------------------------------------------------------
    # LOAD FOOD DATABASE
    # --------------------------------------------------------

    if food_df is None:
        food_df = load_food_database()

    # --------------------------------------------------------
    # CALCULATE FOOD NUTRIENTS
    # --------------------------------------------------------

    nutrition = calculate_nutrients(
        food_name=food_name,
        quantity_grams=quantity_grams,
        food_df=food_df
    )

    carbs = nutrition["Carbs_g"]
    protein = nutrition["Protein_g"]

    # --------------------------------------------------------
    # SEND VALUES TO ANN
    # --------------------------------------------------------

    predicted_iauc = predict_glycemic_response(
        hba1c=hba1c,
        fasting_glucose=fasting_glucose,
        carbs=carbs,
        protein=protein,
        meal_time=meal_time,
        meal_type=meal_type
    )

    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------

    result = {
        "Food": nutrition["Food"],
        "Quantity_g": nutrition["Quantity_g"],
        "Carbs_g": carbs,
        "Protein_g": protein,
        "HbA1c": float(hba1c),
        "Fasting_Glucose": float(fasting_glucose),
        "Meal_Time": str(meal_time),
        "Meal_Type": str(meal_type),
        "Predicted_iAUC": float(predicted_iauc),
    }

    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("CGMacros MEAL PREDICTION TEST")
    print("=" * 60)

    try:

        # ----------------------------------------------------
        # LOAD FOOD DATABASE
        # ----------------------------------------------------

        food_database = load_food_database()

        print()
        print(
            f"Food database loaded: "
            f"{len(food_database)} foods"
        )

        # ----------------------------------------------------
        # TEST USER INFORMATION
        # ----------------------------------------------------

        test_hba1c = 5.4
        test_fasting_glucose = 91

        # ----------------------------------------------------
        # TEST MEAL
        # ----------------------------------------------------

        test_food = "Rice, raw, milled"
        test_quantity = 100
        test_meal_time = "12:30"
        test_meal_type = "lunch"

        print()
        print("-" * 60)
        print("TEST MEAL")
        print("-" * 60)

        print(f"Food:              {test_food}")
        print(f"Quantity:          {test_quantity} g")
        print(f"HbA1c:             {test_hba1c}")
        print(f"Fasting glucose:   {test_fasting_glucose}")
        print(f"Meal time:         {test_meal_time}")
        print(f"Meal type:         {test_meal_type}")

        # ----------------------------------------------------
        # RUN COMPLETE PREDICTION
        # ----------------------------------------------------

        result = predict_meal(
            food_name=test_food,
            quantity_grams=test_quantity,
            hba1c=test_hba1c,
            fasting_glucose=test_fasting_glucose,
            meal_time=test_meal_time,
            meal_type=test_meal_type,
            food_df=food_database
        )

        # ----------------------------------------------------
        # SHOW NUTRIENTS
        # ----------------------------------------------------

        print()
        print("-" * 60)
        print("CALCULATED NUTRIENTS")
        print("-" * 60)

        print(
            f"Carbohydrates:     "
            f"{result['Carbs_g']:.2f} g"
        )

        print(
            f"Protein:           "
            f"{result['Protein_g']:.2f} g"
        )

        # ----------------------------------------------------
        # SHOW ANN RESULT
        # ----------------------------------------------------

        print()
        print("-" * 60)
        print("ANN PREDICTION")
        print("-" * 60)

        print(
            f"Predicted iAUC:    "
            f"{result['Predicted_iAUC']:.2f}"
        )

        print()
        print("=" * 60)
        print("Meal prediction test completed successfully.")
        print("=" * 60)
        print()

    except Exception as error:

        print()
        print("ERROR")
        print("-" * 60)
        print(error)

        print()
        print("=" * 60)
        print("Meal prediction test failed.")
        print("=" * 60)
        print()