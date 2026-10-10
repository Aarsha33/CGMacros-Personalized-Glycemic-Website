"""
CGMacros Food Database

Connects the Indian food nutrition database to the
CGMacros ANN prediction system.

Nutrition values are based on the downloaded food
database and are reported per 100 grams.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FOOD_DATABASE_PATH = (
    PROJECT_ROOT / "data" / "indian-foods.csv"
)


# ============================================================
# LOAD FOOD DATABASE
# ============================================================

def load_food_database():
    """
    Load the Indian food database.

    Required columns:

        name
        protein_g
        carbs_g

    Nutrition values are per 100 g.
    """

    if not FOOD_DATABASE_PATH.exists():

        raise FileNotFoundError(
            f"Food database not found:\n"
            f"{FOOD_DATABASE_PATH}"
        )

    food_df = pd.read_csv(
        FOOD_DATABASE_PATH
    )

    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    food_df.columns = (
        food_df.columns
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "name",
        "protein_g",
        "carbs_g",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in food_df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Food database is missing required columns: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------------
    # Clean food names
    # --------------------------------------------------------

    food_df["name"] = (
        food_df["name"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Convert nutrition values to numeric
    # --------------------------------------------------------

    food_df["protein_g"] = pd.to_numeric(
        food_df["protein_g"],
        errors="coerce"
    )

    food_df["carbs_g"] = pd.to_numeric(
        food_df["carbs_g"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    food_df = food_df.dropna(
        subset=[
            "name",
            "protein_g",
            "carbs_g",
        ]
    )

    # --------------------------------------------------------
    # Remove negative nutrition values
    # --------------------------------------------------------

    food_df = food_df[
        (food_df["protein_g"] >= 0)
        &
        (food_df["carbs_g"] >= 0)
    ]

    # --------------------------------------------------------
    # Remove duplicate food names
    # --------------------------------------------------------

    food_df = food_df.drop_duplicates(
        subset=["name"],
        keep="first"
    )

    # --------------------------------------------------------
    # Reset index
    # --------------------------------------------------------

    food_df = food_df.reset_index(
        drop=True
    )

    return food_df


# ============================================================
# SEARCH FOOD
# ============================================================

def search_food(
    search_text,
    food_df=None
):
    """
    Search for foods whose names contain
    the supplied search text.

    Example:

        search_food("rice")
    """

    if food_df is None:

        food_df = load_food_database()

    search_text = str(
        search_text
    ).strip().lower()

    if not search_text:

        return food_df.copy()

    matches = food_df[
        food_df["name"]
        .str.lower()
        .str.contains(
            search_text,
            na=False
        )
    ]

    return matches.reset_index(
        drop=True
    )


# ============================================================
# GET EXACT FOOD
# ============================================================

def get_food(
    food_name,
    food_df=None
):
    """
    Find one food by exact name.

    The search is case-insensitive.
    """

    if food_df is None:

        food_df = load_food_database()

    food_name = str(
        food_name
    ).strip().lower()

    matches = food_df[
        food_df["name"]
        .str.lower()
        == food_name
    ]

    if matches.empty:

        raise ValueError(
            f"Food '{food_name}' was not found "
            f"in the database."
        )

    return matches.iloc[0]


# ============================================================
# CALCULATE NUTRIENTS
# ============================================================

def calculate_nutrients(
    food_name,
    quantity_grams,
    food_df=None
):
    """
    Calculate carbohydrates and protein for
    a selected food quantity.

    Database values are per 100 g.

    Formula:

        nutrient =
            nutrient_per_100g × quantity / 100
    """

    if food_df is None:

        food_df = load_food_database()

    # --------------------------------------------------------
    # Validate quantity
    # --------------------------------------------------------

    try:

        quantity_grams = float(
            quantity_grams
        )

    except (TypeError, ValueError):

        raise ValueError(
            "Quantity must be a numeric value."
        )

    if quantity_grams <= 0:

        raise ValueError(
            "Quantity must be greater than 0 grams."
        )

    # --------------------------------------------------------
    # Find food
    # --------------------------------------------------------

    food = get_food(
        food_name,
        food_df
    )

    # --------------------------------------------------------
    # Nutrition per 100 g
    # --------------------------------------------------------

    carbs_per_100g = float(
        food["carbs_g"]
    )

    protein_per_100g = float(
        food["protein_g"]
    )

    # --------------------------------------------------------
    # Calculate nutrition for selected quantity
    # --------------------------------------------------------

    carbs = (
        carbs_per_100g
        * quantity_grams
        / 100.0
    )

    protein = (
        protein_per_100g
        * quantity_grams
        / 100.0
    )

    return {
        "Food": food["name"],
        "Quantity_g": quantity_grams,
        "Carbs_g": carbs,
        "Protein_g": protein,
    }


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("CGMacros FOOD DATABASE TEST")
    print("=" * 60)

    try:

        # ----------------------------------------------------
        # Load database
        # ----------------------------------------------------

        food_database = load_food_database()

        print()
        print(
            f"Foods loaded: {len(food_database)}"
        )

        # ----------------------------------------------------
        # Display columns
        # ----------------------------------------------------

        print()
        print("Database columns:")

        print(
            list(food_database.columns)
        )

        # ----------------------------------------------------
        # Display first five foods
        # ----------------------------------------------------

        print()
        print("First 5 foods:")
        print(
            food_database[
                [
                    "name",
                    "protein_g",
                    "carbs_g",
                ]
            ]
            .head()
            .to_string(index=False)
        )

        # ----------------------------------------------------
        # Search test
        # ----------------------------------------------------

        print()
        print("-" * 60)
        print("FOOD SEARCH TEST")
        print("-" * 60)

        rice_results = search_food(
            "rice",
            food_database
        )

        print(
            rice_results[
                [
                    "name",
                    "protein_g",
                    "carbs_g",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

        # ----------------------------------------------------
        # Nutrition calculation test
        # ----------------------------------------------------

        print()
        print("-" * 60)
        print("NUTRITION CALCULATION TEST")
        print("-" * 60)

        # Use the first food as a guaranteed
        # database entry for testing.

        test_food = food_database.iloc[0]["name"]

        nutrition = calculate_nutrients(
            food_name=test_food,
            quantity_grams=150,
            food_df=food_database
        )

        print(
            f"Food:        {nutrition['Food']}"
        )

        print(
            f"Quantity:    "
            f"{nutrition['Quantity_g']:.1f} g"
        )

        print(
            f"Carbs:       "
            f"{nutrition['Carbs_g']:.2f} g"
        )

        print(
            f"Protein:     "
            f"{nutrition['Protein_g']:.2f} g"
        )

        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

        print()
        print("-" * 60)
        print(
            "Food database test completed successfully."
        )
        print("-" * 60)

    except Exception as error:

        print()
        print("ERROR")
        print("-" * 60)
        print(error)

    print()
    print("=" * 60)