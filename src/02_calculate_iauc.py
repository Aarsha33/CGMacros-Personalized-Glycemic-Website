from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CGMacros - Calculate Meal-Level Incremental AUC (iAUC)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "combined_cgmacros.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "meal_iauc_dataset.csv"

POST_MEAL_MINUTES = 120
BASELINE_LOOKBACK_MINUTES = 30


print("=" * 70)
print("CGMacros MEAL-LEVEL iAUC CALCULATION")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load combined dataset
# ------------------------------------------------------------

print("\nLoading combined dataset...")

df = pd.read_csv(INPUT_FILE)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    errors="coerce"
)

print(f"Rows loaded: {len(df):,}")


# ------------------------------------------------------------
# 2. Normalize meal types
# ------------------------------------------------------------

def normalize_meal_type(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    if value in ["breakfast"]:
        return "breakfast"

    if value in ["lunch"]:
        return "lunch"

    if value in ["dinner"]:
        return "dinner"

    if value in ["snack", "snacks"] or value.startswith("snack "):
        return "snack"

    return np.nan


df["Meal_Type_Normalized"] = df["Meal Type"].apply(
    normalize_meal_type
)


# ------------------------------------------------------------
# 3. Select meal rows
# ------------------------------------------------------------

meal_rows = df[
    df["Meal_Type_Normalized"].notna()
].copy()

meal_rows = meal_rows.sort_values(
    ["Participant_ID", "Timestamp"]
).reset_index(drop=True)

print(f"Meal rows found: {len(meal_rows):,}")

print("\nNormalized meal types:")
print(
    meal_rows["Meal_Type_Normalized"]
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# 4. Prepare CGM data
# ------------------------------------------------------------

cgm_df = df[
    [
        "Participant_ID",
        "Timestamp",
        "Libre GL"
    ]
].copy()

cgm_df["Libre GL"] = pd.to_numeric(
    cgm_df["Libre GL"],
    errors="coerce"
)

cgm_df = cgm_df.dropna(
    subset=["Timestamp", "Libre GL"]
)

cgm_df = cgm_df.sort_values(
    ["Participant_ID", "Timestamp"]
)

print(f"\nValid Libre GL rows: {len(cgm_df):,}")


# ------------------------------------------------------------
# 5. Calculate iAUC for each meal
# ------------------------------------------------------------

results = []

print("\nCalculating meal responses...")

for meal_number, meal in meal_rows.iterrows():

    participant_id = meal["Participant_ID"]
    meal_time = meal["Timestamp"]

    # --------------------------------------------------------
    # Find participant's CGM data
    # --------------------------------------------------------

    participant_cgm = cgm_df[
        cgm_df["Participant_ID"] == participant_id
    ]

    if participant_cgm.empty:
        continue

    # --------------------------------------------------------
    # Baseline window
    #
    # Look for the closest valid Libre glucose measurement
    # from 30 minutes before the meal up to the meal time.
    # --------------------------------------------------------

    baseline_start = (
        meal_time
        - pd.Timedelta(minutes=BASELINE_LOOKBACK_MINUTES)
    )

    baseline_data = participant_cgm[
        (participant_cgm["Timestamp"] >= baseline_start)
        &
        (participant_cgm["Timestamp"] <= meal_time)
    ]

    if baseline_data.empty:
        continue

    # Closest glucose measurement to meal time
    baseline_row = baseline_data.iloc[
        (baseline_data["Timestamp"] - meal_time)
        .abs()
        .argsort()
        .iloc[0]
    ]

    baseline_glucose = float(
        baseline_row["Libre GL"]
    )

    # --------------------------------------------------------
    # Post-meal 0-120 minute window
    # --------------------------------------------------------

    postmeal_end = (
        meal_time
        + pd.Timedelta(minutes=POST_MEAL_MINUTES)
    )

    postmeal_data = participant_cgm[
        (participant_cgm["Timestamp"] >= meal_time)
        &
        (participant_cgm["Timestamp"] <= postmeal_end)
    ].copy()

    if postmeal_data.empty:
        continue

    # --------------------------------------------------------
    # Calculate incremental glucose
    #
    # Only glucose above baseline contributes to iAUC.
    # --------------------------------------------------------

    postmeal_data["Incremental_Glucose"] = (
        postmeal_data["Libre GL"]
        - baseline_glucose
    ).clip(lower=0)

    # Time in minutes from meal
    postmeal_data["Minutes_From_Meal"] = (
        (
            postmeal_data["Timestamp"]
            - meal_time
        ).dt.total_seconds()
        / 60.0
    )

    postmeal_data = postmeal_data.sort_values(
        "Minutes_From_Meal"
    )

    # --------------------------------------------------------
    # Trapezoidal integration
    # --------------------------------------------------------

    if len(postmeal_data) < 2:
        continue

    time_values = postmeal_data[
        "Minutes_From_Meal"
    ].to_numpy()

    glucose_values = postmeal_data[
        "Incremental_Glucose"
    ].to_numpy()

    iauc = np.trapezoid(
        glucose_values,
        time_values
    )

    # --------------------------------------------------------
    # Store meal-level record
    # --------------------------------------------------------

    result = {
        "Participant_ID": participant_id,
        "Meal_Timestamp": meal_time,
        "Meal_Type": meal["Meal_Type_Normalized"],

        "Calories": pd.to_numeric(
            meal["Calories"],
            errors="coerce"
        ),

        "Carbs": pd.to_numeric(
            meal["Carbs"],
            errors="coerce"
        ),

        "Protein": pd.to_numeric(
            meal["Protein"],
            errors="coerce"
        ),

        "Fat": pd.to_numeric(
            meal["Fat"],
            errors="coerce"
        ),

        "Fiber": pd.to_numeric(
            meal["Fiber"],
            errors="coerce"
        ),

        "Amount_Consumed": pd.to_numeric(
            meal["Amount Consumed"],
            errors="coerce"
        ),

        "Baseline_Glucose": baseline_glucose,

        "Postmeal_CGM_Points": len(
            postmeal_data
        ),

        "iAUC": float(iauc)
    }

    results.append(result)


# ------------------------------------------------------------
# 6. Create meal-level dataset
# ------------------------------------------------------------

meal_dataset = pd.DataFrame(results)

print("\n" + "=" * 70)
print("iAUC RESULTS")
print("=" * 70)

print(f"\nMeals with valid iAUC: {len(meal_dataset):,}")

if len(meal_dataset) == 0:
    raise RuntimeError(
        "No valid meal-level iAUC records were created."
    )


# ------------------------------------------------------------
# 7. Basic target statistics
# ------------------------------------------------------------

print("\niAUC statistics:")

print(
    meal_dataset["iAUC"]
    .describe()
    .to_string()
)

print("\nMean iAUC by meal type:")

print(
    meal_dataset
    .groupby("Meal_Type")["iAUC"]
    .agg(["count", "mean", "median"])
    .round(2)
    .to_string()
)


# ------------------------------------------------------------
# 8. Check zero iAUC values
# ------------------------------------------------------------

zero_iauc = (
    meal_dataset["iAUC"] == 0
).sum()

print(
    f"\nZero-iAUC meals: {zero_iauc:,}"
)


# ------------------------------------------------------------
# 9. Save meal-level dataset
# ------------------------------------------------------------

meal_dataset.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(f"\nOutput file:")
print(OUTPUT_FILE)

print("\nMeal-level iAUC calculation completed successfully.")