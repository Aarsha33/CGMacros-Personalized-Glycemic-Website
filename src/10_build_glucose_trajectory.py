from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "combined_cgmacros.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "glucose_trajectory_dataset.csv"


# ============================================================
# SETTINGS
# ============================================================

TIME_POINTS = [0, 30, 60, 90, 120]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("BUILDING GLUCOSE TRAJECTORY DATASET")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    errors="coerce"
)

print(f"Raw rows: {len(df):,}")


# ============================================================
# NORMALIZE MEAL TYPES
# ============================================================

df["Meal Type"] = (
    df["Meal Type"]
    .astype(str)
    .str.strip()
    .str.lower()
)

meal_type_map = {
    "breakfast": "breakfast",
    "lunch": "lunch",
    "dinner": "dinner",
    "snack": "snack"
}

df["Meal_Type"] = df["Meal Type"].map(meal_type_map)


# ============================================================
# KEEP VALID MEAL ROWS
# ============================================================

meal_rows = df[
    df["Meal_Type"].notna()
    & df["Timestamp"].notna()
    & df["Libre GL"].notna()
].copy()

print(f"Valid meal-related rows: {len(meal_rows):,}")


# ============================================================
# GET UNIQUE MEALS
# ============================================================

meals = (
    meal_rows[
        [
            "Participant_ID",
            "Timestamp",
            "Meal_Type",
            "Carbs",
            "Protein"
        ]
    ]
    .drop_duplicates()
    .sort_values(["Participant_ID", "Timestamp"])
    .reset_index(drop=True)
)

print(f"Unique meals found: {len(meals):,}")


# ============================================================
# BUILD TRAJECTORY
# ============================================================

records = []

for _, meal in meals.iterrows():

    participant = meal["Participant_ID"]
    meal_time = meal["Timestamp"]

    # Participant CGM data
    participant_data = df[
        (df["Participant_ID"] == participant)
        & df["Timestamp"].notna()
        & df["Libre GL"].notna()
    ].copy()

    # Time difference from meal
    participant_data["minutes_after_meal"] = (
        participant_data["Timestamp"] - meal_time
    ).dt.total_seconds() / 60.0

    for target_time in TIME_POINTS:

        # Find glucose reading closest to target time
        candidates = participant_data[
            participant_data["minutes_after_meal"].between(
                target_time - 10,
                target_time + 10
            )
        ].copy()

        if candidates.empty:
            glucose = np.nan
        else:
            candidates["time_distance"] = (
                candidates["minutes_after_meal"] - target_time
            ).abs()

            closest = candidates.sort_values(
                "time_distance"
            ).iloc[0]

            glucose = float(closest["Libre GL"])

        records.append({
            "Participant_ID": participant,
            "Meal_Timestamp": meal_time,
            "Meal_Type": meal["Meal_Type"],
            "Carbs": meal["Carbs"],
            "Protein": meal["Protein"],
            "Time_After_Meal": target_time,
            "Glucose": glucose
        })


# ============================================================
# CREATE DATAFRAME
# ============================================================

trajectory = pd.DataFrame(records)


# ============================================================
# REMOVE INCOMPLETE TRAJECTORIES
# ============================================================

trajectory_counts = (
    trajectory
    .groupby(
        [
            "Participant_ID",
            "Meal_Timestamp"
        ]
    )["Glucose"]
    .count()
)

complete_meals = trajectory_counts[
    trajectory_counts == len(TIME_POINTS)
].index


trajectory = trajectory.set_index(
    ["Participant_ID", "Meal_Timestamp"]
)

trajectory = trajectory.loc[
    trajectory.index.isin(complete_meals)
].reset_index()


# ============================================================
# SAVE
# ============================================================

trajectory.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("TRAJECTORY DATASET CREATED")
print("=" * 70)

print(f"Output file:")
print(OUTPUT_FILE)

print(f"Rows: {len(trajectory):,}")

print(
    f"Complete meal trajectories: "
    f"{trajectory[['Participant_ID', 'Meal_Timestamp']].drop_duplicates().shape[0]:,}"
)

print()
print("Time points:")
print(sorted(trajectory["Time_After_Meal"].unique()))

print()
print("Missing glucose values:")
print(trajectory["Glucose"].isna().sum())

print()
print("First rows:")
print(trajectory.head(10).to_string(index=False))

print()
print("=" * 70)