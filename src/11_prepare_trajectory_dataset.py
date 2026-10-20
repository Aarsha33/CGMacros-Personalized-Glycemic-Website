from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAJECTORY_FILE = (
    PROJECT_ROOT / "data" / "glucose_trajectory_dataset.csv"
)

BIO_FILE = (
    PROJECT_ROOT / "data" / "CGMacros" / "bio.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "glucose_trajectory_personalized.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PREPARING PERSONALIZED GLUCOSE TRAJECTORY DATASET")
print("=" * 70)

trajectory = pd.read_csv(TRAJECTORY_FILE)
bio = pd.read_csv(BIO_FILE)

print(f"Trajectory rows: {len(trajectory):,}")
print(f"Bio rows: {len(bio):,}")


# ============================================================
# CLEAN BIO COLUMN NAMES
# ============================================================

bio.columns = bio.columns.str.strip()

print("\nBio columns:")
print(bio.columns.tolist())


# ============================================================
# FIND REQUIRED BIO COLUMNS
# ============================================================

participant_col = "subject"
hba1c_col = "A1c PDL (Lab)"
fasting_col = "Fasting GLU - PDL (Lab)"


required_columns = [
    participant_col,
    hba1c_col,
    fasting_col
]

missing = [
    col for col in required_columns
    if col not in bio.columns
]

if missing:
    raise ValueError(
        f"Missing required bio columns: {missing}"
    )


# ============================================================
# PREPARE BIO DATA
# ============================================================

bio_small = bio[
    required_columns
].copy()

bio_small = bio_small.rename(
    columns={
        participant_col: "Participant_ID",
        hba1c_col: "HbA1c",
        fasting_col: "Fasting_Glucose"
    }
)


# ============================================================
# CLEAN PARTICIPANT IDs
# ============================================================

trajectory["Participant_ID"] = (
    trajectory["Participant_ID"]
    .astype(str)
    .str.strip()
)

bio_small["Participant_ID"] = (
    bio_small["Participant_ID"]
    .astype(int)
    .apply(lambda x: f"CGMacros-{x:03d}")
)

# ============================================================
# CONVERT NUMERIC VALUES
# ============================================================

bio_small["HbA1c"] = pd.to_numeric(
    bio_small["HbA1c"],
    errors="coerce"
)

bio_small["Fasting_Glucose"] = pd.to_numeric(
    bio_small["Fasting_Glucose"],
    errors="coerce"
)


# ============================================================
# REMOVE DUPLICATE PARTICIPANTS
# ============================================================

bio_small = (
    bio_small
    .drop_duplicates("Participant_ID")
)


# ============================================================
# MERGE
# ============================================================

merged = trajectory.merge(
    bio_small,
    on="Participant_ID",
    how="left",
    validate="many_to_one"
)


# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\nMissing values after merge:")

print(
    merged[
        [
            "HbA1c",
            "Fasting_Glucose"
        ]
    ]
    .isna()
    .sum()
)


# ============================================================
# REMOVE INCOMPLETE RECORDS
# ============================================================

merged = merged.dropna(
    subset=[
        "HbA1c",
        "Fasting_Glucose",
        "Glucose",
        "Carbs",
        "Protein"
    ]
).copy()


# ============================================================
# CREATE MEAL TIME FEATURES
# ============================================================

merged["Meal_Timestamp"] = pd.to_datetime(
    merged["Meal_Timestamp"],
    errors="coerce"
)

merged["Meal_Hour"] = (
    merged["Meal_Timestamp"].dt.hour
    + merged["Meal_Timestamp"].dt.minute / 60.0
)

import numpy as np

merged["Meal_Hour_Sin"] = np.sin(
    2 * np.pi * merged["Meal_Hour"] / 24
)

merged["Meal_Hour_Cos"] = np.cos(
    2 * np.pi * merged["Meal_Hour"] / 24
)


# ============================================================
# SELECT FINAL COLUMNS
# ============================================================

final_columns = [
    "Participant_ID",
    "Meal_Timestamp",
    "Meal_Type",
    "HbA1c",
    "Fasting_Glucose",
    "Carbs",
    "Protein",
    "Meal_Hour_Sin",
    "Meal_Hour_Cos",
    "Time_After_Meal",
    "Glucose"
]

merged = merged[final_columns]


# ============================================================
# SAVE
# ============================================================

merged.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PERSONALIZED TRAJECTORY DATASET CREATED")
print("=" * 70)

print(f"Output:")
print(OUTPUT_FILE)

print(f"Rows: {len(merged):,}")

print(
    "Unique meals:",
    merged[
        ["Participant_ID", "Meal_Timestamp"]
    ].drop_duplicates().shape[0]
)

print(
    "Unique participants:",
    merged["Participant_ID"].nunique()
)

print("\nFinal columns:")
print(merged.columns.tolist())

print("\nFirst rows:")
print(
    merged.head(10).to_string(index=False)
)

print("\nMissing values:")
print(
    merged.isna().sum()
)

print("=" * 70)