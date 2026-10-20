from pathlib import Path
import pandas as pd


# ============================================================
# CGMacros - Merge Participant Bio/Health Data
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MEAL_FILE = PROJECT_ROOT / "data" / "meal_iauc_dataset.csv"
BIO_FILE = PROJECT_ROOT / "data" / "CGMacros" / "bio.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "meal_personalized_dataset.csv"


print("=" * 70)
print("CGMacros PERSONALIZED DATASET CREATION")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load meal-level iAUC data
# ------------------------------------------------------------

print("\nLoading meal-level dataset...")

meal_df = pd.read_csv(MEAL_FILE)

meal_df["Meal_Timestamp"] = pd.to_datetime(
    meal_df["Meal_Timestamp"],
    errors="coerce"
)

print(f"Meal records: {len(meal_df):,}")
print(f"Participants: {meal_df['Participant_ID'].nunique()}")


# ------------------------------------------------------------
# 2. Load participant bio data
# ------------------------------------------------------------

print("\nLoading participant bio data...")

bio_df = pd.read_csv(BIO_FILE)

print(f"Bio records: {len(bio_df):,}")
print(f"Bio columns: {len(bio_df.columns)}")


# ------------------------------------------------------------
# 3. Clean column names
# ------------------------------------------------------------

bio_df.columns = (
    bio_df.columns
    .str.strip()
)


# ------------------------------------------------------------
# 4. Rename important bio columns
# ------------------------------------------------------------

bio_df = bio_df.rename(
    columns={
        "subject": "Subject",
        "Age": "Age",
        "Gender": "Gender",
        "BMI": "BMI",
        "Body weight": "Body_Weight",
        "Height": "Height",
        "A1c PDL (Lab)": "HbA1c",
        "Fasting GLU -PDL (Lab)": "Fasting_Glucose",
        "Fasting GLU - PDL (Lab)": "Fasting_Glucose",
    }
)


# ------------------------------------------------------------
# 5. Create matching participant ID
# ------------------------------------------------------------

bio_df["Participant_ID"] = (
    "CGMacros-"
    + bio_df["Subject"].astype(int).astype(str).str.zfill(3)
)


# ------------------------------------------------------------
# 6. Keep only variables needed for personalization
# ------------------------------------------------------------

bio_features = [
    "Participant_ID",
    "Age",
    "Gender",
    "BMI",
    "Body_Weight",
    "Height",
    "HbA1c",
    "Fasting_Glucose",
]

bio_clean = bio_df[bio_features].copy()


# ------------------------------------------------------------
# 7. Convert numeric variables
# ------------------------------------------------------------

numeric_columns = [
    "Age",
    "BMI",
    "Body_Weight",
    "Height",
    "HbA1c",
    "Fasting_Glucose",
]

for column in numeric_columns:

    bio_clean[column] = pd.to_numeric(
        bio_clean[column],
        errors="coerce"
    )


# ------------------------------------------------------------
# 8. Check duplicate participants
# ------------------------------------------------------------

duplicates = bio_clean[
    bio_clean["Participant_ID"].duplicated(
        keep=False
    )
]

print("\nDuplicate participant records:")

if duplicates.empty:
    print("None")
else:
    print(duplicates.to_string(index=False))


# ------------------------------------------------------------
# 9. Check missing values
# ------------------------------------------------------------

print("\nBio-data missing values:")

print(
    bio_clean.isna()
    .sum()
    .to_string()
)


# ------------------------------------------------------------
# 10. Merge meal data with participant data
# ------------------------------------------------------------

print("\nMerging participant information with meals...")

merged_df = meal_df.merge(
    bio_clean,
    on="Participant_ID",
    how="left",
    validate="many_to_one"
)


# ------------------------------------------------------------
# 11. Verify merge
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MERGE RESULTS")
print("=" * 70)

print(f"\nMeal records before merge: {len(meal_df):,}")
print(f"Meal records after merge:  {len(merged_df):,}")

print(
    f"Participants after merge: "
    f"{merged_df['Participant_ID'].nunique()}"
)

print("\nMissing personalized variables after merge:")

personalized_columns = [
    "Age",
    "Gender",
    "BMI",
    "Body_Weight",
    "Height",
    "HbA1c",
    "Fasting_Glucose",
]

print(
    merged_df[personalized_columns]
    .isna()
    .sum()
    .to_string()
)


# ------------------------------------------------------------
# 12. Display sample
# ------------------------------------------------------------

print("\nSample personalized meal records:")

sample_columns = [
    "Participant_ID",
    "Meal_Timestamp",
    "Meal_Type",
    "Carbs",
    "Protein",
    "Baseline_Glucose",
    "iAUC",
    "Age",
    "Gender",
    "BMI",
    "HbA1c",
    "Fasting_Glucose",
]

print(
    merged_df[sample_columns]
    .head(10)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 13. Save
# ------------------------------------------------------------

merged_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(f"\nOutput file:")
print(OUTPUT_FILE)

print("\nPersonalized meal dataset created successfully.")