from pathlib import Path
import pandas as pd


# ============================================================
# CGMacros - Load and Combine All Participant Data
# ============================================================

# Project locations
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CGMACROS_DIR = DATA_DIR / "CGMacros"

OUTPUT_FILE = DATA_DIR / "combined_cgmacros.csv"


print("=" * 70)
print("CGMacros DATA LOADING")
print("=" * 70)

# ------------------------------------------------------------
# 1. Find participant folders
# ------------------------------------------------------------

participant_dirs = sorted(
    [
        folder
        for folder in CGMACROS_DIR.iterdir()
        if folder.is_dir() and folder.name.startswith("CGMacros-")
    ]
)

print(f"\nParticipant folders found: {len(participant_dirs)}")

if len(participant_dirs) != 45:
    print("WARNING: Expected 45 participants.")

# ------------------------------------------------------------
# 2. Load each participant CSV
# ------------------------------------------------------------

all_data = []

for participant_dir in participant_dirs:

    participant_id = participant_dir.name

    csv_file = participant_dir / f"{participant_id}.csv"

    if not csv_file.exists():
        print(f"WARNING: CSV not found: {csv_file}")
        continue

    print(f"Loading {participant_id}...")

    df = pd.read_csv(csv_file)

    # Add participant identifier
    df["Participant_ID"] = participant_id

    all_data.append(df)

# ------------------------------------------------------------
# 3. Combine all participants
# ------------------------------------------------------------

if not all_data:
    raise RuntimeError("No participant CSV files were loaded.")

combined_df = pd.concat(
    all_data,
    ignore_index=True
)

# ------------------------------------------------------------
# 4. Clean column names
# ------------------------------------------------------------

combined_df.columns = (
    combined_df.columns
    .str.strip()
)

# ------------------------------------------------------------
# 5. Convert timestamp
# ------------------------------------------------------------

combined_df["Timestamp"] = pd.to_datetime(
    combined_df["Timestamp"],
    errors="coerce"
)

# ------------------------------------------------------------
# 6. Basic information
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("COMBINED DATASET")
print("=" * 70)

print(f"\nRows: {len(combined_df):,}")
print(f"Columns: {len(combined_df.columns)}")

print("\nColumns:")
for column in combined_df.columns:
    print(f"  - {column}")

print("\nParticipant count:")
print(combined_df["Participant_ID"].nunique())

print("\nRows per participant:")
print(
    combined_df["Participant_ID"]
    .value_counts()
    .sort_index()
    .to_string()
)

# ------------------------------------------------------------
# 7. Meal information
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MEAL INFORMATION")
print("=" * 70)

meal_rows = combined_df[
    combined_df["Meal Type"].notna()
]

print(f"\nRows containing Meal Type: {len(meal_rows):,}")

print("\nMeal types:")
print(
    meal_rows["Meal Type"]
    .value_counts(dropna=False)
    .to_string()
)

# ------------------------------------------------------------
# 8. Missing values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = combined_df.isna().sum()

missing_summary = pd.DataFrame({
    "Missing": missing,
    "Missing_Percent": (
        missing / len(combined_df) * 100
    ).round(2)
})

print(missing_summary.to_string())

# ------------------------------------------------------------
# 9. Save combined dataset
# ------------------------------------------------------------

combined_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(f"\nOutput file:")
print(OUTPUT_FILE)

print("\nCGMacros loading completed successfully.")