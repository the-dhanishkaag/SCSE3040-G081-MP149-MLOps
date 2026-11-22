
from pathlib import Path

import pandas as pd
import yaml


# Load project configuration
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

raw_dir = Path(params["data"]["raw_dir"])
processed_dir = Path(params["data"]["processed_dir"])

cleaning = params["cleaning"]

# Load raw datasets
train = pd.read_csv(raw_dir / "train.csv")
later = pd.read_csv(raw_dir / "later.csv")
holdout = pd.read_csv(raw_dir / "holdout_features.csv")


# Display dataset information
print("Raw datasets loaded successfully!")
print("Training data:", train.shape)
print("Later data:", later.shape)
print("Holdout data:", holdout.shape)

# Remove duplicate rows from training data
if cleaning["remove_duplicates"]:
    original_rows = len(train)
    train = train.drop_duplicates().copy()
    removed_rows = original_rows - len(train)

    print("Duplicate rows removed:", removed_rows)
    print("Training rows after cleaning:", len(train))

# Correct target values with extra zero
target = cleaning["target_correction"]
target_column = target["column"]
reference_column = target["reference_column"]

ratio = train[target_column] / train[reference_column]

typo_mask = (
    (ratio >= target["ratio_min"]) &
    (ratio <= target["ratio_max"])
)

corrected_count = typo_mask.sum()

train.loc[typo_mask, target_column] = (
    train.loc[typo_mask, target_column]
    / target["correction_factor"]
)

print("Target typos corrected:", corrected_count)


# Replace invalid sentinel values
sentinel = cleaning["sentinel"]
sentinel_column = sentinel["column"]
sentinel_value = sentinel["value"]
replacement_value = sentinel["replacement"]

for name, df in [
    ("train", train),
    ("later", later),
    ("holdout", holdout)
]:
    sentinel_count = (df[sentinel_column] == sentinel_value).sum()

    df[sentinel_column] = df[sentinel_column].replace(
        sentinel_value,
        replacement_value
    )

    print(name, "sentinel values replaced:", sentinel_count)

# Standardize boat names
boat_mapping = {
    "trawler": "trawler",
    "tra.": "trawler",
    "country boat": "country_boat",
    "country_boat": "country_boat",
    "cou.": "country_boat",
    "motorised": "motorised",
    "mot.": "motorised"
}


for name, df in [
    ("train", train),
    ("later", later),
    ("holdout", holdout)
]:
    df["boat"] = (
        df["boat"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(boat_mapping)
    )

    print(name, "boat categories:", sorted(df["boat"].dropna().unique()))

# Validate boat categories
for name, df in [
    ("train", train),
    ("later", later),
    ("holdout", holdout)
]:
    if df["boat"].isna().any():
        raise ValueError(f"Unknown boat category found in {name}")

print("Boat category validation passed!")


# Remove data leakage columns
leakage_columns = cleaning["leakage_columns"]

train = train.drop(columns=leakage_columns, errors="ignore")
later = later.drop(columns=leakage_columns, errors="ignore")
holdout = holdout.drop(columns=leakage_columns, errors="ignore")

print("Leakage columns removed:", leakage_columns)
print("Training columns:", train.columns.tolist())
print("Later columns:", later.columns.tolist())
print("Holdout columns:", holdout.columns.tolist())
# Final data validation
datasets = {
    "train": train,
    "later": later,
    "holdout": holdout
}

for name, df in datasets.items():
    assert df.isna().sum().sum() == 0, f"Missing values in {name}"
    assert df.duplicated().sum() == 0, f"Duplicates in {name}"
    assert not (df["net_length_m"] == -999).any(), f"Sentinel values in {name}"
    assert "kg_sold_at_auction" not in df.columns, f"Leakage in {name}"

    print(name, "final validation passed!")

assert "catch_kg" in train.columns
assert "catch_kg" in later.columns
assert "catch_kg" not in holdout.columns

print("All final validation checks passed!")


# Save processed datasets
processed_dir.mkdir(parents=True, exist_ok=True)

train.to_csv(processed_dir / "train_clean.csv", index=False)
later.to_csv(processed_dir / "later_clean.csv", index=False)
holdout.to_csv(processed_dir / "holdout_clean.csv", index=False)

print("Cleaned datasets saved successfully!")

