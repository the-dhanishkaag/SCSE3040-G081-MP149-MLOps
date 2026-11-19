import pandas as pd

train = pd.read_csv("data/raw/train.csv")
later = pd.read_csv("data/raw/later.csv")
holdout = pd.read_csv("data/raw/holdout_features.csv")

print("TRAIN SHAPE:", train.shape)
print("\nTRAIN COLUMNS:")
print(train.columns.tolist())

print("\nTRAIN DATA TYPES:")
print(train.dtypes)

print("\nMISSING VALUES:")
print(train.isnull().sum())

print("\nLATER SHAPE:", later.shape)
print("HOLDOUT SHAPE:", holdout.shape)

print("\nTRAIN SAMPLE:")
print(train.head())

print("\nTARGET SUMMARY:")
print(train["catch_kg"].describe())
print("\n--- DUPLICATE ROWS ---")
print("Exact duplicate rows:", train.duplicated().sum())

print("\n--- SENTINEL VALUE ---")
numeric_columns = train.select_dtypes(include="number").columns
print((train[numeric_columns] == -999).sum())
print("\n--- TARGET TYPOS ---")
unique_train = train.drop_duplicates()
typos = unique_train[unique_train["catch_kg"] > 1000].copy()
typos["ratio"] = typos["catch_kg"] / typos["kg_sold_at_auction"]

print("Possible 10x target typos:", len(typos))
print(typos[["record_id", "catch_kg", "kg_sold_at_auction", "ratio"]])
print("\n--- SPELLING VARIANTS ---")
print("Raw boat labels:")
print(train["boat"].value_counts().to_string())

print("\nCanonical boat labels in later data:")
print(later["boat"].value_counts().to_string())
print("\n--- STUCK SENSOR CHECK ---")

numeric_columns = [
    "trip_hours",
    "distance_from_shore_km",
    "crew",
    "wave_height_m",
    "water_temp_c",
    "net_length_m"
]

sorted_train = train.sort_values(["date", "record_id"])

for col in numeric_columns:
    values = sorted_train[col].tolist()

    if col == "net_length_m":
        values = [None if x == -999 else x for x in values]

    max_run = 1
    current_run = 1

    for i in range(1, len(values)):
        if values[i] is not None and values[i] == values[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1

    print(col, "longest repeated run:", max_run)