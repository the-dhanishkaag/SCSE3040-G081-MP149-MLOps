import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")

def test_cleaned_datasets_exist():
    assert (DATA_DIR / "train_clean.csv").exists()
    assert (DATA_DIR / "later_clean.csv").exists()
    assert (DATA_DIR / "holdout_clean.csv").exists()

def test_training_data_valid():
    df = pd.read_csv(DATA_DIR / "train_clean.csv")
    assert not df.empty
    assert "catch_kg" in df.columns
    assert df.isnull().sum().sum() == 0

def test_later_data_valid():
    df = pd.read_csv(DATA_DIR / "later_clean.csv")
    assert not df.empty
    assert "catch_kg" in df.columns
    assert df.isnull().sum().sum() == 0

def test_holdout_data_valid():
    df = pd.read_csv(DATA_DIR / "holdout_clean.csv")
    assert not df.empty
    assert "catch_kg" not in df.columns
    assert df.isnull().sum().sum() == 0

def test_no_data_leakage_column():
    for filename in ["train_clean.csv", "later_clean.csv", "holdout_clean.csv"]:
        df = pd.read_csv(DATA_DIR / filename)
        assert "kg_sold_at_auction" not in df.columns
def test_no_duplicates_in_cleaned_training_data():
    df = pd.read_csv(DATA_DIR / "train_clean.csv")
    assert df.duplicated().sum() == 0


def test_sentinel_value_replaced():
    for filename in ["train_clean.csv", "later_clean.csv", "holdout_clean.csv"]:
        df = pd.read_csv(DATA_DIR / filename)
        assert not (df["net_length_m"] == -999).any()


def test_boat_categories_standardized():
    allowed = {"trawler", "country_boat", "motorised"}

    for filename in ["train_clean.csv", "later_clean.csv", "holdout_clean.csv"]:
        df = pd.read_csv(DATA_DIR / filename)
        assert set(df["boat"].unique()).issubset(allowed)