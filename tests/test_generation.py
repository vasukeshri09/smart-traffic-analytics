
"""Tests for the synthetic traffic data generation pipeline."""

from pathlib import Path

import pandas as pd

from config import DATA_RAW


DATA_DIR = Path(DATA_RAW)


def load_dataset(filename):
    """Load a generated CSV file."""
    file_path = DATA_DIR / filename

    assert file_path.exists(), f"Missing dataset: {file_path}"

    return pd.read_csv(file_path)


def test_all_datasets_exist():
    """Check that all four CSV files were generated."""

    required_files = [
        "segments.csv",
        "weather.csv",
        "traffic.csv",
        "accidents.csv",
    ]

    for filename in required_files:
        assert (DATA_DIR / filename).exists()


def test_segments_dataset():
    """Verify road segment count and road type distribution."""

    segments = load_dataset("segments.csv")

    assert len(segments) == 60
    assert segments["segment_id"].nunique() == 60

    expected_distribution = {
        "highway": 10,
        "arterial": 20,
        "junction": 15,
        "local": 15,
    }

    actual_distribution = segments["road_type"].value_counts().to_dict()

    assert actual_distribution == expected_distribution

    assert segments["lat"].notna().all()
    assert segments["lon"].notna().all()


def test_weather_dataset():
    """Verify hourly weather observations."""

    weather = load_dataset("weather.csv")

    assert len(weather) == 2184
    assert weather["timestamp"].nunique() == 2184

    required_columns = [
        "timestamp",
        "temperature",
        "humidity",
        "rainfall",
        "visibility",
    ]

    assert all(col in weather.columns for col in required_columns)

    assert weather["temperature"].notna().all()
    assert weather["humidity"].between(0, 100).all()
    assert (weather["rainfall"] >= 0).all()


def test_traffic_dataset():
    """Verify traffic observations and injected quality issues."""

    traffic = load_dataset("traffic.csv")

    # 60 segments x 2184 hours, plus injected duplicates.
    assert len(traffic) == 131302

    assert traffic["segment_id"].nunique() == 60

    # Check injected sensor issues.
    assert (traffic["speed"] == 0).sum() == 40
    assert (traffic["speed"] == 999).sum() == 25
    assert traffic["speed"].isna().sum() == 80
    assert traffic["density"].isna().sum() == 80

    # Check duplicated records.
    duplicate_count = traffic.duplicated().sum()
    assert duplicate_count == 262


def test_accidents_dataset():
    """Verify accident events and their segment references."""

    accidents = load_dataset("accidents.csv")
    traffic = load_dataset("traffic.csv")
    segments = load_dataset("segments.csv")

    assert len(accidents) > 0

    assert accidents["accident_id"].is_unique

    valid_segment_ids = set(segments["segment_id"])

    assert set(accidents["segment_id"]).issubset(valid_segment_ids)

    accident_rate = len(accidents) / (60 * 2184)

    assert 0.065 <= accident_rate <= 0.075

    assert accidents["severity"].isin(
        ["minor", "major", "fatal"]
    ).all()

    assert accidents["accident_type"].notna().all()


def test_dataset_date_range():
    """Verify that generated data covers the required period."""

    traffic = load_dataset("traffic.csv")
    weather = load_dataset("weather.csv")

    traffic_dates = pd.to_datetime(traffic["timestamp"])
    weather_dates = pd.to_datetime(weather["timestamp"])

    assert traffic_dates.min() == pd.Timestamp("2024-01-01 00:00:00")
    assert traffic_dates.max() == pd.Timestamp("2024-03-31 23:00:00")

    assert weather_dates.min() == pd.Timestamp("2024-01-01 00:00:00")
    assert weather_dates.max() == pd.Timestamp("2024-03-31 23:00:00")
