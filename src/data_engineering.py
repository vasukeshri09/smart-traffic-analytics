
"""Bronze-Silver-Gold data engineering pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
BRONZE = ROOT / "data" / "bronze"
SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold"

LOW_THRESHOLD = 0.45
HIGH_THRESHOLD = 0.72


def create_directories():
    for folder in [BRONZE, SILVER, GOLD]:
        folder.mkdir(parents=True, exist_ok=True)


def bronze_layer():
    """Preserve a copy of all original datasets."""

    print("\n[BRONZE] Copying raw datasets...")

    files = [
        "segments.csv",
        "weather.csv",
        "traffic.csv",
        "accidents.csv",
    ]

    for filename in files:
        df = pd.read_csv(RAW / filename)
        df.to_parquet(BRONZE / filename.replace(".csv", ".parquet"),
                      index=False)
        print(f"Saved: {filename}")

    print("[BRONZE] Completed.")


def silver_layer():
    """Clean and validate traffic, weather and accident data."""

    print("\n[SILVER] Cleaning datasets...")

    segments = pd.read_csv(RAW / "segments.csv")
    traffic = pd.read_csv(RAW / "traffic.csv")
    weather = pd.read_csv(RAW / "weather.csv")
    accidents = pd.read_csv(RAW / "accidents.csv")

    for df in [traffic, weather, accidents]:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Remove duplicate traffic observations.
    traffic = traffic.drop_duplicates(
        subset=["timestamp", "segment_id"]
    ).copy()

    # Attach segment-specific speed limits.
    traffic = traffic.merge(
        segments[["segment_id", "free_flow_speed"]],
        on="segment_id",
        how="left",
        validate="many_to_one"
    )

    # Reject impossible sensor speeds.
    invalid_speed = (
        (traffic["speed"] <= 0)
        | (traffic["speed"] > traffic["free_flow_speed"] * 1.2)
    )

    traffic.loc[invalid_speed, "speed"] = np.nan

    # Interpolate missing traffic values separately for each segment.
    traffic = traffic.sort_values(
        ["segment_id", "timestamp"]
    )

    for column in ["speed", "density", "volume", "occupancy"]:

        traffic[column] = traffic.groupby("segment_id")[column].transform(
            lambda series: series.interpolate(
                method="linear",
                limit_direction="both"
            )
        )

        traffic[column] = traffic[column].fillna(
            traffic.groupby("segment_id")[column].transform("median")
        )

    # Keep values within physically plausible ranges.
    traffic["speed"] = traffic["speed"].clip(lower=1)
    traffic["density"] = traffic["density"].clip(lower=0)
    traffic["volume"] = traffic["volume"].clip(lower=0)
    traffic["occupancy"] = traffic["occupancy"].clip(0, 100)

    # Clean weather values.
    weather_ranges = {
        "temperature": (-10, 60),
        "humidity": (0, 100),
        "rainfall": (0, 200),
        "visibility": (0, 30),
    }

    for column, (lower, upper) in weather_ranges.items():
        weather[column] = pd.to_numeric(
            weather[column], errors="coerce"
        ).clip(lower, upper)

        weather[column] = weather[column].interpolate(
            limit_direction="both"
        )

    # Accident data validation.
    valid_segments = set(segments["segment_id"])

    accidents = accidents[
        accidents["segment_id"].isin(valid_segments)
    ].copy()

    # Save cleaned data.
    segments.to_parquet(SILVER / "segments.parquet", index=False)
    traffic.to_parquet(SILVER / "traffic.parquet", index=False)
    weather.to_parquet(SILVER / "weather.parquet", index=False)
    accidents.to_parquet(SILVER / "accidents.parquet", index=False)

    print("Clean traffic rows:", len(traffic))
    print("Weather rows:", len(weather))
    print("Accident rows:", len(accidents))
    print("[SILVER] Completed.")


def gold_layer():
    """Create leakage-safe features and one-hour-ahead targets."""

    print("\n[GOLD] Creating ML features...")

    traffic = pd.read_parquet(SILVER / "traffic.parquet")
    weather = pd.read_parquet(SILVER / "weather.parquet")
    segments = pd.read_parquet(SILVER / "segments.parquet")
    accidents = pd.read_parquet(SILVER / "accidents.parquet")

    traffic["timestamp"] = pd.to_datetime(traffic["timestamp"])
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    accidents["timestamp"] = pd.to_datetime(accidents["timestamp"])

    # Join current traffic and weather observations.
    data = traffic.merge(
        weather,
        on="timestamp",
        how="left",
        validate="many_to_one"
    )

    data = data.merge(
    segments[
        [
            "segment_id",
            "road_type",
            "lat",
            "lon",
            "lanes",
            "length_km",
            "is_blackspot_area",
        ]
    ],
    on="segment_id",
    how="left",
    validate="many_to_one"
)

    # Accident count for each segment-hour.
    accident_counts = (
        accidents.groupby(["segment_id", "timestamp"])
        .size()
        .rename("accident_count")
        .reset_index()
    )

    data = data.merge(
        accident_counts,
        on=["segment_id", "timestamp"],
        how="left"
    )

    data["accident_count"] = (
        data["accident_count"].fillna(0).astype(int)
    )

    data = data.sort_values(
        ["segment_id", "timestamp"]
    ).reset_index(drop=True)

    grouped = data.groupby("segment_id", sort=False)

    # Time features.
    data["hour"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["month"] = data["timestamp"].dt.month
    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    data["is_morning_peak"] = data["hour"].between(
        7, 10
    ).astype(int)

    data["is_evening_peak"] = data["hour"].between(
        16, 20
    ).astype(int)

    # Cyclical time features.
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

    data["day_sin"] = np.sin(
        2 * np.pi * data["day_of_week"] / 7
    )
    data["day_cos"] = np.cos(
        2 * np.pi * data["day_of_week"] / 7
    )

    # Historical traffic features.
    data["speed_lag_1"] = grouped["speed"].shift(1)
    data["speed_lag_2"] = grouped["speed"].shift(2)
    data["density_lag_1"] = grouped["density"].shift(1)
    data["volume_lag_1"] = grouped["volume"].shift(1)

    data["speed_rolling_3"] = grouped["speed"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=3).mean()
    )

    data["density_rolling_3"] = grouped["density"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=3).mean()
    )

    data["accident_count_lag_1"] = grouped[
        "accident_count"
    ].shift(1)

    # Next-hour congestion target.
    data["next_speed"] = grouped["speed"].shift(-1)

    data["next_speed_ratio"] = (
        data["next_speed"] / data["free_flow_speed"]
    )

    data["congestion_target"] = np.select(
        [
            data["next_speed_ratio"] >= HIGH_THRESHOLD,
            data["next_speed_ratio"] >= LOW_THRESHOLD,
        ],
        [
            "Low",
            "Medium",
        ],
        default="High"
    )

    # Next-hour accident risk target.
    data["accident_target"] = (
        grouped["accident_count"].shift(-1).fillna(0) > 0
    ).astype(int)

    # Remove rows without sufficient history or future targets.
    required = [
        "speed_lag_2",
        "speed_rolling_3",
        "density_rolling_3",
        "next_speed",
    ]

    data = data.dropna(subset=required).copy()

    # Do not use future information as model input.
    # next_speed and next_speed_ratio are retained only as
    # evaluation/reference columns, not as predictors.

    data.to_parquet(GOLD / "model_dataset.parquet", index=False)
    data.to_csv(GOLD / "model_dataset.csv", index=False)

    print("Gold dataset rows:", len(data))
    print("Gold dataset columns:", len(data.columns))

    print("\nCongestion target distribution:")
    print(data["congestion_target"].value_counts())

    print("\nAccident target distribution:")
    print(data["accident_target"].value_counts())

    print("[GOLD] Completed.")


def main():
    create_directories()

    print("=" * 55)
    print("TRAFFIC DATA ENGINEERING PIPELINE")
    print("=" * 55)

    bronze_layer()
    silver_layer()
    gold_layer()

    print("\nALL DATA ENGINEERING STAGES COMPLETED")
    print("Output directory:", GOLD)


if __name__ == "__main__":
    main()
