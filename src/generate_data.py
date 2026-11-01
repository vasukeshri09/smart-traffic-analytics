
"""Generate synthetic Delhi traffic, weather, and accident datasets."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from config import (
    DATA_RAW,
    START_DATE,
    END_DATE,
    SEED,
    ACCIDENT_RATE_TARGET,
    DENSITY_MIN,
    DENSITY_MAX,
    VISIBILITY_MIN,
    VISIBILITY_MAX,
    ensure_dirs,
)

from src.delhi_network import get_delhi_road_segments_df


# ---------------------------------------------------------
# 1. INITIALIZATION
# ---------------------------------------------------------

rng = np.random.default_rng(SEED)

OUTPUT_DIR = Path(DATA_RAW)


# ---------------------------------------------------------
# 2. GENERATE HOURLY WEATHER DATA
# ---------------------------------------------------------

def generate_weather(timestamps):
    """Generate hourly synthetic weather observations."""

    n = len(timestamps)

    hours = timestamps.hour.to_numpy()
    days = timestamps.dayofyear.to_numpy()

    # Seasonal temperature pattern for Delhi.
    seasonal_temp = (
        25
        + 10 * np.sin(2 * np.pi * (days - 100) / 365)
    )

    daily_temp = 4 * np.sin(
        2 * np.pi * (hours - 8) / 24
    )

    temperature = (
        seasonal_temp
        + daily_temp
        + rng.normal(0, 1.5, n)
    )

    # Generate rainfall using a simple two-state Markov process.
    rainfall = np.zeros(n)
    is_raining = False

    for i in range(n):
        if is_raining:
            is_raining = rng.random() < 0.75
        else:
            is_raining = rng.random() < 0.08

        if is_raining:
            rainfall[i] = rng.exponential(1.5)

    humidity = (
        45
        + rainfall * 8
        + rng.normal(0, 10, n)
    )

    humidity = np.clip(humidity, 20, 100)

    visibility = (
        10
        - rainfall * 0.8
        - np.maximum(0, humidity - 75) * 0.05
        + rng.normal(0, 1, n)
    )

    visibility = np.clip(
        visibility,
        VISIBILITY_MIN,
        VISIBILITY_MAX
    )

    weather = pd.DataFrame({
        "timestamp": timestamps,
        "temperature": np.round(temperature, 2),
        "humidity": np.round(humidity, 2),
        "rainfall": np.round(rainfall, 2),
        "visibility": np.round(visibility, 2),
    })

    return weather


# ---------------------------------------------------------
# 3. GENERATE TRAFFIC DATA
# ---------------------------------------------------------

def generate_traffic(segments, weather):
    """Generate hourly traffic observations for all segments."""

    timestamps = weather["timestamp"].to_numpy()
    weather_lookup = weather.set_index("timestamp")

    traffic_rows = []

    for segment in segments.itertuples(index=False):

        road_type = segment.road_type
        free_flow = segment.free_flow_speed
        lanes = segment.lanes

        # Road-specific maximum traffic density.
        capacity = {
            "highway": 140,
            "arterial": 110,
            "junction": 90,
            "local": 70,
        }[road_type]

        # Base density differs by road type.
        base_density = {
            "highway": 35,
            "arterial": 45,
            "junction": 55,
            "local": 30,
        }[road_type]

        previous_density = base_density

        for timestamp in timestamps:

            timestamp = pd.Timestamp(timestamp)
            hour = timestamp.hour
            weekday = timestamp.dayofweek

            weather_row = weather_lookup.loc[timestamp]

            # Morning and evening traffic peaks.
            morning_peak = np.exp(
                -((hour - 9) ** 2) / 8
            )

            evening_peak = np.exp(
                -((hour - 18) ** 2) / 10
            )

            peak_factor = morning_peak + evening_peak

            # Weekday traffic is generally higher.
            weekday_factor = 1.0 if weekday < 5 else 0.75

            expected_density = (
                base_density
                + 35 * peak_factor * weekday_factor
                + 3 * weather_row["rainfall"]
            )

            # AR(1)-style density variation.
            density = (
                0.65 * previous_density
                + 0.35 * expected_density
                + rng.normal(0, 7)
            )

            density = float(np.clip(
                density,
                DENSITY_MIN,
                min(DENSITY_MAX, capacity)
            ))

            previous_density = density

            # Greenshields-style speed-density relationship.
            density_ratio = density / capacity

            speed = free_flow * max(
                0.05,
                1 - density_ratio
            )

            # Weather reduces driving speed.
            speed *= max(
                0.55,
                1 - 0.025 * weather_row["rainfall"]
            )

            if weather_row["visibility"] < 3:
                speed *= 0.85

            # Add random sensor and traffic variation.
            speed += rng.normal(0, 4)

            speed = float(np.clip(speed, 2, free_flow * 1.15))

            # Approximate traffic volume.
            volume = density * speed * lanes * 0.35

            occupancy = np.clip(
                density_ratio * 100 + rng.normal(0, 3),
                0,
                100
            )

            traffic_rows.append({
                "timestamp": timestamp,
                "segment_id": segment.segment_id,
                "density": round(density, 2),
                "speed": round(speed, 2),
                "volume": round(volume, 2),
                "occupancy": round(float(occupancy), 2),
            })

    return pd.DataFrame(traffic_rows)


# ---------------------------------------------------------
# 4. GENERATE ACCIDENT EVENTS
# ---------------------------------------------------------

def generate_accidents(traffic, segments, weather):
    """Generate accident events with a calibrated average rate."""

    segment_lookup = segments.set_index("segment_id")
    weather_lookup = weather.set_index("timestamp")

    accident_data = traffic.copy()

    accident_data = accident_data.merge(
        segments[["segment_id", "road_type", "lat", "lon"]],
        on="segment_id",
        how="left"
    )

    accident_data = accident_data.merge(
        weather,
        on="timestamp",
        how="left"
    )

    # Higher density and rainfall increase simulated risk.
    risk_score = (
        0.035 * accident_data["density"]
        - 0.025 * accident_data["speed"]
        + 0.20 * accident_data["rainfall"]
        + 0.02 * accident_data["humidity"]
        + accident_data["road_type"].map({
            "highway": 0.25,
            "arterial": 0.15,
            "junction": 0.45,
            "local": 0.05,
        })
    )

    # Calibrate the intercept so the average event probability
    # is approximately ACCIDENT_RATE_TARGET.
    def probability_difference(intercept):
        probabilities = 1 / (
            1 + np.exp(-(intercept + risk_score.to_numpy()))
        )

        return probabilities.mean() - ACCIDENT_RATE_TARGET

    intercept = brentq(
        probability_difference,
        -30,
        30
    )

    probabilities = 1 / (
        1 + np.exp(-(intercept + risk_score.to_numpy()))
    )

    accident_mask = rng.random(len(accident_data)) < probabilities

    accident_rows = accident_data.loc[accident_mask].copy()

    if accident_rows.empty:
        return pd.DataFrame(columns=[
            "accident_id",
            "timestamp",
            "segment_id",
            "lat",
            "lon",
            "severity",
            "accident_type",
        ])

    severity = rng.choice(
        ["minor", "major", "fatal"],
        size=len(accident_rows),
        p=[0.65, 0.28, 0.07]
    )

    accident_type = rng.choice(
        [
            "collision",
            "rear_end",
            "vehicle_skid",
            "pedestrian",
            "other",
        ],
        size=len(accident_rows),
        p=[0.35, 0.30, 0.12, 0.13, 0.10]
    )

    accidents = pd.DataFrame({
        "accident_id": [
            f"ACC{i:06d}"
            for i in range(1, len(accident_rows) + 1)
        ],
        "timestamp": accident_rows["timestamp"].to_numpy(),
        "segment_id": accident_rows["segment_id"].to_numpy(),
        "lat": accident_rows["lat"].to_numpy(),
        "lon": accident_rows["lon"].to_numpy(),
        "severity": severity,
        "accident_type": accident_type,
    })

    return accidents


# ---------------------------------------------------------
# 5. INTRODUCE REALISTIC SENSOR DATA QUALITY ISSUES
# ---------------------------------------------------------


def introduce_sensor_errors(traffic):
    """Inject missing values, invalid speeds and duplicate records."""

    traffic = traffic.copy()

    # Invalid zero-speed readings.
    zero_indices = rng.choice(
        traffic.index.to_numpy(),
        size=40,
        replace=False
    )

    traffic.loc[zero_indices, "speed"] = 0

    # Impossible speed spikes.
    remaining_indices = np.setdiff1d(
        traffic.index.to_numpy(),
        zero_indices
    )

    spike_indices = rng.choice(
        remaining_indices,
        size=25,
        replace=False
    )

    traffic.loc[spike_indices, "speed"] = 999

    # Missing speed values.
    excluded_indices = np.concatenate([
        zero_indices,
        spike_indices
    ])

    remaining_indices = np.setdiff1d(
        traffic.index.to_numpy(),
        excluded_indices
    )

    missing_speed_indices = rng.choice(
        remaining_indices,
        size=80,
        replace=False
    )

    traffic.loc[missing_speed_indices, "speed"] = np.nan

    # Missing density values.
    missing_density_indices = rng.choice(
        traffic.index.to_numpy(),
        size=80,
        replace=False
    )

    traffic.loc[missing_density_indices, "density"] = np.nan

    # Add duplicate records.
    duplicate_count = max(1, int(len(traffic) * 0.002))

    duplicates = traffic.sample(
        n=duplicate_count,
        random_state=SEED
    )

    traffic = pd.concat(
        [traffic, duplicates],
        ignore_index=True
    )

    return traffic



# ---------------------------------------------------------
# 6. MAIN DATA GENERATION PIPELINE
# ---------------------------------------------------------

def main():

    ensure_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nStarting synthetic data generation...")
    print("-------------------------------------")

    timestamps = pd.date_range(
        start=START_DATE,
        end=END_DATE,
        freq="h"
    )

    print(f"Hourly timestamps: {len(timestamps)}")

    # Generate road network.
    segments = get_delhi_road_segments_df()

    # Generate weather.
    print("\nGenerating weather data...")
    weather = generate_weather(timestamps)

    # Generate clean traffic.
    print("Generating traffic data...")
    traffic = generate_traffic(segments, weather)

    # Generate accidents from clean traffic.
    print("Generating accident events...")
    accidents = generate_accidents(
        traffic,
        segments,
        weather
    )

    # Introduce sensor issues only after accident generation.
    print("Introducing sensor data quality issues...")
    traffic_raw = introduce_sensor_errors(traffic)

    # Save all datasets.
    print("\nSaving datasets...")

    segments.to_csv(
        OUTPUT_DIR / "segments.csv",
        index=False
    )

    weather.to_csv(
        OUTPUT_DIR / "weather.csv",
        index=False
    )

    traffic_raw.to_csv(
        OUTPUT_DIR / "traffic.csv",
        index=False
    )

    accidents.to_csv(
        OUTPUT_DIR / "accidents.csv",
        index=False
    )

    print("\nDATA GENERATION COMPLETED")
    print("-------------------------")

    print(f"Segments:  {len(segments):,}")
    print(f"Weather:   {len(weather):,}")
    print(f"Traffic:   {len(traffic_raw):,}")
    print(f"Accidents: {len(accidents):,}")

    print("\nTraffic road type distribution:")
    print(segments["road_type"].value_counts())

    print("\nAccident rate in clean segment-hours:")
    print(f"{len(accidents) / len(traffic) * 100:.2f}%")

    print("\nFiles saved in:")
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
