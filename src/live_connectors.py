
"""Connect to Open-Meteo and TomTom for live Delhi observations."""

import os
from pathlib import Path
from datetime import datetime, timezone

import requests
import pandas as pd
from dotenv import load_dotenv


# ---------------------------------------------------------
# PROJECT CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
STREAM_DIR = ROOT / "data" / "stream_in"

# Load environment variables from project root.
load_dotenv(ROOT / ".env")

TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "").strip()

if TOMTOM_API_KEY:
    print("[CONFIG] TomTom API key loaded successfully.")
else:
    print("[CONFIG] TomTom API key not found.")

# Delhi reference location.
DELHI_LAT = 28.6139
DELHI_LON = 77.2090

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# TomTom Traffic Flow - Flow Segment Data API.
TOMTOM_URL = (
    "https://api.tomtom.com/traffic/services/4/"
    "flowSegmentData/absolute/10/json"
)


# ---------------------------------------------------------
# COMMON FUNCTIONS
# ---------------------------------------------------------

def get_timestamp():
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def save_csv(df, filename):
    """Append observations to a CSV file."""

    STREAM_DIR.mkdir(parents=True, exist_ok=True)

    output = STREAM_DIR / filename

    df.to_csv(
        output,
        mode="a",
        header=not output.exists(),
        index=False,
    )

    return output


def safe_error(error, api_key=None):
    """Prevent API keys from appearing in printed errors."""

    message = str(error)

    if api_key:
        message = message.replace(api_key, "***")

    return message


# ---------------------------------------------------------
# OPEN-METEO WEATHER CONNECTOR
# ---------------------------------------------------------

def fetch_weather():
    """Fetch current Delhi weather from Open-Meteo."""

    print("\n[OPEN-METEO] Fetching live weather...")

    params = {
        "latitude": DELHI_LAT,
        "longitude": DELHI_LON,
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "visibility,"
            "wind_speed_10m"
        ),
        "timezone": "Asia/Kolkata",
    }

    try:
        response = requests.get(
            OPEN_METEO_URL,
            params=params,
            timeout=20,
        )

        response.raise_for_status()

        result = response.json()
        current = result.get("current", {})

        if not current:
            raise ValueError("No current weather data returned.")

        record = {
            "retrieved_at": get_timestamp(),
            "location": "Delhi",
            "latitude": DELHI_LAT,
            "longitude": DELHI_LON,
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "rainfall": current.get("precipitation"),
            "visibility": current.get("visibility"),
            "wind_speed": current.get("wind_speed_10m"),
            "observation_time": current.get("time"),
        }

        df = pd.DataFrame([record])

        output = save_csv(df, "live_weather.csv")

        print("[OPEN-METEO] Weather received successfully.")
        print(df.to_string(index=False))
        print("Saved to:", output)

        return True

    except (
        requests.RequestException,
        ValueError,
    ) as error:

        print("[OPEN-METEO] Error:", error)
        return False


# ---------------------------------------------------------
# TOMTOM TRAFFIC CONNECTOR
# ---------------------------------------------------------

def fetch_tomtom_traffic():
    """Fetch live traffic flow for seven representative blackspots."""

    print("\n[TOMTOM] Fetching live traffic...")

    if not TOMTOM_API_KEY:
        print(
            "[TOMTOM] Skipped: TOMTOM_API_KEY is missing "
            "from the project root .env file."
        )
        return False

    segments_path = RAW_DIR / "segments.csv"

    if not segments_path.exists():
        print("[TOMTOM] Segments dataset not found:", segments_path)
        return False

    try:
        segments = pd.read_csv(segments_path)

    except (OSError, pd.errors.ParserError) as error:
        print("[TOMTOM] Could not read segments:", error)
        return False

    required_columns = {
        "segment_id",
        "corridor_name",
        "nearest_blackspot",
        "lat",
        "lon",
    }

    missing_columns = required_columns - set(segments.columns)

    if missing_columns:
        print(
            "[TOMTOM] Missing columns in segments.csv:",
            sorted(missing_columns),
        )
        return False

    # Select one representative road segment per blackspot.
    representatives = (
        segments.dropna(
            subset=[
                "nearest_blackspot",
                "lat",
                "lon",
            ]
        )
        .sort_values("segment_id")
        .drop_duplicates(
            subset="nearest_blackspot",
            keep="first",
        )
    )

    print(
        f"[TOMTOM] Testing {len(representatives)} "
        "representative locations."
    )

    records = []

    for segment in representatives.itertuples(index=False):

        blackspot = segment.nearest_blackspot

        params = {
            "key": TOMTOM_API_KEY,
            "point": f"{segment.lat},{segment.lon}",
            "unit": "KMPH",
        }

        try:
            response = requests.get(
                TOMTOM_URL,
                params=params,
                timeout=20,
            )

            response.raise_for_status()

            result = response.json()

            flow = result.get("flowSegmentData")

            if not flow:
                print(
                    f"[TOMTOM] No flow data returned for {blackspot}."
                )
                continue

            # Extract current traffic measurements.
            record = {
                "retrieved_at": get_timestamp(),

                "segment_id": segment.segment_id,
                "corridor_name": segment.corridor_name,
                "blackspot": blackspot,

                "latitude": segment.lat,
                "longitude": segment.lon,

                "current_speed": flow.get("currentSpeed"),
                "free_flow_speed": flow.get("freeFlowSpeed"),

                "current_travel_time": flow.get(
                    "currentTravelTime"
                ),

                "free_flow_travel_time": flow.get(
                    "freeFlowTravelTime"
                ),

                "confidence": flow.get("confidence"),
                "road_closure": flow.get("roadClosure"),

                "frc": flow.get("frc"),
            }

            records.append(record)

            print(
                f"[TOMTOM] Received: {blackspot} | "
                f"Current speed: {record['current_speed']} km/h | "
                f"Free-flow speed: {record['free_flow_speed']} km/h"
            )

        except requests.RequestException as error:

            print(
                f"[TOMTOM] Request failed for {blackspot}: "
                f"{safe_error(error, TOMTOM_API_KEY)}"
            )

        except ValueError as error:

            print(
                f"[TOMTOM] Invalid JSON response for {blackspot}: "
                f"{error}"
            )

    # Save all successfully collected observations.
    if records:

        df = pd.DataFrame(records)

        output = save_csv(df, "live_traffic.csv")

        print("\n[TOMTOM] Traffic observations saved:", len(df))
        print("Saved to:", output)

        print("\nTraffic observations:")
        print(
            df[
                [
                    "blackspot",
                    "current_speed",
                    "free_flow_speed",
                    "confidence",
                    "road_closure",
                ]
            ].to_string(index=False)
        )

        return True

    print("[TOMTOM] No traffic observations collected.")
    return False


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    STREAM_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DELHI LIVE DATA CONNECTORS")
    print("=" * 60)

    weather_success = fetch_weather()

    traffic_success = fetch_tomtom_traffic()

    print("\nCONNECTOR SUMMARY")
    print("-" * 30)

    print(
        "Open-Meteo:",
        "Success" if weather_success else "Failed",
    )

    print(
        "TomTom:",
        "Success" if traffic_success else "Failed/Skipped",
    )

    print("\nLive data directory:", STREAM_DIR)


if __name__ == "__main__":
    main()
