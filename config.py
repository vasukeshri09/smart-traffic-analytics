
"""Central configuration for Smart Traffic Analytics."""

import os
from pathlib import Path

# Reproducibility
SEED = 42

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_BRONZE = DATA_DIR / "bronze"
DATA_SILVER = DATA_DIR / "silver"
DATA_GOLD = DATA_DIR / "gold"
DATA_STREAM_IN = DATA_DIR / "stream_in"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

METRICS_PATH = REPORTS_DIR / "metrics.json"
HOTSPOTS_MAP_PATH = REPORTS_DIR / "hotspots.html"

# Delhi geographical reference
GEO_CENTER_LAT = 28.6139
GEO_CENTER_LON = 77.2090

# Dataset time range
START_DATE = "2024-01-01 00:00:00"
END_DATE = "2024-03-31 23:00:00"

# Training: January and February
# Testing: March
TRAIN_END = "2024-03-01 00:00:00"

# Network configuration
NUM_SEGMENTS = 60

ROAD_TYPE_COUNTS = {
    "highway": 10,
    "arterial": 20,
    "junction": 15,
    "local": 15,
}

FREE_FLOW_SPEEDS = {
    "highway": 80.0,
    "arterial": 50.0,
    "junction": 35.0,
    "local": 30.0,
}

DEFAULT_LANES = {
    "highway": 4,
    "arterial": 3,
    "junction": 2,
    "local": 1,
}

# Physical constraints
MIN_SPEED = 4.0

DENSITY_MIN = 0.0
DENSITY_MAX = 120.0

VISIBILITY_MIN = 0.3
VISIBILITY_MAX = 10.0

# Congestion thresholds
THRESHOLD_HIGH = 0.45
THRESHOLD_MEDIUM = 0.72

# Accident generation
ACCIDENT_RATE_TARGET = 0.07

# Spatial clustering
DBSCAN_EPS_METERS = 350.0
EARTH_RADIUS_METERS = 6371000.0

DBSCAN_EPS_RADIANS = (
    DBSCAN_EPS_METERS / EARTH_RADIUS_METERS
)

DBSCAN_MIN_SAMPLES = 60

KMEANS_K_MIN = 3
KMEANS_K_MAX = 12

# Kafka configuration
KAFKA_BOOTSTRAP = os.getenv(
    "KAFKA_BOOTSTRAP",
    "localhost:9092"
)

TOPICS = [
    "traffic.raw",
    "weather.raw",
    "accidents.raw",
]


def ensure_dirs():
    """Create required project directories."""

    directories = [
        DATA_RAW,
        DATA_BRONZE,
        DATA_SILVER,
        DATA_GOLD,
        DATA_STREAM_IN,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        DATA_GOLD / "predictions",
        DATA_GOLD / "accident_feed",
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )


if __name__ == "__main__":
    ensure_dirs()
    print("Project directories initialized.")
