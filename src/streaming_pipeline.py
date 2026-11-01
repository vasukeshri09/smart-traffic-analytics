
"""Stateful traffic stream processing with 72-hour replay and optional Kafka."""

import os
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

GOLD_FILE = ROOT / "data" / "gold" / "model_dataset.parquet"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
STREAM_DIR = ROOT / "data" / "stream_in"

REPLAY_HOURS = 72
KAFKA_TOPIC = "traffic.scored"


def choose_model(task):
    """Choose the model with the highest weighted F1 score."""

    comparison_file = REPORT_DIR / "model_comparison.csv"

    if comparison_file.exists():
        results = pd.read_csv(comparison_file)
        task_results = results[results["task"] == task]

        if not task_results.empty:
            best = task_results.sort_values(
                "f1_weighted", ascending=False
            ).iloc[0]

            return str(best["model"])

    return "random_forest"


def load_models():
    """Load the best saved congestion and accident models."""

    models = {}

    for task in ["congestion", "accident"]:

        model_name = choose_model(task)
        model_path = MODEL_DIR / f"{task}_{model_name}.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. "
                "Run model training first."
            )

        artifact = joblib.load(model_path)

        models[task] = {
            "name": model_name,
            "pipeline": artifact["pipeline"],
            "features": artifact["features"],
            "label_encoder": artifact.get("label_encoder"),
        }

        print(f"Loaded {task} model: {model_name}")

    return models


def connect_kafka():
    """Connect to Kafka if configured and available."""

    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS")

    if not bootstrap:
        print("\nKafka not configured. Using local-file mode.")
        return None

    try:
        from kafka import KafkaProducer

        producer = KafkaProducer(
            bootstrap_servers=bootstrap.split(","),
            value_serializer=lambda value: json.dumps(
                value, default=str
            ).encode("utf-8"),
            request_timeout_ms=5000,
            max_block_ms=5000,
        )

        print("Connected to Kafka:", bootstrap)
        return producer

    except Exception as error:
        print("Kafka unavailable:", error)
        print("Continuing in local-file mode.")
        return None


def load_replay_data():
    """Load the most recent 72 hours of Gold observations."""

    if not GOLD_FILE.exists():
        raise FileNotFoundError(
            "Gold dataset not found. Run data engineering first."
        )

    data = pd.read_parquet(GOLD_FILE)
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    latest = data["timestamp"].max()
    earliest = latest - pd.Timedelta(hours=REPLAY_HOURS - 1)

    replay = data[
        (data["timestamp"] >= earliest)
        & (data["timestamp"] <= latest)
    ].copy()

    replay = replay.sort_values(
        ["timestamp", "segment_id"]
    ).reset_index(drop=True)

    if replay.empty:
        raise ValueError("No records found for replay.")

    print("\nReplay period:")
    print("From:", earliest)
    print("To:  ", latest)
    print("Records:", len(replay))
    print("Segments:", replay["segment_id"].nunique())

    return replay


def score_event(row, models, state):
    """Score one event and update per-segment processing state."""

    segment_id = str(row["segment_id"])
    timestamp = row["timestamp"]

    current_speed = float(row["speed"])

    previous_state = state.get(segment_id)

    if previous_state is None:
        previous_speed = None
        speed_change = None
        observation_count = 1
    else:
        previous_speed = previous_state["speed"]
        speed_change = current_speed - previous_speed
        observation_count = previous_state["count"] + 1

    # Update state for this road segment.
    state[segment_id] = {
        "speed": current_speed,
        "timestamp": timestamp,
        "count": observation_count,
    }

    output = {
        "timestamp": timestamp,
        "segment_id": segment_id,
        "speed": current_speed,
        "density": float(row["density"]),
        "state_previous_speed": previous_speed,
        "state_speed_change": speed_change,
        "state_observation_count": observation_count,
    }

    # Congestion prediction.
    congestion_model = models["congestion"]
    congestion_features = pd.DataFrame(
        [[row[column] for column in congestion_model["features"]]],
        columns=congestion_model["features"],
    )

    congestion_pipeline = congestion_model["pipeline"]

    congestion_prediction = congestion_pipeline.predict(
        congestion_features
    )[0]

    congestion_probabilities = congestion_pipeline.predict_proba(
        congestion_features
    )[0]

    congestion_classes = (
        congestion_pipeline.named_steps["classifier"].classes_
    )

    encoder = congestion_model["label_encoder"]

    if encoder is not None:
        congestion_prediction = encoder.inverse_transform(
            [int(congestion_prediction)]
        )[0]

        congestion_classes = encoder.inverse_transform(
            np.asarray(congestion_classes, dtype=int)
        )

    output["predicted_congestion"] = str(congestion_prediction)

    output["congestion_probabilities"] = {
        str(label): float(probability)
        for label, probability in zip(
            congestion_classes,
            congestion_probabilities,
        )
    }

    # Accident risk prediction.
    accident_model = models["accident"]
    accident_features = pd.DataFrame(
        [[row[column] for column in accident_model["features"]]],
        columns=accident_model["features"],
    )

    accident_pipeline = accident_model["pipeline"]

    accident_probabilities = accident_pipeline.predict_proba(
        accident_features
    )[0]

    accident_classes = (
        accident_pipeline.named_steps["classifier"].classes_
    )

    positive_positions = np.where(
        np.asarray(accident_classes) == 1
    )[0]

    if len(positive_positions):
        accident_probability = float(
            accident_probabilities[positive_positions[0]]
        )
    else:
        accident_probability = 0.0

    output["accident_probability"] = accident_probability
    output["predicted_accident_risk"] = int(
        accident_probability >= 0.5
    )

    return output


def main():

    STREAM_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SMART TRAFFIC STREAMING PIPELINE")
    print("=" * 60)

    models = load_models()
    replay = load_replay_data()

    producer = connect_kafka()

    state = {}
    results = []

    output_csv = STREAM_DIR / "scored_stream.csv"
    output_jsonl = STREAM_DIR / "scored_stream.jsonl"

    # Start fresh for this replay run.
    if output_csv.exists():
        output_csv.unlink()

    if output_jsonl.exists():
        output_jsonl.unlink()

    print("\nStarting 72-hour replay...")

    start_time = time.time()

    for index, row in replay.iterrows():

        result = score_event(row, models, state)
        results.append(result)

        # Publish to Kafka when available.
        if producer is not None:
            try:
                producer.send(KAFKA_TOPIC, result)
            except Exception as error:
                print("Kafka publishing failed:", error)
                producer.close()
                producer = None

        # Save JSONL event.
        with open(
            output_jsonl, "a", encoding="utf-8"
        ) as file:
            file.write(json.dumps(result, default=str) + "\n")

        # Progress display.
        if (index + 1) % 500 == 0:
            print(f"Processed {index + 1}/{len(replay)} events")

    # Flush Kafka messages.
    if producer is not None:
        producer.flush()
        producer.close()

    # Save scored events as CSV.
    results_df = pd.DataFrame(results)

    results_df.drop(
        columns=["congestion_probabilities"]
    ).to_csv(output_csv, index=False)

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("STREAM PROCESSING COMPLETED")
    print("=" * 60)

    print("Processed events:", len(results_df))
    print("Segments tracked:", len(state))
    print("Processing time:", round(elapsed, 2), "seconds")

    print("\nCongestion predictions:")
    print(results_df["predicted_congestion"].value_counts())

    print("\nPredicted accident risks:")
    print(results_df["predicted_accident_risk"].value_counts())

    print("\nAverage accident probability:",
          round(results_df["accident_probability"].mean(), 4))

    print("\nOutput files:")
    print(output_csv)
    print(output_jsonl)


if __name__ == "__main__":
    main()
