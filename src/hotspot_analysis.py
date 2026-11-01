
"""Detect Delhi accident hotspots using DBSCAN and K-Means."""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
SILVER = ROOT / "data" / "silver"
REPORTS = ROOT / "reports"

EARTH_RADIUS_METRES = 6_371_000
DBSCAN_EPS_METRES = 350
DBSCAN_MIN_SAMPLES = 60


def load_data():
    """Load accident and road-segment datasets."""

    accidents = pd.read_csv(RAW / "accidents.csv")
    segments = pd.read_parquet(SILVER / "segments.parquet")

    return accidents, segments


def run_dbscan(accidents):
    """Detect geographic accident clusters using Haversine distance."""

    print("\n[DBSCAN] Detecting accident hotspots...")

    coordinates = accidents[["lat", "lon"]].dropna().to_numpy()

    # Haversine expects coordinates in radians.
    coordinates_rad = np.radians(coordinates)

    model = DBSCAN(
        eps=DBSCAN_EPS_METRES / EARTH_RADIUS_METRES,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric="haversine",
        algorithm="ball_tree",
        n_jobs=-1,
    )

    labels = model.fit_predict(coordinates_rad)

    results = pd.DataFrame({
        "lat": coordinates[:, 0],
        "lon": coordinates[:, 1],
        "cluster": labels,
    })

    # -1 represents noise, not a hotspot.
    hotspots = results[results["cluster"] != -1].copy()

    if not hotspots.empty:
        summary = (
            hotspots.groupby("cluster")
            .agg(
                accident_count=("cluster", "size"),
                latitude=("lat", "mean"),
                longitude=("lon", "mean"),
            )
            .reset_index()
            .sort_values("accident_count", ascending=False)
        )
    else:
        summary = pd.DataFrame(
            columns=[
                "cluster",
                "accident_count",
                "latitude",
                "longitude",
            ]
        )

    results.to_csv(
        REPORTS / "dbscan_accident_labels.csv",
        index=False,
    )

    summary.to_csv(
        REPORTS / "dbscan_hotspots.csv",
        index=False,
    )

    print("Accident observations:", len(results))
    print("Hotspot clusters:", len(summary))
    print("Noise observations:", (labels == -1).sum())

    return summary


def run_kmeans(accidents, segments):
    """Group road segments and select k using silhouette score."""

    print("\n[K-MEANS] Grouping accident-prone road segments...")

    # Count accidents for each road segment.
    accident_counts = (
        accidents.groupby("segment_id")
        .size()
        .rename("accident_count")
    )

    data = segments.merge(
        accident_counts,
        on="segment_id",
        how="left",
    )

    data["accident_count"] = (
        data["accident_count"].fillna(0)
    )

    # Geographic coordinates are converted to approximate
    # kilometres relative to the mean latitude.
    mean_lat = data["lat"].mean()

    data["x_km"] = (
        data["lon"] - data["lon"].mean()
    ) * 111.32 * np.cos(np.radians(mean_lat))

    data["y_km"] = (
        data["lat"] - data["lat"].mean()
    ) * 111.32

    features = data[
        ["x_km", "y_km", "accident_count"]
    ].to_numpy()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    scores = []

    # Evaluate k from 3 to 12.
    for k in range(3, 13):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
            algorithm="lloyd",
        )

        labels = model.fit_predict(scaled_features)

        score = silhouette_score(
            scaled_features,
            labels,
        )

        scores.append({
            "k": k,
            "silhouette_score": score,
        })

        print(f"k={k}: silhouette score={score:.4f}")

    scores_df = pd.DataFrame(scores)

    best_k = int(
        scores_df.loc[
            scores_df["silhouette_score"].idxmax(),
            "k",
        ]
    )

    print("\nBest number of clusters:", best_k)

    # Fit final K-Means model.
    final_model = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=10,
    )

    data["kmeans_cluster"] = final_model.fit_predict(
        scaled_features
    )

    cluster_summary = (
        data.groupby("kmeans_cluster")
        .agg(
            segment_count=("segment_id", "count"),
            accident_count=("accident_count", "sum"),
            latitude=("lat", "mean"),
            longitude=("lon", "mean"),
        )
        .reset_index()
        .sort_values("accident_count", ascending=False)
    )

    data.to_csv(
        REPORTS / "kmeans_segment_clusters.csv",
        index=False,
    )

    cluster_summary.to_csv(
        REPORTS / "kmeans_cluster_summary.csv",
        index=False,
    )

    scores_df.to_csv(
        REPORTS / "kmeans_silhouette_scores.csv",
        index=False,
    )

    return data, cluster_summary, scores_df


def blackspot_analysis(accidents, segments):
    """Rank the predefined accident blackspot areas."""

    print("\n[BLACKSPOT] Ranking predefined blackspots...")

    counts = (
        accidents.groupby("segment_id")
        .size()
        .rename("accident_count")
    )

    data = segments.merge(
        counts,
        on="segment_id",
        how="left",
    )

    data["accident_count"] = (
        data["accident_count"].fillna(0).astype(int)
    )

    summary = (
        data.groupby("nearest_blackspot")
        .agg(
            segment_count=("segment_id", "count"),
            accident_count=("accident_count", "sum"),
        )
        .reset_index()
        .sort_values("accident_count", ascending=False)
    )

    summary.to_csv(
        REPORTS / "blackspot_ranking.csv",
        index=False,
    )

    print(summary.to_string(index=False))

    return summary


def main():

    REPORTS.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("ACCIDENT HOTSPOT ANALYSIS")
    print("=" * 55)

    accidents, segments = load_data()

    print("Accident records:", len(accidents))
    print("Road segments:", len(segments))

    dbscan_summary = run_dbscan(accidents)

    _, kmeans_summary, scores = run_kmeans(
        accidents,
        segments,
    )

    blackspots = blackspot_analysis(
        accidents,
        segments,
    )

    print("\n" + "=" * 55)
    print("HOTSPOT ANALYSIS COMPLETED")
    print("=" * 55)

    print("\nDBSCAN hotspots:", len(dbscan_summary))
    print("K-Means clusters:", len(kmeans_summary))
    print("Best silhouette score:",
          round(scores["silhouette_score"].max(), 4))

    print("\nReports saved in:", REPORTS)


if __name__ == "__main__":
    main()
