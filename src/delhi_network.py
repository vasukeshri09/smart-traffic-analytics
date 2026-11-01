
"""Delhi NCR road network and accident blackspot reference data."""

import pandas as pd


CORRIDORS = [
    ("NH-48", "highway", 28.5560, 77.0950),
    ("DND Flyway", "highway", 28.5530, 77.3000),
    ("Inner Ring Road", "arterial", 28.5900, 77.2050),
    ("Outer Ring Road", "arterial", 28.6100, 77.1450),
    ("Barapullah Elevated Road", "arterial", 28.5800, 77.2450),
    ("Vikas Marg", "arterial", 28.6380, 77.2780),
    ("Mathura Road", "arterial", 28.5750, 77.2550),
    ("GT Karnal Road", "highway", 28.7350, 77.1650),
    ("Rohtak Road", "arterial", 28.6800, 77.0950),
    ("NH-9", "highway", 28.6500, 77.3150),
    ("Aurobindo Marg", "arterial", 28.5350, 77.2050),
    ("Mehrauli-Gurgaon Road", "arterial", 28.5050, 77.1750),
    ("Najafgarh Road", "arterial", 28.6200, 77.0950),
    ("Pankha Road", "arterial", 28.6150, 77.0850),
    ("Wazirabad Road", "arterial", 28.7050, 77.2300),
    ("Shahdara Road", "arterial", 28.6750, 77.2850),
    ("Ridge Road", "local", 28.6250, 77.1950),
    ("Africa Avenue", "local", 28.5650, 77.1850),
    ("Lodhi Road", "arterial", 28.5900, 77.2250),
    ("Ring Road - Dhaula Kuan stretch", "arterial", 28.5920, 77.1600),
]


BLACKSPOTS = {
    "Mukarba Chowk": (28.7355, 77.1575),
    "Kashmere Gate": (28.6675, 77.2280),
    "Dhaula Kuan": (28.5920, 77.1610),
    "Anand Vihar": (28.6465, 77.3160),
    "Ashram Chowk": (28.5715, 77.2590),
    "Peera Garhi": (28.6795, 77.0785),
    "Punjabi Bagh": (28.6690, 77.1320),
}


LANES = {
    "highway": 4,
    "arterial": 3,
    "junction": 2,
    "local": 1,
}


FREE_FLOW_SPEEDS = {
    "highway": 80.0,
    "arterial": 50.0,
    "junction": 35.0,
    "local": 30.0,
}


# Segment-level distribution:
# 10 highway, 20 arterial, 15 junction, 15 local.
# This 12-segment pattern is repeated five times.

ROAD_TYPE_PATTERN = (
    ["highway"] * 2
    + ["arterial"] * 4
    + ["junction"] * 3
    + ["local"] * 3
) * 5


def get_delhi_road_segments_df():
    """Generate 60 representative Delhi NCR road segments."""

    rows = []
    segment_number = 1

    offsets = [
        (-0.004, -0.005),
        (0.000, 0.000),
        (0.004, 0.005),
    ]

    for corridor, corridor_type, lat, lon in CORRIDORS:

        for section, (dlat, dlon) in enumerate(offsets, start=1):

            seg_lat = lat + dlat
            seg_lon = lon + dlon

            # Assign the segment-level road type.
            road_type = ROAD_TYPE_PATTERN[segment_number - 1]

            # Find the nearest accident blackspot.
            nearest_blackspot = min(
                BLACKSPOTS,
                key=lambda name: (
                    (BLACKSPOTS[name][0] - seg_lat) ** 2
                    + (BLACKSPOTS[name][1] - seg_lon) ** 2
                ),
            )

            distance_sq = (
                (BLACKSPOTS[nearest_blackspot][0] - seg_lat) ** 2
                + (BLACKSPOTS[nearest_blackspot][1] - seg_lon) ** 2
            )

            rows.append({
                "segment_id": f"DEL{segment_number:03d}",
                "corridor_name": f"{corridor} - Section {section}",
                "corridor_reference_type": corridor_type,
                "road_type": road_type,
                "lat": round(seg_lat, 6),
                "lon": round(seg_lon, 6),
                "lanes": LANES[road_type],
                "free_flow_speed": FREE_FLOW_SPEEDS[road_type],
                "length_km": round(0.8 + 0.2 * section, 2),
                "nearest_blackspot": nearest_blackspot,
                "is_blackspot_area": int(distance_sq <= 0.0004),
                "coordinate_quality": "approximate_simulation_anchor",
            })

            segment_number += 1

    return pd.DataFrame(rows)


if __name__ == "__main__":

    segments = get_delhi_road_segments_df()

    print("\nROAD NETWORK SUMMARY")
    print("--------------------")

    print("Total segments:", len(segments))
    print("Unique corridors:", len(CORRIDORS))
    print("Blackspots:", len(BLACKSPOTS))

    print("\nRoad type distribution:")
    print(segments["road_type"].value_counts().reindex(
        ["highway", "arterial", "junction", "local"],
        fill_value=0
    ))

    print("\nFirst 10 segments:")
    print(segments.head(10).to_string(index=False))
