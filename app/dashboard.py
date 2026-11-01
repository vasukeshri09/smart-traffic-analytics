
"""Smart Traffic Congestion and Accident Risk Analytics Dashboard."""

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium

from streamlit_folium import st_folium


# =========================================================
# 1. CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold"
STREAM = ROOT / "data" / "stream_in"
REPORTS = ROOT / "reports"

st.set_page_config(
    page_title="Smart Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# 2. VISUAL THEME
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --bg: #080d18;
    --panel: #111a2b;
    --panel2: #17243a;
    --border: rgba(148,163,184,0.16);
    --muted: #94a3b8;
    --text: #e5edf8;
    --cyan: #38bdf8;
    --green: #34d399;
    --red: #fb7185;
    --amber: #fbbf24;
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(14,165,233,.12), transparent 35%),
        radial-gradient(ellipse at 90% 10%, rgba(99,102,241,.10), transparent 30%),
        var(--bg);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(8,13,24,.85);
}

[data-testid="stSidebar"] {
    display: none;
}

.block-container {
    max-width: 1600px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #f1f5f9 !important;
    letter-spacing: -0.7px;
    font-weight: 800 !important;
}

p, label, li {
    color: #cbd5e1;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #17243a, #101827);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 19px 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,.18);
    transition: transform .25s ease, border-color .25s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    border-color: rgba(56,189,248,.55);
}

[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: .86rem !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-size: 1.75rem !important;
    font-weight: 800 !important;
}

[data-testid="stMetricDelta"] {
    font-size: .8rem !important;
}

.stButton > button {
    border-radius: 10px;
    border: 1px solid rgba(56,189,248,.35);
    background: linear-gradient(110deg, #0e7490, #2563eb);
    color: white;
    font-weight: 700;
    transition: all .2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 22px rgba(56,189,248,.22);
    border-color: #38bdf8;
}

div[role="radiogroup"] {
    gap: 8px;
    flex-wrap: wrap;
}

div[role="radiogroup"] label {
    background: #17243a;
    border: 1px solid var(--border);
    padding: 9px 15px;
    border-radius: 10px;
    font-weight: 600;
    transition: all .2s ease;
}

div[role="radiogroup"] label:hover {
    border-color: var(--cyan);
    transform: translateY(-2px);
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

.stSelectbox, .stDateInput, .stMultiSelect {
    border-radius: 10px;
}

hr {
    border-color: var(--border);
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 27px 30px;
    border-radius: 20px;
    border: 1px solid rgba(56,189,248,.25);
    background: linear-gradient(115deg,
        rgba(14,116,144,.24),
        rgba(30,41,59,.88) 48%,
        rgba(49,46,129,.30));
    margin-bottom: 20px;
    box-shadow: 0 12px 40px rgba(0,0,0,.22);
}

.hero:after {
    content: "";
    position: absolute;
    width: 300px;
    height: 300px;
    border-radius: 50%;
    right: -100px;
    top: -180px;
    background: rgba(56,189,248,.13);
    filter: blur(10px);
    animation: pulse 5s ease-in-out infinite alternate;
}

.hero-title {
    font-size: 2.25rem;
    font-weight: 900;
    color: #f8fafc;
    letter-spacing: -1.3px;
    margin: 0;
}

.hero-subtitle {
    color: #a5bfd4;
    font-size: .98rem;
    margin-top: 7px;
}

.eyebrow {
    color: #67e8f9;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: .72rem;
    font-weight: 800;
    margin-bottom: 8px;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 12px;
    background: rgba(16,185,129,.12);
    border: 1px solid rgba(52,211,153,.3);
    border-radius: 30px;
    color: #6ee7b7;
    font-size: .78rem;
    font-weight: 700;
}

.live-dot {
    width: 8px;
    height: 8px;
    background: #34d399;
    border-radius: 50%;
    box-shadow: 0 0 12px #34d399;
    animation: blink 1.5s infinite;
}

.section-label {
    color: #67e8f9;
    font-size: .75rem;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 5px;
}

.info-card {
    background: linear-gradient(145deg, #17243a, #101827);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 17px;
    height: 100%;
    transition: all .25s ease;
}

.info-card:hover {
    transform: translateY(-3px);
    border-color: rgba(56,189,248,.5);
}

.info-label {
    color: #94a3b8;
    font-size: .8rem;
    font-weight: 600;
}

.info-value {
    color: #f8fafc;
    font-size: 1.6rem;
    font-weight: 800;
    margin-top: 5px;
}

.info-detail {
    color: #94a3b8;
    font-size: .78rem;
    margin-top: 4px;
}

.good { color: #34d399; }
.warning { color: #fbbf24; }
.danger { color: #fb7185; }

[data-testid="stCaptionContainer"] {
    color: #94a3b8;
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: .35; }
}

@keyframes pulse {
    from { transform: scale(.85); opacity: .5; }
    to { transform: scale(1.2); opacity: 1; }
}

@media (max-width: 768px) {
    .hero-title { font-size: 1.6rem; }
    .block-container { padding: 1rem; }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. DATA LOADING
# =========================================================

@st.cache_data(ttl=30)
def load_csv(path):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=30)
def load_parquet(path):
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def parse_dates(df):
    """Convert available date columns safely."""

    if df.empty:
        return df

    df = df.copy()

    for column in ["timestamp", "retrieved_at", "observation_time"]:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
                utc=True,
            )

    return df


def numeric_column(df, column):
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def latest_record(df, time_column):
    if df.empty:
        return None

    if time_column in df.columns:
        return df.sort_values(time_column).iloc[-1]

    return df.iloc[-1]


# Historical data
segments = load_csv(RAW / "segments.csv")
traffic = load_parquet(SILVER / "traffic.parquet")
weather = load_parquet(SILVER / "weather.parquet")
accidents = load_csv(RAW / "accidents.csv")
gold = load_parquet(GOLD / "model_dataset.parquet")

model_results = load_csv(REPORTS / "model_comparison.csv")
dbscan = load_csv(REPORTS / "dbscan_hotspots.csv")
kmeans = load_csv(REPORTS / "kmeans_segment_clusters.csv")
blackspots = load_csv(REPORTS / "blackspot_ranking.csv")
stream = load_csv(STREAM / "scored_stream.csv")

# Live data
live_weather = load_csv(STREAM / "live_weather.csv")
live_traffic = load_csv(STREAM / "live_traffic.csv")

traffic = parse_dates(traffic)
weather = parse_dates(weather)
accidents = parse_dates(accidents)
gold = parse_dates(gold)
stream = parse_dates(stream)
live_weather = parse_dates(live_weather)
live_traffic = parse_dates(live_traffic)

# Silver traffic does not necessarily contain road_type.
# Add segment metadata for dashboard filtering.
if (
    not traffic.empty
    and not segments.empty
    and "segment_id" in traffic.columns
    and "segment_id" in segments.columns
):
    metadata = [
        c for c in [
            "segment_id",
            "road_type",
            "corridor_name",
            "nearest_blackspot",
            "free_flow_speed",
            "lat",
            "lon",
        ]
        if c in segments.columns
    ]

    if "road_type" not in traffic.columns:
        traffic = traffic.merge(
            segments[metadata].drop_duplicates("segment_id"),
            on="segment_id",
            how="left",
            suffixes=("", "_segment"),
        )


# =========================================================
# 4. LIVE TRAFFIC METRICS
# =========================================================

if not live_traffic.empty:

    live_traffic = live_traffic.copy()

    live_traffic["current_speed"] = numeric_column(
        live_traffic, "current_speed"
    )

    live_traffic["free_flow_speed"] = numeric_column(
        live_traffic, "free_flow_speed"
    )

    live_traffic["confidence"] = numeric_column(
        live_traffic, "confidence"
    )

    # Percentage reduction relative to free-flow speed.
    denominator = live_traffic["free_flow_speed"].replace(0, np.nan)

    live_traffic["speed_reduction_pct"] = (
        (
            denominator - live_traffic["current_speed"]
        ) / denominator * 100
    ).clip(lower=0, upper=100)

    live_traffic["speed_reduction_pct"] = (
        live_traffic["speed_reduction_pct"].fillna(0)
    )

    live_traffic["congestion_status"] = np.select(
        [
            live_traffic["speed_reduction_pct"] >= 50,
            live_traffic["speed_reduction_pct"] >= 25,
        ],
        [
            "Heavy",
            "Moderate",
        ],
        default="Light",
    )

    if "road_closure" in live_traffic.columns:
        live_traffic["road_closure"] = (
            live_traffic["road_closure"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )
    else:
        live_traffic["road_closure"] = False


# =========================================================
# 5. PLOTLY THEME
# =========================================================

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(
        family="Inter, sans-serif",
        color="#cbd5e1",
    ),
    margin=dict(l=15, r=15, t=45, b=15),
    hoverlabel=dict(
        bgcolor="#17243a",
        font_size=12,
    ),
)

def style_figure(fig, height=380):
    fig.update_layout(**PLOT_LAYOUT, height=height)
    return fig


# =========================================================
# 6. SHARED UI HELPERS
# =========================================================

def section_heading(title, subtitle=None):
    st.markdown(
        f'<div class="section-label">{title}</div>',
        unsafe_allow_html=True,
    )

    if subtitle:
        st.caption(subtitle)


def show_empty(message):
    st.info(message)


def display_hero():
    st.markdown("""
    <div class="hero">
        <div class="eyebrow">Delhi NCR · Big Data Analytics</div>
        <div class="hero-title">Traffic Intelligence Center</div>
        <div class="hero-subtitle">
            Congestion analytics · Accident hotspots ·
            Machine learning · Live traffic monitoring
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_live_status():
    if not live_traffic.empty or not live_weather.empty:
        st.markdown(
            '<span class="status-pill">'
            '<span class="live-dot"></span>'
            'LIVE DATA CONNECTED'
            '</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-pill" style="color:#fbbf24;">'
            'NO LIVE DATA AVAILABLE'
            '</span>',
            unsafe_allow_html=True,
        )


def metric_row():
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Road Segments", f"{len(segments):,}")
    c2.metric("Historical Traffic Records", f"{len(traffic):,}")
    c3.metric("Accident Events", f"{len(accidents):,}")
    c4.metric("DBSCAN Hotspots", f"{len(dbscan):,}")


def traffic_status_color(status):
    return {
        "Heavy": "red",
        "Moderate": "orange",
        "Light": "green",
    }.get(status, "blue")


# =========================================================
# 7. MAP GENERATION
# =========================================================

def make_traffic_map(show_live=False, show_hotspots=True):
    """Display approximate segment anchors and optional live traffic."""

    if segments.empty:
        return None

    m = folium.Map(
        location=[28.6139, 77.2090],
        zoom_start=11,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    colors = {
        "highway": "#ef4444",
        "arterial": "#3b82f6",
        "junction": "#f59e0b",
        "local": "#22c55e",
    }

    # Historical segment anchors.
    for row in segments.itertuples(index=False):

        popup = (
            f"<b>{row.corridor_name}</b><br>"
            f"Road type: {row.road_type}<br>"
            f"Free-flow speed: {row.free_flow_speed} km/h<br>"
            f"Nearest blackspot: {row.nearest_blackspot}"
        )

        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=4,
            color=colors.get(row.road_type, "#94a3b8"),
            fill=True,
            fill_color=colors.get(row.road_type, "#94a3b8"),
            fill_opacity=.65,
            weight=1,
            popup=folium.Popup(popup, max_width=300),
            tooltip=row.corridor_name,
        ).add_to(m)

    # Live TomTom measurements.
    if show_live and not live_traffic.empty:

        for row in live_traffic.itertuples(index=False):

            color = traffic_status_color(row.congestion_status)

            popup = (
                f"<b>{row.blackspot}</b><br>"
                f"Current speed: {row.current_speed} km/h<br>"
                f"Free-flow speed: {row.free_flow_speed} km/h<br>"
                f"Speed reduction: {row.speed_reduction_pct:.1f}%<br>"
                f"Confidence: {row.confidence:.2f}<br>"
                f"Road closure: {row.road_closure}"
            )

            folium.CircleMarker(
                location=[row.latitude, row.longitude],
                radius=11,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=.9,
                weight=3,
                popup=folium.Popup(popup, max_width=300),
                tooltip=f"{row.blackspot} | {row.congestion_status}",
            ).add_to(m)

    # DBSCAN hotspot areas.
    if show_hotspots and not dbscan.empty:

        for row in dbscan.itertuples(index=False):

            if pd.isna(row.latitude) or pd.isna(row.longitude):
                continue

            folium.Circle(
                location=[row.latitude, row.longitude],
                radius=350,
                color="#fb7185",
                fill=True,
                fill_color="#fb7185",
                fill_opacity=.12,
                tooltip=f"DBSCAN cluster: {row.cluster}",
            ).add_to(m)

    folium.LayerControl().add_to(m)

    return m


# =========================================================
# 8. NAVIGATION
# =========================================================

display_hero()

pages = [
    "Overview",
    "Traffic Analytics",
    "Accident Hotspots",
    "Prediction Centre",
    "Live Monitoring",
    "Model Evaluation",
]

page = st.radio(
    "Dashboard Navigation",
    pages,
    horizontal=True,
    label_visibility="collapsed",
    key="main_navigation",
)

st.markdown("---")


# =========================================================
# 9. OVERVIEW
# =========================================================

if page == "Overview":

    section_heading(
        "SYSTEM OVERVIEW",
        "A unified view of historical analytics and current observations.",
    )

    metric_row()

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Road Network Composition")

        if not segments.empty:

            counts = (
                segments["road_type"]
                .value_counts()
                .rename_axis("Road Type")
                .reset_index(name="Segments")
            )

            fig = px.pie(
                counts,
                names="Road Type",
                values="Segments",
                hole=.62,
                color_discrete_sequence=[
                    "#38bdf8",
                    "#818cf8",
                    "#fbbf24",
                    "#34d399",
                ],
            )

            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
                marker=dict(line=dict(color="#101827", width=3)),
            )

            st.plotly_chart(
                style_figure(fig, 350),
                use_container_width=True,
            )

    with right:

        st.subheader("Historical Congestion Distribution")

        if not gold.empty and "congestion_target" in gold.columns:

            counts = (
                gold["congestion_target"]
                .value_counts()
                .rename_axis("Congestion")
                .reset_index(name="Records")
            )

            fig = px.bar(
                counts,
                x="Congestion",
                y="Records",
                color="Congestion",
                text="Records",
                color_discrete_sequence=[
                    "#34d399",
                    "#fbbf24",
                    "#fb7185",
                ],
            )

            fig.update_layout(
                xaxis_title="Congestion Class",
                yaxis_title="Number of Records",
                showlegend=False,
            )

            st.plotly_chart(
                style_figure(fig, 350),
                use_container_width=True,
            )

        else:
            show_empty("Historical congestion dataset unavailable.")

    st.divider()

    section_heading("DELHI ROAD NETWORK", "Approximate road-segment locations.")

    road_map = make_traffic_map(show_live=True)

    if road_map is not None:
        st_folium(
            road_map,
            width=None,
            height=550,
            key="overview_map",
        )

    st.caption(
        "Historical traffic and accident records are synthetic simulation "
        "data. Coordinates are approximate anchors, not surveyed road geometry. "
        "Live markers represent individual TomTom query locations."
    )


# =========================================================
# 10. TRAFFIC ANALYTICS
# =========================================================

elif page == "Traffic Analytics":

    section_heading(
        "HISTORICAL TRAFFIC ANALYTICS",
        "Explore traffic patterns across the January–March 2024 simulation.",
    )

    if traffic.empty:
        show_empty("Silver traffic dataset is unavailable.")

    else:

        start_date = traffic["timestamp"].min().date()
        end_date = traffic["timestamp"].max().date()

        selected_range = st.date_input(
            "Select date range",
            value=(start_date, end_date),
            min_value=start_date,
            max_value=end_date,
        )

        filtered = traffic.copy()

        if isinstance(selected_range, tuple) and len(selected_range) == 2:

            filtered = filtered[
                filtered["timestamp"].dt.date.between(
                    selected_range[0],
                    selected_range[1],
                )
            ]

        if "road_type" in filtered.columns:

            road_types = sorted(
                filtered["road_type"].dropna().unique()
            )

            selected_type = st.selectbox(
                "Filter by road type",
                ["All"] + road_types,
            )

            if selected_type != "All":
                filtered = filtered[
                    filtered["road_type"] == selected_type
                ]

        if filtered.empty:
            show_empty("No records match the selected filters.")

        else:

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Average Speed",
                f"{filtered['speed'].mean():.1f} km/h",
            )

            c2.metric(
                "Average Density",
                f"{filtered['density'].mean():.2f}",
            )

            c3.metric(
                "Average Volume",
                f"{filtered['volume'].mean():.1f}",
            )

            hourly = (
                filtered.assign(
                    hour=filtered["timestamp"].dt.hour
                )
                .groupby("hour")
                .agg(
                    average_speed=("speed", "mean"),
                    average_density=("density", "mean"),
                    average_volume=("volume", "mean"),
                )
                .reset_index()
            )

            st.subheader("Hourly Traffic Patterns")

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=hourly["hour"],
                    y=hourly["average_speed"],
                    name="Average speed",
                    mode="lines+markers",
                    line=dict(width=3, color="#38bdf8"),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=hourly["hour"],
                    y=hourly["average_density"],
                    name="Average density",
                    mode="lines+markers",
                    yaxis="y2",
                    line=dict(width=3, color="#fbbf24"),
                )
            )

            fig.update_layout(
                title="Speed and Density by Hour",
                xaxis_title="Hour of day",
                yaxis=dict(title="Speed (km/h)"),
                yaxis2=dict(
                    title="Density",
                    overlaying="y",
                    side="right",
                ),
                hovermode="x unified",
            )

            st.plotly_chart(
                style_figure(fig, 420),
                use_container_width=True,
            )

            fig = px.bar(
                hourly,
                x="hour",
                y="average_volume",
                title="Average Traffic Volume by Hour",
                color="average_volume",
                color_continuous_scale="Teal",
            )

            st.plotly_chart(
                style_figure(fig, 350),
                use_container_width=True,
            )

            st.subheader("Traffic Records")

            st.dataframe(
                filtered.head(200),
                use_container_width=True,
                hide_index=True,
            )


# =========================================================
# 11. ACCIDENT HOTSPOTS
# =========================================================

elif page == "Accident Hotspots":

    section_heading(
        "ACCIDENT HOTSPOT INTELLIGENCE",
        "Explore simulated accident patterns and clustering results.",
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Accident Events", f"{len(accidents):,}")
    c2.metric("DBSCAN Clusters", f"{len(dbscan):,}")
    c3.metric("Predefined Blackspots", f"{len(blackspots):,}")

    st.divider()

    st.subheader("Accident and Road Network Map")

    road_map = make_traffic_map(show_live=False, show_hotspots=True)

    if road_map is not None:
        st_folium(
            road_map,
            width=None,
            height=540,
            key="hotspot_map",
        )

    left, right = st.columns(2)

    with left:

        st.subheader("DBSCAN Hotspots")

        if not dbscan.empty:

            if "accident_count" in dbscan.columns:
                display_dbscan = dbscan.sort_values(
                    "accident_count",
                    ascending=False,
                )
            else:
                display_dbscan = dbscan

            st.dataframe(
                display_dbscan,
                use_container_width=True,
                hide_index=True,
            )

        else:
            show_empty("No DBSCAN hotspot clusters available.")

    with right:

        st.subheader("K-Means Cluster Summary")

        cluster_summary = load_csv(
            REPORTS / "kmeans_cluster_summary.csv"
        )

        if not cluster_summary.empty:
            st.dataframe(
                cluster_summary,
                use_container_width=True,
                hide_index=True,
            )
        else:
            show_empty("K-Means cluster summary unavailable.")

    st.divider()

    st.subheader("Predefined Blackspot Ranking")

    if not blackspots.empty:

        if "accident_count" in blackspots.columns:

            fig = px.bar(
                blackspots.sort_values(
                    "accident_count",
                    ascending=False,
                ),
                x="nearest_blackspot",
                y="accident_count",
                color="accident_count",
                color_continuous_scale="Reds",
                title="Accident Events by Blackspot",
                labels={
                    "nearest_blackspot": "Blackspot",
                    "accident_count": "Accident count",
                },
            )

            st.plotly_chart(
                style_figure(fig, 380),
                use_container_width=True,
            )

        st.dataframe(
            blackspots,
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Accident locations and clustering results are derived from "
        "the project's simulated accident dataset."
    )


# =========================================================
# 12. PREDICTION CENTRE
# =========================================================

elif page == "Prediction Centre":

    section_heading(
        "MACHINE LEARNING PREDICTIONS",
        "Historical 72-hour stream replay scored using trained models.",
    )

    if stream.empty:

        show_empty(
            "No scored streaming results found. "
            "Run python -m src.streaming_pipeline first."
        )

    else:

        latest = stream.copy()

        if "timestamp" in latest.columns:
            latest = latest.sort_values("timestamp")

        st.markdown(
            '<span class="status-pill">'
            'HISTORICAL MODEL REPLAY'
            '</span>',
            unsafe_allow_html=True,
        )

        if "timestamp" in latest.columns:
            st.caption(
                "Latest processed observation: "
                + str(latest["timestamp"].max())
            )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Processed Events",
            f"{len(latest):,}",
        )

        if "predicted_accident_risk" in latest.columns:

            risk_values = pd.to_numeric(
                latest["predicted_accident_risk"],
                errors="coerce",
            )

            c2.metric(
                "High Accident Risk Predictions",
                int(risk_values.fillna(0).sum()),
            )

        else:
            c2.metric("High Accident Risk Predictions", "N/A")

        if "accident_probability" in latest.columns:

            probabilities = pd.to_numeric(
                latest["accident_probability"],
                errors="coerce",
            )

            c3.metric(
                "Average Accident Probability",
                f"{probabilities.mean():.2%}",
            )

        else:
            c3.metric("Average Accident Probability", "N/A")

        st.divider()

        if "predicted_congestion" in latest.columns:

            st.subheader("Predicted Congestion Distribution")

            counts = (
                latest["predicted_congestion"]
                .value_counts()
                .rename_axis("Congestion")
                .reset_index(name="Count")
            )

            fig = px.pie(
                counts,
                names="Congestion",
                values="Count",
                hole=.58,
                color_discrete_sequence=[
                    "#34d399",
                    "#fbbf24",
                    "#fb7185",
                ],
            )

            st.plotly_chart(
                style_figure(fig, 360),
                use_container_width=True,
            )

        if "accident_probability" in latest.columns:

            st.subheader("Accident Risk Probability Distribution")

            fig = px.histogram(
                latest,
                x="accident_probability",
                nbins=30,
                color_discrete_sequence=["#38bdf8"],
            )

            fig.update_layout(
                xaxis_title="Predicted accident probability",
                yaxis_title="Number of observations",
            )

            st.plotly_chart(
                style_figure(fig, 350),
                use_container_width=True,
            )

        if "segment_id" in latest.columns:

            st.subheader("Segment-level Predictions")

            selected_segment = st.selectbox(
                "Select road segment",
                sorted(latest["segment_id"].dropna().unique()),
            )

            segment_results = latest[
                latest["segment_id"] == selected_segment
            ]

            if "timestamp" in segment_results.columns:
                segment_results = segment_results.sort_values(
                    "timestamp"
                )

            st.dataframe(
                segment_results.tail(50),
                use_container_width=True,
                hide_index=True,
            )

        st.warning(
            "These are predictions from the historical replay dataset, "
            "not predictions generated from the current TomTom readings."
        )


# =========================================================
# 13. LIVE MONITORING
# =========================================================

def render_live_monitoring():

    section_heading(
        "LIVE OPERATIONS CENTER",
        "Current weather and TomTom traffic observations for Delhi.",
    )

    # Reload data so the live page can pick up newly appended CSV rows.
    current_weather = parse_dates(
        load_csv(STREAM / "live_weather.csv")
    )

    current_traffic = parse_dates(
        load_csv(STREAM / "live_traffic.csv")
    )

    if not current_traffic.empty:

        current_traffic = current_traffic.copy()

        current_traffic["current_speed"] = numeric_column(
            current_traffic, "current_speed"
        )

        current_traffic["free_flow_speed"] = numeric_column(
            current_traffic, "free_flow_speed"
        )

        denominator = current_traffic["free_flow_speed"].replace(
            0, np.nan
        )

        current_traffic["speed_reduction_pct"] = (
            (denominator - current_traffic["current_speed"])
            / denominator * 100
        ).clip(0, 100).fillna(0)

        current_traffic["congestion_status"] = np.select(
            [
                current_traffic["speed_reduction_pct"] >= 50,
                current_traffic["speed_reduction_pct"] >= 25,
            ],
            ["Heavy", "Moderate"],
            default="Light",
        )

        if "road_closure" in current_traffic.columns:
            current_traffic["road_closure"] = (
                current_traffic["road_closure"]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes"])
            )
        else:
            current_traffic["road_closure"] = False

    display_live_status()

    if not current_weather.empty:

        latest_weather = latest_record(
            current_weather,
            "retrieved_at",
        )

        st.subheader("Current Delhi Weather")

        weather_values = [
            (
                "Temperature",
                latest_weather.get("temperature"),
                "°C",
            ),
            (
                "Humidity",
                latest_weather.get("humidity"),
                "%",
            ),
            (
                "Rainfall",
                latest_weather.get("rainfall"),
                "mm",
            ),
            (
                "Visibility",
                latest_weather.get("visibility"),
                "m",
            ),
            (
                "Wind Speed",
                latest_weather.get("wind_speed"),
                "km/h",
            ),
        ]

        cols = st.columns(5)

        for col, (label, value, unit) in zip(cols, weather_values):

            display_value = (
                f"{value:.1f} {unit}"
                if pd.notna(value)
                else "N/A"
            )

            with col:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="info-label">{label}</div>
                        <div class="info-value">{display_value}</div>
                        <div class="info-detail">Open-Meteo</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.caption(
            "Weather observation: "
            + str(latest_weather.get("observation_time", "Not available"))
            + " | Retrieved: "
            + str(latest_weather.get("retrieved_at", "Not available"))
        )

    else:
        show_empty("No live weather observations available.")

    st.divider()

    st.subheader("TomTom Live Traffic")

    if not current_traffic.empty:

        latest_time = (
            current_traffic["retrieved_at"].max()
            if "retrieved_at" in current_traffic.columns
            else None
        )

        if latest_time is not None:
            st.caption(f"Latest retrieval: {latest_time}")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Locations Reporting",
            current_traffic["blackspot"].nunique(),
        )

        c2.metric(
            "Average Current Speed",
            f"{current_traffic['current_speed'].mean():.1f} km/h",
        )

        c3.metric(
            "Average Free-flow Speed",
            f"{current_traffic['free_flow_speed'].mean():.1f} km/h",
        )

        c4.metric(
            "Reported Road Closures",
            int(current_traffic["road_closure"].sum()),
        )

        st.subheader("Traffic Conditions by Blackspot")

        # Keep the most recent observation for each blackspot.
        if "retrieved_at" in current_traffic.columns:

            latest_per_location = (
                current_traffic.sort_values("retrieved_at")
                .drop_duplicates("blackspot", keep="last")
            )

        else:
            latest_per_location = (
                current_traffic.drop_duplicates(
                    "blackspot",
                    keep="last",
                )
            )

        display_columns = [
            "blackspot",
            "current_speed",
            "free_flow_speed",
            "speed_reduction_pct",
            "congestion_status",
            "confidence",
            "road_closure",
        ]

        display_columns = [
            c for c in display_columns
            if c in latest_per_location.columns
        ]

        display_table = latest_per_location[display_columns].copy()

        if "speed_reduction_pct" in display_table.columns:
            display_table["speed_reduction_pct"] = (
                display_table["speed_reduction_pct"].round(1)
            )

        if "confidence" in display_table.columns:
            display_table["confidence"] = (
                display_table["confidence"].round(3)
            )

        st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "current_speed": st.column_config.NumberColumn(
                    "Current Speed",
                    format="%.1f km/h",
                ),
                "free_flow_speed": st.column_config.NumberColumn(
                    "Free-flow Speed",
                    format="%.1f km/h",
                ),
                "speed_reduction_pct": st.column_config.NumberColumn(
                    "Speed Reduction",
                    format="%.1f%%",
                ),
                "confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    min_value=0,
                    max_value=1,
                    format="%.2f",
                ),
                "road_closure": st.column_config.CheckboxColumn(
                    "Road Closure",
                ),
            },
        )

        left, right = st.columns(2)

        with left:

            st.subheader("Current vs Free-flow Speed")

            speed_chart = latest_per_location.melt(
                id_vars="blackspot",
                value_vars=[
                    "current_speed",
                    "free_flow_speed",
                ],
                var_name="Speed Type",
                value_name="Speed (km/h)",
            )

            fig = px.bar(
                speed_chart,
                x="blackspot",
                y="Speed (km/h)",
                color="Speed Type",
                barmode="group",
                color_discrete_map={
                    "current_speed": "#38bdf8",
                    "free_flow_speed": "#64748b",
                },
            )

            st.plotly_chart(
                style_figure(fig, 380),
                use_container_width=True,
            )

        with right:

            st.subheader("Speed Reduction by Location")

            fig = px.bar(
                latest_per_location.sort_values(
                    "speed_reduction_pct",
                    ascending=False,
                ),
                x="blackspot",
                y="speed_reduction_pct",
                color="congestion_status",
                text="speed_reduction_pct",
                color_discrete_map={
                    "Light": "#34d399",
                    "Moderate": "#fbbf24",
                    "Heavy": "#fb7185",
                },
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside",
            )

            fig.update_layout(
                yaxis_title="Speed reduction (%)",
                xaxis_title="Blackspot",
                showlegend=True,
            )

            st.plotly_chart(
                style_figure(fig, 380),
                use_container_width=True,
            )

        st.divider()

        st.subheader("Live Delhi Traffic Map")

        live_map = make_traffic_map(
            show_live=True,
            show_hotspots=False,
        )

        if live_map is not None:
            st_folium(
                live_map,
                width=None,
                height=550,
                key="live_traffic_map",
            )

        st.caption(
            "Live markers show the seven queried TomTom locations. "
            "Colours represent speed reduction: green = light, "
            "orange = moderate, red = heavy. These are approximate "
            "segment query points, not complete road-level coverage."
        )

        st.subheader("Raw TomTom Observations")

        st.dataframe(
            current_traffic.sort_values(
                "retrieved_at",
                ascending=False,
            ).head(100),
            use_container_width=True,
            hide_index=True,
        )

    else:
        show_empty(
            "No TomTom observations found. Run "
            "python src/live_connectors.py to fetch live traffic."
        )

    st.divider()

    st.subheader("Historical Streaming Pipeline")

    if not stream.empty:

        c1, c2 = st.columns(2)

        c1.metric("Scored Events", f"{len(stream):,}")

        if "segment_id" in stream.columns:
            c2.metric(
                "Tracked Segments",
                stream["segment_id"].nunique(),
            )

        st.dataframe(
            stream.sort_values(
                "timestamp",
                ascending=False,
            ).head(30),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "These records are historical model replay outputs, "
            "not live predictions from TomTom."
        )

    else:
        show_empty("No historical streaming results available.")

    st.divider()

    st.caption(
        "Live observations are appended to local CSV files by "
        "src/live_connectors.py. This page refreshes automatically "
        "when Streamlit fragment support is available."
    )

    if st.button("Refresh Live Data Now", key="refresh_live"):

        st.cache_data.clear()
        st.rerun()


# Use Streamlit native fragment refresh when available.
# The fallback keeps the dashboard compatible with older Streamlit.
fragment = getattr(
    st,
    "fragment",
    lambda **kwargs: (lambda function: function),
)

if page == "Live Monitoring":

    @fragment(run_every="30s")
    def live_page():
        render_live_monitoring()

    live_page()


# =========================================================
# 14. MODEL EVALUATION
# =========================================================

elif page == "Model Evaluation":

    section_heading(
        "MODEL PERFORMANCE",
        "Chronological training and testing evaluation.",
    )

    if model_results.empty:

        show_empty("Model comparison report not found.")

    else:

        st.subheader("Model Comparison")

        st.dataframe(
            model_results,
            use_container_width=True,
            hide_index=True,
        )

        available_metrics = [
            metric for metric in [
                "accuracy",
                "precision_weighted",
                "recall_weighted",
                "f1_weighted",
                "pr_auc",
            ]
            if metric in model_results.columns
        ]

        if available_metrics:

            metric = st.selectbox(
                "Select evaluation metric",
                available_metrics,
            )

            fig = px.bar(
                model_results,
                x="model",
                y=metric,
                color="task",
                barmode="group",
                text_auto=".3f",
                title=f"Model Comparison – {metric}",
                color_discrete_sequence=[
                    "#38bdf8",
                    "#a78bfa",
                ],
            )

            st.plotly_chart(
                style_figure(fig, 420),
                use_container_width=True,
            )

        if "pr_auc" in model_results.columns:

            accident_results = model_results[
                model_results["task"] == "accident"
            ]

            if not accident_results.empty:

                st.subheader("Accident Prediction PR-AUC")

                fig = px.bar(
                    accident_results,
                    x="model",
                    y="pr_auc",
                    color="model",
                    text_auto=".3f",
                    title="Accident Model Precision-Recall AUC",
                )

                st.plotly_chart(
                    style_figure(fig, 350),
                    use_container_width=True,
                )

        st.info(
            "Evaluation uses a chronological split: training before "
            "1 March 2024 and testing during March 2024. "
            "The underlying historical dataset is simulated."
        )


# =========================================================
# 15. FOOTER
# =========================================================

st.markdown("---")

footer_left, footer_right = st.columns(2)

with footer_left:
    st.caption(
        "SMART TRAFFIC INTELLIGENCE | Academic Big Data Analytics Project"
    )

with footer_right:
    st.caption(
        "Synthetic historical simulation · Open-Meteo · TomTom Traffic Flow"
    )
