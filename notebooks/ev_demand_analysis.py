import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_data


# =========================================================
# CREATE RESULTS FOLDER
# =========================================================

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

volume, occupancy, duration, price, stations = load_data()

print("=" * 60)
print("EV CHARGING DEMAND ANALYSIS")
print("=" * 60)

print("\nDatasets loaded successfully!")

print("Volume:", volume.shape)
print("Occupancy:", occupancy.shape)
print("Duration:", duration.shape)
print("Price:", price.shape)
print("Stations:", stations.shape)


# =========================================================
# CONVERT VOLUME TO LONG FORMAT
# =========================================================

volume_long = volume.melt(
    id_vars=["timestamp"],
    var_name="station_id",
    value_name="charging_demand"
)

volume_long["station_id"] = pd.to_numeric(
    volume_long["station_id"],
    errors="coerce"
)

volume_long = volume_long.dropna(subset=["station_id"])

volume_long["station_id"] = volume_long["station_id"].astype(int)


# =========================================================
# TIME FEATURES
# =========================================================

volume_long["hour"] = volume_long["timestamp"].dt.hour

volume_long["day_of_week"] = volume_long[
    "timestamp"
].dt.day_name()

volume_long["date"] = volume_long[
    "timestamp"
].dt.date


# =========================================================
# BASIC DEMAND STATISTICS
# =========================================================

print("\n" + "=" * 60)
print("DEMAND STATISTICS")
print("=" * 60)

print(volume_long["charging_demand"].describe())


# =========================================================
# HOURLY DEMAND
# =========================================================

hourly_demand = (
    volume_long
    .groupby("hour")["charging_demand"]
    .mean()
    .sort_index()
)

print("\n" + "=" * 60)
print("AVERAGE DEMAND BY HOUR")
print("=" * 60)

print(hourly_demand)

top_hours = (
    hourly_demand
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 highest-demand hours:")
print(top_hours)

hourly_demand.to_csv(
    RESULTS_DIR / "hourly_demand.csv"
)


# =========================================================
# DAY-OF-WEEK DEMAND
# =========================================================

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

daily_demand = (
    volume_long
    .groupby("day_of_week")["charging_demand"]
    .mean()
    .reindex(day_order)
)

print("\n" + "=" * 60)
print("AVERAGE DEMAND BY DAY")
print("=" * 60)

print(daily_demand)

daily_demand.to_csv(
    RESULTS_DIR / "daily_demand.csv"
)


# =========================================================
# STATION-LEVEL DEMAND
# =========================================================

station_demand = (
    volume_long
    .groupby("station_id")["charging_demand"]
    .mean()
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("TOP 10 HIGHEST-DEMAND STATIONS")
print("=" * 60)

print(station_demand.head(10))

print("\n" + "=" * 60)
print("BOTTOM 10 LOWEST-DEMAND STATIONS")
print("=" * 60)

print(station_demand.tail(10))

station_demand.to_csv(
    RESULTS_DIR / "station_demand.csv"
)


# =========================================================
# TOP 10 STATIONS DATAFRAME
# =========================================================

top_stations = (
    station_demand
    .head(10)
    .sort_values()
)

top_stations_df = (
    top_stations
    .reset_index()
)

top_stations_df.columns = [
    "station_id",
    "average_demand"
]

top_stations_df.to_csv(
    RESULTS_DIR / "top_10_stations.csv",
    index=False
)


# =========================================================
# OCCUPANCY ANALYSIS
# =========================================================

print("\n" + "=" * 60)
print("OCCUPANCY ANALYSIS")
print("=" * 60)

occupancy_long = occupancy.melt(
    id_vars=["timestamp"],
    var_name="station_id",
    value_name="occupancy"
)

occupancy_long["station_id"] = pd.to_numeric(
    occupancy_long["station_id"],
    errors="coerce"
)

occupancy_long = occupancy_long.dropna(
    subset=["station_id"]
)

occupancy_long["station_id"] = occupancy_long[
    "station_id"
].astype(int)


# Merge demand and occupancy
demand_occupancy = volume_long[
    ["timestamp", "station_id", "charging_demand"]
].merge(
    occupancy_long,
    on=["timestamp", "station_id"],
    how="inner"
)


# Correlation
occupancy_correlation = (
    demand_occupancy[
        ["charging_demand", "occupancy"]
    ]
    .corr()
    .iloc[0, 1]
)

print(
    f"\nCorrelation between charging demand "
    f"and occupancy: {occupancy_correlation:.4f}"
)


# =========================================================
# DURATION ANALYSIS
# =========================================================

print("\n" + "=" * 60)
print("DURATION ANALYSIS")
print("=" * 60)

duration_long = duration.melt(
    id_vars=["timestamp"],
    var_name="station_id",
    value_name="charging_duration"
)

duration_long["station_id"] = pd.to_numeric(
    duration_long["station_id"],
    errors="coerce"
)

duration_long = duration_long.dropna(
    subset=["station_id"]
)

duration_long["station_id"] = duration_long[
    "station_id"
].astype(int)


demand_duration = volume_long[
    ["timestamp", "station_id", "charging_demand"]
].merge(
    duration_long,
    on=["timestamp", "station_id"],
    how="inner"
)

duration_correlation = (
    demand_duration[
        ["charging_demand", "charging_duration"]
    ]
    .corr()
    .iloc[0, 1]
)

print(
    f"\nCorrelation between charging demand "
    f"and duration: {duration_correlation:.4f}"
)


# =========================================================
# PRICE ANALYSIS
# =========================================================

print("\n" + "=" * 60)
print("PRICE ANALYSIS")
print("=" * 60)

price_long = price.melt(
    id_vars=["timestamp"],
    var_name="station_id",
    value_name="price"
)

price_long["station_id"] = pd.to_numeric(
    price_long["station_id"],
    errors="coerce"
)

price_long = price_long.dropna(
    subset=["station_id"]
)

price_long["station_id"] = price_long[
    "station_id"
].astype(int)


demand_price = volume_long[
    ["timestamp", "station_id", "charging_demand"]
].merge(
    price_long,
    on=["timestamp", "station_id"],
    how="inner"
)

price_correlation = (
    demand_price[
        ["charging_demand", "price"]
    ]
    .corr()
    .iloc[0, 1]
)

print(
    f"\nCorrelation between charging demand "
    f"and price: {price_correlation:.4f}"
)


# =========================================================
# SAVE CORRELATION RESULTS
# =========================================================

correlation_results = pd.DataFrame({
    "factor": [
        "Occupancy",
        "Charging Duration",
        "Price"
    ],
    "correlation_with_demand": [
        occupancy_correlation,
        duration_correlation,
        price_correlation
    ]
})

correlation_results.to_csv(
    RESULTS_DIR / "factor_correlations.csv",
    index=False
)


# =========================================================
# VISUALIZATION 1
# HOURLY DEMAND
# =========================================================

plt.figure(figsize=(10, 5))

plt.plot(
    hourly_demand.index,
    hourly_demand.values,
    marker="o"
)

plt.xlabel("Hour of Day")
plt.ylabel("Average Charging Demand")

plt.title(
    "EV Charging Demand by Hour"
)

plt.xticks(range(24))

plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "hourly_demand.png",
    dpi=300
)

plt.show()


# =========================================================
# VISUALIZATION 2
# TOP 10 STATIONS
# =========================================================

plt.figure(figsize=(10, 6))

plt.barh(
    top_stations.index.astype(str),
    top_stations.values
)

plt.xlabel(
    "Average Charging Demand"
)

plt.ylabel(
    "Station ID"
)

plt.title(
    "Top 10 EV Charging Stations by Demand"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "top_10_stations.png",
    dpi=300
)

plt.show()


# =========================================================
# VISUALIZATION 3
# DEMAND BY DAY
# =========================================================

plt.figure(figsize=(10, 5))

plt.bar(
    daily_demand.index,
    daily_demand.values
)

plt.xlabel("Day of Week")

plt.ylabel(
    "Average Charging Demand"
)

plt.title(
    "EV Charging Demand by Day of Week"
)

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "daily_demand.png",
    dpi=300
)

plt.show()


# =========================================================
# VISUALIZATION 4
# OCCUPANCY VS DEMAND
# =========================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    demand_occupancy["occupancy"],
    demand_occupancy["charging_demand"],
    alpha=0.2
)

plt.xlabel("Occupancy")

plt.ylabel(
    "Charging Demand"
)

plt.title(
    "Charging Demand vs Station Occupancy"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "occupancy_vs_demand.png",
    dpi=300
)

plt.show()


# =========================================================
# VISUALIZATION 5
# DURATION VS DEMAND
# =========================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    demand_duration["charging_duration"],
    demand_duration["charging_demand"],
    alpha=0.2
)

plt.xlabel(
    "Charging Duration"
)

plt.ylabel(
    "Charging Demand"
)

plt.title(
    "Charging Demand vs Charging Duration"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "duration_vs_demand.png",
    dpi=300
)

plt.show()


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

print("\nResults saved to:")
print(RESULTS_DIR)

print("\nGenerated files:")

for file in sorted(RESULTS_DIR.iterdir()):
    print("-", file.name)

print("\nKey findings:")

print(
    f"Peak demand hour: "
    f"{hourly_demand.idxmax()}:00"
)

print(
    f"Peak hourly average demand: "
    f"{hourly_demand.max():.2f}"
)

print(
    f"Highest-demand station: "
    f"{station_demand.idxmax()}"
)

print(
    f"Highest station average demand: "
    f"{station_demand.max():.2f}"
)

print(
    f"Occupancy correlation: "
    f"{occupancy_correlation:.4f}"
)

print(
    f"Duration correlation: "
    f"{duration_correlation:.4f}"
)

print(
    f"Price correlation: "
    f"{price_correlation:.4f}"
)