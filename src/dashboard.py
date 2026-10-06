from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DASHBOARD_DIR = RESULTS_DIR / "dashboard"

DASHBOARD_DIR.mkdir(exist_ok=True)


# =========================================================
# LOAD RESULT FILES
# =========================================================

print("=" * 60)
print("CREATING POWER BI DASHBOARD PACKAGE")
print("=" * 60)

hourly = pd.read_csv(
    RESULTS_DIR / "hourly_demand.csv"
)

daily = pd.read_csv(
    RESULTS_DIR / "daily_demand.csv"
)

stations = pd.read_csv(
    RESULTS_DIR / "top_10_stations.csv"
)

metrics = pd.read_csv(
    RESULTS_DIR / "model_metrics.csv"
)

features = pd.read_csv(
    RESULTS_DIR / "feature_importance.csv"
)


# =========================================================
# SHOW FILE STRUCTURES
# =========================================================

print("\nLoaded files successfully.")

print("\nHourly columns:")
print(hourly.columns.tolist())

print("\nDaily columns:")
print(daily.columns.tolist())

print("\nStation columns:")
print(stations.columns.tolist())

print("\nMetrics:")
print(metrics)

print("\nFeature columns:")
print(features.columns.tolist())


# =========================================================
# 1. HOURLY DEMAND CHART
# =========================================================

plt.figure(figsize=(10, 5))

plt.plot(
    hourly.iloc[:, 0],
    hourly.iloc[:, 1],
    marker="o"
)

plt.title(
    "EV Charging Demand by Hour"
)

plt.xlabel(
    "Hour of Day"
)

plt.ylabel(
    "Average Charging Demand"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    DASHBOARD_DIR / "01_hourly_demand.png",
    dpi=200
)

plt.close()


# =========================================================
# 2. DAILY DEMAND CHART
# =========================================================

plt.figure(figsize=(10, 5))

plt.plot(
    daily.iloc[:, 0],
    daily.iloc[:, 1],
    marker="o"
)

plt.title(
    "EV Charging Demand Over Time"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Charging Demand"
)

plt.xticks(
    rotation=45
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    DASHBOARD_DIR / "02_daily_demand.png",
    dpi=200
)

plt.close()


# =========================================================
# 3. TOP STATIONS
# =========================================================

plt.figure(figsize=(9, 6))

plt.barh(
    stations.iloc[:, 0].astype(str),
    stations.iloc[:, 1]
)

plt.title(
    "Top 10 EV Charging Stations by Demand"
)

plt.xlabel(
    "Average Charging Demand"
)

plt.ylabel(
    "Station ID"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    DASHBOARD_DIR / "03_top_stations.png",
    dpi=200
)

plt.close()


# =========================================================
# 4. FEATURE IMPORTANCE
# =========================================================

feature_plot = features.copy()

# Use the last column as importance/coefficient
importance_column = feature_plot.columns[-1]

feature_plot = feature_plot.sort_values(
    importance_column
)

plt.figure(figsize=(9, 6))

plt.barh(
    feature_plot.iloc[:, 0],
    feature_plot[importance_column]
)

plt.title(
    "Factors Influencing EV Charging Demand"
)

plt.xlabel(
    importance_column
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()

plt.savefig(
    DASHBOARD_DIR / "04_feature_importance.png",
    dpi=200
)

plt.close()


# =========================================================
# 5. READ MODEL METRICS SAFELY
# =========================================================

# Handle either:
# metric,value
# OR
# MAE,RMSE,R2

if (
    "metric" in metrics.columns
    and "value" in metrics.columns
):

    metric_values = dict(
        zip(
            metrics["metric"],
            metrics["value"]
        )
    )

    mae = float(
        metric_values["MAE"]
    )

    rmse = float(
        metric_values["RMSE"]
    )

    r2 = float(
        metric_values["R2"]
    )

else:

    # Metrics stored as columns
    mae = float(
        metrics["MAE"].iloc[0]
    )

    rmse = float(
        metrics["RMSE"].iloc[0]
    )

    r2 = float(
        metrics["R2"].iloc[0]
    )


# =========================================================
# 6. CALCULATE KPI VALUES
# =========================================================

peak_hour = hourly.iloc[
    hourly.iloc[:, 1].idxmax(),
    0
]

average_demand = hourly.iloc[
    :, 1
].mean()


# =========================================================
# 7. CREATE KPI TABLE
# =========================================================

kpis = pd.DataFrame({

    "Metric": [
        "MAE",
        "RMSE",
        "R2",
        "Peak Hour",
        "Average Demand"
    ],

    "Value": [
        mae,
        rmse,
        r2,
        peak_hour,
        average_demand
    ]
})


kpis.to_csv(
    DASHBOARD_DIR / "dashboard_kpis.csv",
    index=False
)


# =========================================================
# 8. CREATE POWER BI DATA FILES
# =========================================================

hourly.to_csv(
    DASHBOARD_DIR / "powerbi_hourly.csv",
    index=False
)

daily.to_csv(
    DASHBOARD_DIR / "powerbi_daily.csv",
    index=False
)

stations.to_csv(
    DASHBOARD_DIR / "powerbi_stations.csv",
    index=False
)

features.to_csv(
    DASHBOARD_DIR / "powerbi_features.csv",
    index=False
)

metrics.to_csv(
    DASHBOARD_DIR / "powerbi_metrics.csv",
    index=False
)


# =========================================================
# 9. FINAL OUTPUT
# =========================================================

print("\n" + "=" * 60)
print("POWER BI DASHBOARD PACKAGE CREATED")
print("=" * 60)

print("\nModel metrics:")
print(
    f"MAE  : {mae:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)

print(
    f"R²   : {r2:.4f}"
)

print(
    f"Peak hour : {peak_hour}"
)

print(
    f"Average demand : {average_demand:.4f}"
)

print("\nDashboard files:")

for file in sorted(
    DASHBOARD_DIR.iterdir()
):
    print(
        "-",
        file.name
    )

print(
    "\nDashboard preparation complete!"
)

print(
    "\nLocation:",
    DASHBOARD_DIR
)