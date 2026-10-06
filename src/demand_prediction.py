from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 60)
print("EV CHARGING DEMAND PREDICTION")
print("=" * 60)

print("\nLoading datasets...")

time = pd.read_csv(DATA_DIR / "time.csv")
volume = pd.read_csv(DATA_DIR / "volume.csv")
occupancy = pd.read_csv(DATA_DIR / "occupancy.csv")
duration = pd.read_csv(DATA_DIR / "duration.csv")
price = pd.read_csv(DATA_DIR / "price.csv")


# =========================================================
# CREATE TIMESTAMP
# =========================================================

time["timestamp"] = pd.to_datetime(
    time[
        [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second"
        ]
    ]
)

timestamp = time["timestamp"]


# =========================================================
# WIDE → LONG FUNCTION
# =========================================================

def convert_to_long(df, value_name):

    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = timestamp

    long_df = df.melt(
        id_vars=["timestamp"],
        var_name="station_id",
        value_name=value_name
    )

    long_df["station_id"] = pd.to_numeric(
        long_df["station_id"],
        errors="coerce"
    )

    long_df = long_df.dropna(
        subset=["station_id"]
    )

    long_df["station_id"] = (
        long_df["station_id"].astype(int)
    )

    return long_df


# =========================================================
# CONVERT DATASETS
# =========================================================

print("\nPreparing datasets...")

volume_long = convert_to_long(
    volume,
    "charging_demand"
)

occupancy_long = convert_to_long(
    occupancy,
    "occupancy"
)

duration_long = convert_to_long(
    duration,
    "charging_duration"
)

price_long = convert_to_long(
    price,
    "price"
)


# =========================================================
# COMBINE DATA
# =========================================================

print("Combining datasets...")

model_data = volume_long.merge(
    occupancy_long,
    on=["timestamp", "station_id"],
    how="inner"
)

model_data = model_data.merge(
    duration_long,
    on=["timestamp", "station_id"],
    how="inner"
)

model_data = model_data.merge(
    price_long,
    on=["timestamp", "station_id"],
    how="inner"
)


print(
    "Combined dataset:",
    model_data.shape
)


# =========================================================
# TIME FEATURES
# =========================================================

model_data["hour"] = (
    model_data["timestamp"].dt.hour
)

model_data["day_of_week"] = (
    model_data["timestamp"].dt.dayofweek
)

model_data["day_of_month"] = (
    model_data["timestamp"].dt.day
)

model_data["month"] = (
    model_data["timestamp"].dt.month
)


# Cyclical time features
model_data["hour_sin"] = np.sin(
    2 * np.pi * model_data["hour"] / 24
)

model_data["hour_cos"] = np.cos(
    2 * np.pi * model_data["hour"] / 24
)

model_data["day_sin"] = np.sin(
    2 * np.pi * model_data["day_of_week"] / 7
)

model_data["day_cos"] = np.cos(
    2 * np.pi * model_data["day_of_week"] / 7
)


# =========================================================
# CLEAN DATA
# =========================================================

model_data = model_data.replace(
    [np.inf, -np.inf],
    np.nan
)

model_data = model_data.dropna()


# =========================================================
# SAMPLE DATA
# =========================================================

MAX_SAMPLES = 300000

if len(model_data) > MAX_SAMPLES:

    model_data = model_data.sample(
        n=MAX_SAMPLES,
        random_state=42
    )

model_data = model_data.reset_index(
    drop=True
)

print(
    "Dataset used for ML:",
    model_data.shape
)


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

np.random.seed(42)

indices = np.arange(
    len(model_data)
)

np.random.shuffle(indices)

split = int(
    len(indices) * 0.80
)

train_indices = indices[:split]
test_indices = indices[split:]

train_data = model_data.iloc[
    train_indices
].copy()

test_data = model_data.iloc[
    test_indices
].copy()


# =========================================================
# STATION TARGET ENCODING
# =========================================================

# Calculate average training demand for each station.
# This avoids treating station IDs as ordinary numbers.

station_mean = (
    train_data
    .groupby("station_id")["charging_demand"]
    .mean()
)

overall_mean = (
    train_data["charging_demand"].mean()
)

train_data["station_demand_mean"] = (
    train_data["station_id"]
    .map(station_mean)
    .fillna(overall_mean)
)

test_data["station_demand_mean"] = (
    test_data["station_id"]
    .map(station_mean)
    .fillna(overall_mean)
)


# =========================================================
# FEATURES
# =========================================================

feature_columns = [
    "occupancy",
    "charging_duration",
    "price",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "station_demand_mean"
]


X_train = train_data[
    feature_columns
].to_numpy(
    dtype=float
)

X_test = test_data[
    feature_columns
].to_numpy(
    dtype=float
)

y_train = train_data[
    "charging_demand"
].to_numpy(
    dtype=float
)

y_test = test_data[
    "charging_demand"
].to_numpy(
    dtype=float
)


# =========================================================
# FEATURE STANDARDIZATION
# =========================================================

mean = X_train.mean(
    axis=0
)

std = X_train.std(
    axis=0
)

std[std == 0] = 1


X_train_scaled = (
    X_train - mean
) / std

X_test_scaled = (
    X_test - mean
) / std


# Add intercept
X_train_scaled = np.column_stack(
    [
        np.ones(len(X_train_scaled)),
        X_train_scaled
    ]
)

X_test_scaled = np.column_stack(
    [
        np.ones(len(X_test_scaled)),
        X_test_scaled
    ]
)


# =========================================================
# RIDGE REGRESSION USING NUMPY
# =========================================================

print("\nTraining regularized regression model...")

alpha = 1.0

identity = np.eye(
    X_train_scaled.shape[1]
)

identity[0, 0] = 0


weights = np.linalg.solve(
    X_train_scaled.T @ X_train_scaled
    + alpha * identity,

    X_train_scaled.T @ y_train
)


print("Model training completed!")


# =========================================================
# PREDICTIONS
# =========================================================

y_pred = (
    X_test_scaled @ weights
)


# =========================================================
# MODEL METRICS
# =========================================================

errors = (
    y_test - y_pred
)

mae = np.mean(
    np.abs(errors)
)

rmse = np.sqrt(
    np.mean(errors ** 2)
)

ss_res = np.sum(
    errors ** 2
)

ss_tot = np.sum(
    (y_test - y_test.mean()) ** 2
)

r2 = 1 - (
    ss_res / ss_tot
)


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(
    f"MAE  : {mae:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)

print(
    f"R²   : {r2:.4f}"
)


# =========================================================
# COEFFICIENT IMPORTANCE
# =========================================================

coefficients = weights[1:]

feature_importance = pd.DataFrame({
    "feature": feature_columns,
    "coefficient": coefficients,
    "absolute_importance": np.abs(coefficients)
})

feature_importance = (
    feature_importance
    .sort_values(
        "absolute_importance",
        ascending=False
    )
)


print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(
    feature_importance[
        [
            "feature",
            "coefficient"
        ]
    ].to_string(index=False)
)


feature_importance.to_csv(
    RESULTS_DIR / "feature_importance.csv",
    index=False
)


# =========================================================
# SAVE PREDICTIONS
# =========================================================

predictions = test_data[
    [
        "timestamp",
        "station_id"
    ]
].copy()

predictions[
    "actual_demand"
] = y_test

predictions[
    "predicted_demand"
] = y_pred

predictions.to_csv(
    RESULTS_DIR / "demand_predictions.csv",
    index=False
)


# =========================================================
# SAVE METRICS
# =========================================================

metrics = pd.DataFrame({
    "metric": [
        "MAE",
        "RMSE",
        "R2"
    ],
    "value": [
        mae,
        rmse,
        r2
    ]
})

metrics.to_csv(
    RESULTS_DIR / "model_metrics.csv",
    index=False
)


# =========================================================
# VISUALIZATION 1
# ACTUAL VS PREDICTED
# =========================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    y_test,
    y_pred,
    alpha=0.20
)

minimum = min(
    y_test.min(),
    y_pred.min()
)

maximum = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum]
)

plt.xlabel(
    "Actual Charging Demand"
)

plt.ylabel(
    "Predicted Charging Demand"
)

plt.title(
    "Actual vs Predicted EV Charging Demand"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "actual_vs_predicted.png",
    dpi=300
)

plt.show()


# =========================================================
# VISUALIZATION 2
# FEATURE COEFFICIENTS
# =========================================================

plot_data = (
    feature_importance
    .sort_values(
        "coefficient"
    )
)

plt.figure(
    figsize=(10, 7)
)

plt.barh(
    plot_data["feature"],
    plot_data["coefficient"]
)

plt.xlabel(
    "Model Coefficient"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Factors Influencing EV Charging Demand"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "feature_coefficients.png",
    dpi=300
)

plt.show()


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("PREDICTION ANALYSIS COMPLETE")
print("=" * 60)

print(
    f"\nMAE  : {mae:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)

print(
    f"R²   : {r2:.4f}"
)

print(
    "\nTop influential features:"
)

print(
    feature_importance.head(5)[
        [
            "feature",
            "coefficient"
        ]
    ].to_string(index=False)
)

print(
    "\nResults saved to:"
)

print(
    RESULTS_DIR
)