from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_data():
    # Load all datasets
    time = pd.read_csv(DATA_DIR / "time.csv")
    volume = pd.read_csv(DATA_DIR / "volume.csv")
    occupancy = pd.read_csv(DATA_DIR / "occupancy.csv")
    duration = pd.read_csv(DATA_DIR / "duration.csv")
    price = pd.read_csv(DATA_DIR / "price.csv")
    stations = pd.read_csv(DATA_DIR / "stations.csv")

    # Create proper datetime from time.csv
    time["timestamp"] = pd.to_datetime(
        time[["year", "month", "day", "hour", "minute", "second"]]
    )

    # Keep only the real timestamp
    timestamp = time["timestamp"]

    # Replace the existing numeric timestamp in each dataset
    # with the actual datetime timestamp
    datasets = {
        "volume": volume,
        "occupancy": occupancy,
        "duration": duration,
        "price": price,
    }

    for name, df in datasets.items():
        if "timestamp" in df.columns:
            df["timestamp"] = timestamp
        else:
            df.insert(0, "timestamp", timestamp)

    return volume, occupancy, duration, price, stations


if __name__ == "__main__":
    volume, occupancy, duration, price, stations = load_data()

    print("EV charging datasets loaded successfully!")

    print("Volume:", volume.shape)
    print("Occupancy:", occupancy.shape)
    print("Duration:", duration.shape)
    print("Price:", price.shape)
    print("Stations:", stations.shape)

    print("\nTime range:")
    print(volume["timestamp"].min(), "to", volume["timestamp"].max())

    print("\nFirst 5 timestamps:")
    print(volume["timestamp"].head())