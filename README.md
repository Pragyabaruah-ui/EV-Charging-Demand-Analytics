# EV Charging Demand Analytics

An end-to-end data analytics and machine learning project for understanding and predicting electric vehicle (EV) charging demand.

The project combines Python-based exploratory data analysis, statistical relationship analysis, machine learning prediction, and Power BI dashboard preparation to identify demand patterns and the factors influencing EV charging activity.

## Project Overview

As electric vehicle adoption increases, charging stations need to understand when and where charging demand is likely to be high.

This project analyzes EV charging data to answer questions such as:

- When does EV charging demand peak?
- Which charging stations experience the highest demand?
- How does station occupancy affect charging demand?
- Does charging duration influence demand?
- How does electricity price relate to charging demand?
- Which variables are the strongest predictors of charging demand?
- Can machine learning be used to predict EV charging demand?

The project follows a complete analytics workflow:

Data → Cleaning → Exploratory Analysis → Correlation Analysis → Machine Learning → Prediction Evaluation → Dashboard Preparation

## Objectives

1. Analyze hourly and daily EV charging demand patterns.
2. Identify high-demand charging stations.
3. Study relationships between charging demand and operational factors.
4. Build a machine learning model to predict charging demand.
5. Evaluate the model using MAE, RMSE, and R².
6. Identify the most influential features affecting charging demand.
7. Prepare analytical datasets and visualizations for Power BI.

## Dataset

The project uses EV charging transaction data containing operational and contextual information related to charging sessions.

Key variables include:

- `hour` — hour of the day
- `charging_demand` — charging demand
- `occupancy` — station occupancy level
- `charging_duration` — duration of charging session
- `price` — charging price
- `station` — charging station identifier
- `date/time` — temporal information

The data is processed using Python and stored in the `data/` directory.

## Exploratory Data Analysis

The exploratory analysis focuses on understanding temporal demand patterns and station-level behavior.

### Hourly Demand

The analysis shows significant variation in charging demand throughout the day.

The highest average charging demand occurs around **6 AM**, with an average demand of approximately **36.68**.

The hourly analysis helps identify periods where charging infrastructure may experience higher utilization.

### Station-Level Analysis

Charging stations are compared based on their total and average demand.

The analysis also generates a list of the top-demand stations to help identify locations that may require greater charging capacity.

### Occupancy Analysis

The correlation between charging demand and occupancy is approximately:

**0.4758**

This indicates a moderate positive relationship between station occupancy and charging demand.

### Charging Duration Analysis

The correlation between charging demand and charging duration is approximately:

**0.4990**

This suggests that longer charging sessions are associated with higher charging demand.

### Price Analysis

The correlation between charging demand and price is approximately:

**-0.2795**

This indicates a weak-to-moderate negative relationship, suggesting that higher prices are generally associated with lower charging demand.

## Machine Learning

A supervised machine learning model was developed to predict EV charging demand.

The workflow includes:

1. Feature selection
2. Train-test split
3. Model training
4. Demand prediction
5. Model evaluation
6. Feature influence analysis
7. Actual vs predicted visualization

The machine learning pipeline is implemented in:

`src/demand_prediction.py`

## Model Performance

The current model achieved the following results on the test data:

| Metric | Result |
|---|---:|
| MAE | 20.2795 |
| RMSE | 44.3043 |
| R² | 0.8068 |

An R² value of approximately **0.81** indicates that the model explains a substantial portion of the variation in charging demand.

The MAE indicates that the model's predictions differ from actual demand by approximately **20.28 demand units on average**.

## Feature Influence

The model identified the following variables as the strongest contributors to demand prediction:

| Feature | Coefficient |
|---|---:|
| Station Demand Mean | 83.2069 |
| Charging Duration | 27.7436 |
| Occupancy | -15.9862 |
| Price | -6.8015 |
| Hour | -4.8416 |

Station-level demand characteristics have the strongest influence in the current model, followed by charging duration and occupancy.

## Visualizations

The project generates several analytical visualizations, including:

- Hourly charging demand
- Daily charging demand
- Top charging stations
- Occupancy vs charging demand
- Feature importance
- Feature coefficients
- Actual vs predicted charging demand

The generated outputs are available in the `results/` directory.

## Power BI

Power BI was used to prepare the project for interactive business-oriented analysis.

The Python pipeline generates Power BI-ready datasets including:

- `powerbi_hourly.csv`
- `powerbi_daily.csv`
- `powerbi_stations.csv`
- `powerbi_features.csv`
- `powerbi_metrics.csv`

These datasets can be imported into Power BI to create an interactive EV charging demand dashboard.

The dashboard is designed around key business questions such as:

- What is the overall charging demand?
- What is the peak charging hour?
- Which stations have the highest demand?
- How does demand change throughout the day?
- Which factors influence charging demand?
- How accurate is the demand prediction model?

## Project Structure

```text
EV-Charging-Demand-Analytics/
│
├── data/
│   ├── occupancy.csv
│   ├── price.csv
│   ├── time.csv
│   └── ...
│
├── notebooks/
│
├── results/
│   ├── dashboard/
│   ├── actual_vs_predicted.png
│   ├── daily_demand.png
│   ├── hourly_demand.png
│   ├── occupancy_vs_demand.png
│   ├── top_10_stations.png
│   ├── demand_predictions.csv
│   ├── factor_correlations.csv
│   ├── feature_importance.csv
│   └── model_metrics.csv
│
├── src/
│   ├── data_loader.py
│   ├── ev_demand_analysis.py
│   ├── demand_prediction.py
│   └── dashboard.py
│
├── .gitignore
├── README.md
└── requirements.txt