# 🚲 Bike Sharing Demand Prediction

A Machine Learning project that predicts hourly bike rental demand using a Random Forest Regressor.

## Project Overview

The model uses time, weather, and working-day information to estimate the total number of rented bikes per hour.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Gradio
- Matplotlib (for future analysis/visualization)

## Dataset

The project uses the Bike Sharing Dataset containing hourly and daily rental records. The hourly dataset contains 17,379 records.

Main input features:

- Hour of day
- Temperature (normalized)
- Humidity (normalized)
- Windspeed (normalized)
- Working day
- Weather situation
- Season

Target:

- `cnt` — total number of rented bikes

## Machine Learning Model

**Algorithm:** Random Forest Regressor

The categorical features `hr`, `season`, and `weathersit` are converted using one-hot encoding.

### Current evaluation

Using an 80/20 train-test split with `random_state=42`:

- **RMSE:** 75.91
- **R² Score:** 0.8180

These values are produced by the training script included in this repository.

## Project Structure

```text
bike-sharing-demand-prediction/
├── main.py
├── train_model.py
├── bike_sharing_model.pkl
├── hour.csv
├── day.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model (optional)

```bash
python train_model.py
```

This creates `bike_sharing_model.pkl`.

### 3. Start the Gradio application

```bash
python main.py
```

The application opens a local Gradio interface where you can enter the conditions and get a predicted hourly bike rental count.

## Dataset Source

The Bike Sharing Dataset was developed by Hadi Fanaee-T and Joao Gama and is associated with the Capital Bikeshare system in Washington, D.C.

## Author

Rahul Pinapatruni
