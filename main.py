import gradio as gr
import joblib
import pandas as pd

MODEL_FILE = "bike_sharing_model.pkl"

model = joblib.load(MODEL_FILE)
all_columns = model.feature_names_in_

def predict_bike_demand(hr, temp, hum, windspeed, workingday, weathersit, season):
    input_data = {
        "temp": [temp],
        "hum": [hum],
        "windspeed": [windspeed],
        "workingday": [workingday],
        f"weathersit_{weathersit}": [1],
        f"season_{season}": [1],
        f"hr_{hr}": [1],
    }

    input_df = pd.DataFrame(0, index=[0], columns=all_columns)

    for col, value in input_data.items():
        if col in input_df.columns:
            input_df[col] = value

    prediction = model.predict(input_df)[0]
    return round(prediction)

app = gr.Interface(
    fn=predict_bike_demand,
    inputs=[
        gr.Slider(0, 23, step=1, label="Hour of Day"),
        gr.Slider(0.0, 1.0, step=0.01, label="Temperature (normalized)"),
        gr.Slider(0.0, 1.0, step=0.01, label="Humidity (normalized)"),
        gr.Slider(0.0, 1.0, step=0.01, label="Windspeed (normalized)"),
        gr.Radio([0, 1], label="Working Day (1 = Yes, 0 = No)"),
        gr.Dropdown(["1", "2", "3", "4"], label="Weather Situation"),
        gr.Dropdown(["1", "2", "3", "4"], label="Season"),
    ],
    outputs="number",
    title="🚲 Bike Sharing Demand Predictor",
    description="Enter time and weather conditions to predict hourly bike rental count."
)

if __name__ == "__main__":
    app.launch()
