import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_FILE = "hour.csv"
MODEL_FILE = "bike_sharing_model.pkl"

features = [
    "temp", "hum", "windspeed", "workingday",
    "weathersit", "season", "hr"
]

data = pd.read_csv(DATA_FILE)

X = data[features].copy()
y = data["cnt"]

# Convert categorical variables into the same one-hot format used by the app.
X = pd.get_dummies(
    X,
    columns=["weathersit", "season", "hr"],
    dtype=int
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = RandomForestRegressor(
    n_estimators=40,
    max_depth=18,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt"
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

joblib.dump(model, MODEL_FILE)
print(f"Model saved as {MODEL_FILE}")
