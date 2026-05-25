# HOUSE PRICE PREDICTION

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load Dataset
data = pd.read_csv("Housing.csv")

# Show Columns
print("\nCOLUMNS:\n")
print(data.columns)

# Select Numeric Data
numeric_data = data.select_dtypes(
    include=['int64', 'float64']
)

# Last column as target
target_column = numeric_data.columns[-1]

print("\nTARGET COLUMN :", target_column)

# Features
X = numeric_data.drop(
    target_column,
    axis=1
)

# Target
y = numeric_data[target_column]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

print("\nMODEL TRAINED SUCCESSFULLY\n")

# Prediction
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)

# Results
print("\nMODEL PERFORMANCE\n")

print("MAE :", mae)

print("MSE :", mse)

print("RMSE :", rmse)

print("R2 SCORE :", r2)

print(
    "\nMODEL ACCURACY :",
    round(r2 * 100, 2),
    "%"
)