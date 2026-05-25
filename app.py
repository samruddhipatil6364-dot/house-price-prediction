import streamlit as st
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

# Numeric columns only
numeric_data = data.select_dtypes(
    include=['int64', 'float64']
)

# Target column
target_column = numeric_data.columns[-1]

# Features and Target
X = numeric_data.drop(
    target_column,
    axis=1
)

y = numeric_data[target_column]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)

# Streamlit UI
st.title("🏠 House Price Prediction")

st.header("📊 Model Performance")

st.write("MAE :", mae)

st.write("MSE :", mse)

st.write("RMSE :", rmse)

st.write("R2 SCORE :", r2)

st.write(
    "MODEL ACCURACY :",
    round(r2 * 100, 2),
    "%"
)

st.header("🏡 Enter House Details")

inputs = []

for col in X.columns:

    value = st.number_input(
        f"Enter {col}",
        value=0.0
    )

    inputs.append(value)

if st.button("Predict Price"):

    sample_data = pd.DataFrame(
        [inputs],
        columns=X.columns
    )

    prediction = model.predict(sample_data)

    st.success(
        f"Predicted House Price: {round(prediction[0], 2)}"
    )