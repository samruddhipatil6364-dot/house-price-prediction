from flask import Flask, request
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

app = Flask(__name__)

# ==============================
# LOAD DATASET
# ==============================

data = pd.read_csv("Housing.csv")

# Numeric Columns
numeric_data = data.select_dtypes(
    include=['int64', 'float64']
)

# Target Column
target_column = numeric_data.columns[-1]

# Features and Target
X = numeric_data.drop(
    target_column,
    axis=1
)

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

# Predictions
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)

# ==============================
# HOME PAGE
# ==============================

@app.route('/', methods=['GET', 'POST'])

def home():

    prediction = ""

    if request.method == 'POST':

        values = []

        for col in X.columns:

            value = float(request.form[col])

            values.append(value)

        sample_data = pd.DataFrame(
            [values],
            columns=X.columns
        )

        result = model.predict(sample_data)

        prediction = round(result[0], 2)

    # Create Input Boxes
    inputs = ""

    for col in X.columns:

        inputs += f"""
        <label>{col}</label><br>
        <input type='text' name='{col}' required><br><br>
        """

    return f"""

    <html>

    <head>

        <title>House Price Prediction</title>

    </head>

    <body style="font-family: Arial; padding: 30px;">

        <h1>🏠 House Price Prediction Project</h1>

        <h2>✅ Model Trained Successfully</h2>

        <hr>

        <h2>📊 Model Performance</h2>

        <h3>MAE : {mae}</h3>

        <h3>MSE : {mse}</h3>

        <h3>RMSE : {rmse}</h3>

        <h3>R2 SCORE : {r2}</h3>

        <h3>MODEL ACCURACY : {round(r2 * 100, 2)}%</h3>

        <hr>

        <h2>🏡 Enter House Details</h2>

        <form method="POST">

            {inputs}

            <button type="submit">
                Predict Price
            </button>

        </form>

        <hr>

        <h2>💰 Predicted House Price:</h2>

        <h1>{prediction}</h1>

    </body>

    </html>

    """

# ==============================
# RUN APP
# ==============================

if __name__ == '__main__':

    app.run(debug=True)