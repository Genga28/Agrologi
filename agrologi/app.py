from flask import Flask, render_template, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the synthetic dataset
def load_data():
    return pd.read_csv('temperature_data.csv', parse_dates=['date'], index_col='date')

# Load the pre-trained ARIMA model
def load_model():
    return joblib.load('temperature_model.pkl')

# Route for homepage with form
@app.route('/')
def index():
    return render_template('index.html')

# Route for generating temperature prediction
@app.route('/predict_temperature', methods=['GET'])
def predict_temperature():
    try:
        # Load the data and model
        data = load_data()
        model = load_model()

        # Make prediction for the next step
        forecast_steps = 1
        forecast = model.get_forecast(steps=forecast_steps)
        forecast_mean = forecast.predicted_mean

        # Return prediction result as JSON
        return jsonify({"forecast": forecast_mean.values[0]})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
