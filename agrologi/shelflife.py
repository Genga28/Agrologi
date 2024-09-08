# app.py
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('shelf_life_model.pkl')

@app.route('/predict_shelf_life', methods=['POST'])
def predict_shelf_life():
    data = request.json
    features = [data['feature1'], data['feature2'], data['feature3']]
    prediction = model.predict([features])[0]
    return jsonify({'shelf_life': prediction})

if __name__ == '__main__':
    app.run(debug=True)