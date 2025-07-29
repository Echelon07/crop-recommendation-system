import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model, scaler, and label encoder
model = joblib.load('crop_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['nitrogen']),
            float(request.form['phosphorus']),
            float(request.form['potassium']),
            float(request.form['temperature']),
            float(request.form['humidity']),
            float(request.form['ph']),
            float(request.form['rainfall'])
        ]
        
        # Scale the features
        scaled_features = scaler.transform([features])
        
        # Predict
        prediction_encoded = model.predict(scaled_features)
        prediction = label_encoder.inverse_transform(prediction_encoded)[0]

        # Pass prediction to template
        return render_template('index.html', prediction_text=f"🌱 Recommended Crop: {prediction}")
    
    except Exception as e:
        return render_template('index.html', prediction_text="❌ Error in prediction: " + str(e))
if __name__ == '__main__':
    app.run(debug=True)
