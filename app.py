from flask import Flask, render_template, request
import numpy as np
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model, scaler, and label encoder
model = joblib.load('crop_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve values from the form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Form input array
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Scale the input
        scaled_input = scaler.transform(input_data)

        # Predict and decode
        prediction = model.predict(scaled_input)
        crop_name = label_encoder.inverse_transform(prediction)[0]

        return render_template('index.html', prediction_text=f"🌱 Recommended Crop: {crop_name.upper()}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"❌ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
