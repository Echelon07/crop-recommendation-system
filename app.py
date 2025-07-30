from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load your model, encoder, and scaler
model = pickle.load(open('crop_model.pkl', 'rb'))  # correct this!
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))



@app.route('/')
def home():
    return render_template('index.html')  # Make sure this file exists in templates/

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        scaled_input = scaler.transform([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(scaled_input)
        crop = label_encoder.inverse_transform(prediction)[0]

        return render_template('index.html', prediction_text=f"Recommended Crop: {crop}")

if __name__ == "__main__":
    app.run(debug=True)

