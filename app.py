from flask import Flask, render_template, request
import joblib
import numpy as np

# Load saved model
model = joblib.load("best_model.pkl")

# Initialize Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Details form route
@app.route('/details')
def details():
    return render_template('details.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get values from form
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)

    # ML model prediction
    ml_prediction = model.predict(final_features)[0]

    # Extract Systolic & Diastolic (11th & 12th features)
    systolic = features[10]
    diastolic = features[11]

    # Rule-based BP classification (American Heart Association Guidelines)
    if systolic >= 180 or diastolic >= 120:
        final_prediction = "Hypertensive Crisis – Seek emergency medical care"
        color = "red"
    elif systolic >= 140 or diastolic >= 90:
        final_prediction = "Stage 2 Hypertension – Immediate medical consultation needed"
        color = "darkred"
    elif systolic >= 130 or diastolic >= 80:
        final_prediction = "Stage 1 Hypertension – Consult Doctor"
        color = "orange"
    elif systolic >= 120 and diastolic < 80:
        final_prediction = "Elevated – Monitor regularly & adopt lifestyle changes"
        color = "blue"
    else:
        final_prediction = "Normal – Maintain a healthy lifestyle"
        color = "green"

    # Combine (optional: show both ML & Rule-based)
    prediction = f"{final_prediction} (ML Prediction: {ml_prediction})"

    return render_template('prediction.html', prediction=prediction, color=color)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
