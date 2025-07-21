from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Corrected feature order
        feature_order = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                         'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

        # Extract features in the correct order
        features = [float(request.form[col]) for col in feature_order]

        # Scale input
        scaled_features = scaler.transform([features])

        # Predict
        prediction = model.predict(scaled_features)[0]

        # Interpret result
        result = "High risk of Heart Disease" if prediction >= 1 else "Low risk of Heart Disease"

        return render_template("result.html", prediction_text=result)

    except Exception as e:
        return f"Error: {e}. Please check your entries and ensure they are numeric."

if __name__ == "__main__":
    app.run(debug=True)
