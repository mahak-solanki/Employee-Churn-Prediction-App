from flask import Flask, render_template, request
import numpy as np
import joblib
import pickle

app = Flask(__name__)

# Load model files
model = joblib.load("xgboost_churn_model.pkl")
scaler = pickle.load(open("scalar.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        input_data = {
            "Age": float(request.form["Age"]),
            "Tenure": float(request.form["Tenure"]),
            "Salary": float(request.form["Salary"]),
            "Overtime Hours": float(request.form["Overtime_Hours"]),
            "Satisfaction Level": float(request.form["Satisfaction_Level"]),
            "Promotions": float(request.form["Promotions"]),
            "Manager Feedback Score": float(request.form["Manager_Feedback_Score"]),
            "Department": request.form["Department"]
        }

        # Encode Department
        input_data["Department"] = encoders["Department"].transform(
            [input_data["Department"]]
        )[0]

        # Correct feature order (VERY IMPORTANT)
        feature_order = [
            "Age", "Tenure", "Salary", "Overtime Hours",
            "Satisfaction Level", "Promotions",
            "Manager Feedback Score", "Department"
        ]

        X = np.array([input_data[f] for f in feature_order]).reshape(1, -1)

        # Scale
        X = scaler.transform(X)

        # Predict
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]

        if probability > 0.5:
            result = f"Result💡 : Employee is likely to Churn✅"
        else:
            result = f"Result💡 : Employee is NOT likely to Churn❌"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(host = "0.0.0.0" , debug=True)

