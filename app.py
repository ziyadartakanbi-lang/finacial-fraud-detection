from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and encoder
model = joblib.load("fraud_model.pkl")
encoder = joblib.load("encoder.pkl")


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    step = float(request.form['step'])
    transaction_type = request.form['type']
    amount = float(request.form['amount'])
    oldbalanceOrg = float(request.form['oldbalanceOrg'])
    newbalanceOrig = float(request.form['newbalanceOrig'])
    oldbalanceDest = float(request.form['oldbalanceDest'])
    newbalanceDest = float(request.form['newbalanceDest'])

    # Encode transaction type
    transaction_type = encoder.transform([transaction_type])[0]
    print("Transaction type:", transaction_type)

    # Create input array
    data = np.array([[
        step,
        transaction_type,
        amount,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest
    ]])

    # Make prediction
    prediction = model.predict(data)
    print("Prediction:", prediction)
    print("Probability:", model.predict_proba(data))
    
    # Prediction confidence
    probability = model.predict_proba(data)[0].max()

    if prediction[0] == 1:
        result = "🚨 Fraudulent Transaction Detected"
    else:
        result = "✅ Legitimate Transaction"

    return render_template(
        "result.html",
        prediction=result,
        probability=round(probability * 100, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)
