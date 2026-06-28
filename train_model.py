import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("creditcard.csv")

# Encode transaction type
encoder = LabelEncoder()
df["type"] = encoder.fit_transform(df["type"])

# Features
X = df[[
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]]

# Target
y = df["isFraud"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Test
prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("Accuracy:", accuracy)
print(classification_report(y_test, prediction))

# Save model
joblib.dump(model, "fraud_model.pkl")
joblib.dump(encoder, "encoder.pkl")

print("Model saved successfully!")

fraud = df[df["isFraud"] == 1].head(5)
print("\nSample Fraud Transactions:")
print(fraud)

# Test the first fraud sample
sample = X[y == 1].iloc[[0]]

pred = model.predict(sample)
prob = model.predict_proba(sample)

print("\nPrediction on first fraud sample:", pred)
print("Probability:", prob)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

print("\nFirst Fraud Record:")
print(df[df["isFraud"] == 1].head(1))