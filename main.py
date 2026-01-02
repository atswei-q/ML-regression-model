from fastapi import FastAPI
from pathlib import Path
import joblib
import pandas as pd
# writing API code
app=FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
#load trained model
model=joblib.load(BASE_DIR / "00_model" / "house_price_model.pkl")
#load feature columns
feature_columns=joblib.load(BASE_DIR / "00_model" /"feature_columns.pkl")

@app.get("/")
def home():
    return {"message": "Welcome to the House Price Prediction API"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])

    # One-hot encode input
    df = pd.get_dummies(df)

    # Add missing columns
    df = df.reindex(columns=feature_columns, fill_value=0)
    # Keep correct column order
    df = df[feature_columns]

    prediction = model.predict(df)
    return {"predicted_price": float(prediction[0])}

