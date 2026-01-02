from fastapi import FastAPI
import joblib
import pandas as pd
# writing API code
app=FastAPI()

#load feature columns
feature_columns=joblib.load("00_model/feature_columns.pkl")

#load trained model
model=joblib.load("00_model/house_price_model.pkl")

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
