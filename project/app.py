from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from model import LogisticRegression

app = FastAPI(title="Telco Customer Churn Prediction")

# Load model and scaler at startup
model = LogisticRegression()
model.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("features.pkl")

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Telco Customer Churn Prediction API. Navigate to /docs for the Swagger UI."}

@app.post("/predict")
def predict_churn(data: CustomerData):
    # Convert data to dataframe
    df = pd.DataFrame([data.dict()])
    
    # Process just like in training
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Ensure all columns from training match
    for col in columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    # Keep only the right columns in the right order
    df_encoded = df_encoded[columns]
    
    # Scale
    x_scaled = scaler.transform(df_encoded)
    
    # Predict
    prob = model.predict_proba(x_scaled)[0]
    prediction = int(prob >= 0.5)
    
    return {
        "churn_prediction": prediction,
        "churn_probability": float(prob)
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
