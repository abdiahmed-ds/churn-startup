import os
from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import joblib
import io

app = FastAPI(title="Churn Prediction Micro-SaaS API", version="1.0")

# Automatically find the directory where main.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "churn_model.pkl")
columns_path = os.path.join(BASE_DIR, "model_columns.pkl")

print(f"Looking for model at: {model_path}")
print(f"Looking for columns at: {columns_path}")

# Load the trained model and column structure safely
try:
    model = joblib.load(model_path)
    model_columns = joblib.load(columns_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    model_columns = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API!"}

@app.post("/predict")
async def predict_churn(file: UploadFile = File(...)):
    if not model or not model_columns:
        raise HTTPException(status_code=500, detail=f"Model files not found at {model_path}. Please run train_model.py first.")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV file.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")
    
    # Preprocess incoming data matching training steps
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        
    df = pd.get_dummies(df, drop_first=True)
    
    # Align columns with what the model expects (fill missing columns with 0)
    df = df.reindex(columns=model_columns, fill_value=0)
    
    # Make predictions and get probabilities
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1] # Probability of churning
    
    df['Predicted_Churn'] = predictions
    df['Churn_Probability'] = probabilities
    
    results = df[['Predicted_Churn', 'Churn_Probability']].to_dict(orient="records")
    return {
        "total_rows_analyzed": len(df),
        "predictions": results
    }