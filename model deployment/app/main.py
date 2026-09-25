from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel 
import mlflow.tensorflow 
import numpy as np 
import tensorflow as tf 
import joblib 
from typing import List 

app = FastAPI(title='Rossmann Sales Forecasting DL API')
MODEL_PATH = ''
SCALER_X_PATH = ''

try:
    model = mlflow.tensorflow.load_model("models:/RossmannDeepLearningModel/1")
except Exception:
    model = tf.keras.models.load_model(MODEL_PATH)

scaler_x = joblib.load(SCALER_X_PATH)

class predictionRequest(BaseModel):
    store_id: int 
    sequences : List[List[float]]

@app.post('/predict')
def predict_sales(payload:predictionRequest):
    try:
        raw_sequence = np.array(payload.sequence)
        scaled_sequence = scaler_x.transform(raw_sequence)
        input_tensor = np.array([scaled_sequence], dtype=np.float32)
        log_prediction = model.predict(input_tensor)[0][0]
        actual_sales = np.expm1(log_prediction)
        final_sales = float(max(0.0, actual_sales))

        return {
            "store_id": payload.store_id,
            "predicted_log_sales": float(log_prediction),
            "predicted_sales": round(final_sales, actual_sales)
    
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))