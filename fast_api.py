from fastapi import FastAPI
from pydantic import BaseModel,Field
from typing import Literal
import joblib
import os
import numpy as np

app=FastAPI(title="Disease Prediction API")

MODEL_PATH = "models/my_model.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found")

model = joblib.load(MODEL_PATH)

label_encoder = joblib.load(LABEL_ENCODER_PATH) if os.path.exists(LABEL_ENCODER_PATH) else None


class PatientInput(BaseModel):
    Age: float = Field(..., example=35)
    Heart_Rate_bpm: float = Field(..., example=82)
    Body_Temperature_C: float = Field(..., example=38.5)
    Oxygen_Saturation_: float = Field(..., example=95)
    Gender_Male: Literal[0, 1] = Field(..., example=1)
    Systolic: float = Field(..., example=120)
    Diastolic: float = Field(..., example=80)

    Body_ache: Literal[0, 1] = Field(..., example=1)
    Cough: Literal[0, 1] = Field(..., example=1)
    Fatigue: Literal[0, 1] = Field(..., example=1)
    Fever: Literal[0, 1] = Field(..., example=1)
    Headache: Literal[0, 1] = Field(..., example=0)
    Runny_nose: Literal[0, 1] = Field(..., example=0)
    Shortness_of_breath: Literal[0, 1] = Field(..., example=0)
    Sore_throat: Literal[0, 1] = Field(..., example=1)

@app.post('/predict')
def predict(data:PatientInput):
    
    
    features = [
        data.Age,
        data.Heart_Rate_bpm,
        data.Body_Temperature_C,
        data.Oxygen_Saturation_,
        data.Gender_Male,
        data.Systolic,
        data.Diastolic,
        data.Body_ache,
        data.Cough,
        data.Fatigue,
        data.Fever,
        data.Headache,
        data.Runny_nose,
        data.Shortness_of_breath,
        data.Sore_throat
    ]
    
    x=np.array(features).reshape(1,-1)
    
    prediction =model.predict(x)[0]
    
    if label_encoder:
        prediction=label_encoder.inverse_transform([prediction])[0]
    
    return {
        "Prediction": prediction,
        
    }
    
    
        
    