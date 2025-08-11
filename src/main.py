from fastapi import FastAPI, HTTPException, UploadFile
import pandas as pd
from joblib import load
import os

app = FastAPI(title="Predictive Maintenance")

MODELS_DIR = "models"
MODEL_MAP = {
    "LR_24h": "model_24h_LR.joblib",
    "RF_24h": "model_24h_RF.joblib",
    "LR_48h": "model_48h_LR.joblib",
    "RF_48h": "model_48h_RF.joblib",
}

# Load all models at startup
models = {}


@app.on_event("startup")
def load_models():
    for name, file in MODEL_MAP.items():
        try:
            model_path = os.path.join(MODELS_DIR, file)
            models[name] = load(model_path)
        except FileNotFoundError:
            raise RuntimeError(f"Model file {file} not found in {MODELS_DIR}")


@app.post("/predict/{model_name}")
async def predict(model_name: str, file: UploadFile):
    if model_name not in MODEL_MAP:
        raise HTTPException(status_code=400, detail="Invalid model name")

    # Read and process input data
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

    # Preprocess data (add your preprocessing steps here)
    # processed_data = preprocess(df)

    # Make prediction
    try:
        model = models[model_name]
        predictions = model.predict(df.values)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
