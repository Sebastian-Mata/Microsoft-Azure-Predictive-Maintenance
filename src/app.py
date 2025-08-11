from fastapi import FastAPI, HTTPException, UploadFile
from contextlib import asynccontextmanager
import pandas as pd
from joblib import load
import os


MODELS_DIR = "data/models"
MODEL_MAP = {
    "LR_24h": "model_24h_LR.joblib",
    "RF_24h": "model_24h_RF_fast.joblib",
    "LR_48h": "model_48h_LR.joblib",
    "RF_48h": "model_48h_RF_fast.joblib",
}

# Load all models at startup
models = {model: load(os.path.join(MODELS_DIR, file)) for model, file in MODEL_MAP.items()}


# # Parameters for Telemetry feature engineering
SENSORS = ["volt", "rotate", "pressure", "vibration"]
LAGS = [1, 3, 6, 12, 24]
ROLL_MEANS = [3, 6, 12, 24, 48]
ROLL_STDS = [6, 24, 48]
SLOPES_K = [3, 6, 12]

# Engineered features
def add_sensor_features(df, sensors, lags, rmeans, rstds, slopes):
    out = df.copy()
    g = out.groupby("machineID", group_keys=False)

    for c in sensors:
        for k in lags:
            out[f"{c}_lag_{k}h"] = g[c].shift(k)

    for c in sensors:
        for k in rmeans:
            out[f"{c}_mean_{k}h"] = (
                g[c].rolling(window=k, min_periods=k).mean()
                 .reset_index(level=0, drop=True)
            )

    for c in sensors:
        for k in rstds:
            out[f"{c}_std_{k}h"] = (
                g[c].rolling(window=k, min_periods=k).std(ddof=0)
                 .reset_index(level=0, drop=True)
            )

    for c in sensors:
        for k in slopes:
            lag_col = f"{c}_lag_{k}h"
            if lag_col not in out.columns:
                out[lag_col] = g[c].shift(k)
            out[f"{c}_slope_{k}h"] = (out[c] - out[lag_col]) / k

    return out


def process_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Targets
    out["will_fail_in_24h"] = 0
    out["will_fail_in_48h"] = 0

    # Error flags
    for e in ["error1", "error2", "error3", "error4", "error5"]:
        out[f"{e}_last_24h"] = 0
        out[f"{e}_last_48h"] = 0

    # Maintenance features
    for c in ["comp1", "comp2", "comp3", "comp4"]:
        out[f"time_since_maint_{c}_h"] = 48
        out[f"time_since_maint_{c}_d"] = 2

    out = add_sensor_features(out, SENSORS, LAGS, ROLL_MEANS, ROLL_STDS, SLOPES_K)

    exclude = {"datetime", "machineID", "will_fail_in_24h", "will_fail_in_48h"}
    features = [c for c in out.columns if c not in exclude]
    return out.loc[:, features].copy()



@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = {}
    for name, file in MODEL_MAP.items():
        try:
            model_path = os.path.join(MODELS_DIR, file)
            app.state.models[name] = load(model_path)
            print(f"Loaded model:{name}")
        except FileNotFoundError:
            raise RuntimeError(f"Model file {file} not found in {MODELS_DIR}")
    yield
    app.state.models.clear()


app = FastAPI(lifespan=lifespan)

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
    df = process_telemetry(df)

    # Make prediction
    try:
        model = models[model_name]
        predictions = model.predict(df)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
