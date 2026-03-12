from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conlist
import numpy as np

from model import (
    load_data, ensure_models_exist, load_models,
    evaluate_models, predict_from_pixels
)

app = FastAPI(title="Digits API (Optdigits 8x8)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

startup_metrics = ensure_models_exist()
mlp, svm_model = load_models()

X_train, y_train, X_test, y_test = load_data()

class PredictRequest(BaseModel):
    model: str  # "mlp" o "svm"
    pixels: conlist(float, min_length=64, max_length=64)

@app.get("/")
def root():
    return {"ok": True, "message": "Digits API running"}

@app.get("/metrics")
def metrics():
    return evaluate_models(mlp, svm_model, X_test, y_test)

@app.get("/sample")
def sample(index: int = 0, split: str = "test"):
    if split == "train":
        x = X_train[index].tolist()
        y = int(y_train[index])
    else:
        x = X_test[index].tolist()
        y = int(y_test[index])

    grid = np.array(x).reshape(8, 8).tolist()
    return {"index": index, "split": split, "pixels": x, "grid8x8": grid, "label": y}

@app.post("/predict")
def predict(req: PredictRequest):
    model = mlp if req.model.lower() == "mlp" else svm_model
    pred, proba = predict_from_pixels(model, req.pixels)
    return {"model": req.model, "prediction": pred, "proba": proba}