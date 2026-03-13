from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model import (
    LABELS,
    analyze_text,
    ensure_model_exists,
    load_dataset,
    load_metrics,
    load_model,
)

app = FastAPI(title="Detector y Descifrador de Cifrado")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://frontend-digit-sistem.vercel.app",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ensure_model_exists()
model = load_model()


class AnalyzeRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {
        "ok": True,
        "message": "API de detección de algoritmos de cifrado funcionando"
    }


@app.get("/labels")
def labels():
    return LABELS


@app.get("/metrics")
def metrics():
    return load_metrics()


@app.get("/sample")
def sample(index: int = 0):
    df = load_dataset()

    if index < 0 or index >= len(df):
        raise HTTPException(status_code=404, detail="Índice fuera de rango")

    row = df.iloc[index]
    real_class = int(row[18])

    return {
        "index": index,
        "text": str(row[0]),
        "real_class": real_class,
        "real_label": LABELS[real_class]
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    text = (req.text or "").strip()

    if not text:
        raise HTTPException(status_code=400, detail="Debes ingresar un texto")

    return analyze_text(model, text)