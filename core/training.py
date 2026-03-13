# Entrenamiento del modelo

import json
import os

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.config import DATASET_PATH, LABELS, METRICS_PATH, MODEL_PATH, MODELS_DIR
from core.crypto_utils import decrypt_caesar_auto
from core.features import extract_features

os.makedirs(MODELS_DIR, exist_ok=True)


# Cargar dataset
def load_dataset():
    return pd.read_csv(DATASET_PATH, header=None)


# Construir features desde texto
def build_feature_matrix(texts):
    X = []
    for text in texts:
        text = str(text)
        rot13_score = 0.0
        _, caesar_details = decrypt_caesar_auto(text)
        caesar_score = caesar_details["caesar_score"]
        features = extract_features(text, rot13_score=rot13_score, caesar_score=caesar_score)
        X.append(features)
    return np.array(X, dtype=float)


# Evaluar modelo
def evaluate_model(model, X_test, y_test, train_size: int, val_accuracy: float, model_name: str):
    pred = model.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "report": classification_report(y_test, pred, output_dict=True),
        "labels": LABELS,
        "train_size": train_size,
        "test_size": int(len(y_test)),
        "validation_accuracy": float(val_accuracy),
        "selected_model": model_name,
        "feature_count": int(X_test.shape[1]),
    }


# Entrenar y guardar
def train_and_save():
    df = load_dataset()

    texts = df.iloc[:, 0].astype(str)
    y = df.iloc[:, 18].astype(int)

    X = build_feature_matrix(texts)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.125, random_state=42, stratify=y_train_full
    )

    candidates = {
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                alpha=0.0003,
                learning_rate="adaptive",
                max_iter=700,
                early_stopping=True,
                n_iter_no_change=25,
                random_state=42,
            ))
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )
    }

    best_name = None
    best_model = None
    best_val_acc = -1.0

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_name = name
            best_model = model

    if best_name == "mlp":
        final_model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                alpha=0.0003,
                learning_rate="adaptive",
                max_iter=700,
                early_stopping=True,
                n_iter_no_change=25,
                random_state=42,
            ))
        ])
    else:
        final_model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )

    final_model.fit(X_train_full, y_train_full)
    dump(final_model, MODEL_PATH)

    metrics = evaluate_model(
        final_model,
        X_test,
        y_test,
        train_size=len(X_train_full),
        val_accuracy=best_val_acc,
        model_name=best_name
    )

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


# Asegurar modelo
def ensure_model_exists():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METRICS_PATH):
        return train_and_save()
    return None


# Cargar modelo
def load_model():
    return load(MODEL_PATH)


# Cargar métricas
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return train_and_save()

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)