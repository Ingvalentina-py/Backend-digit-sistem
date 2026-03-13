import base64
import codecs
import json
import math
import os
from collections import Counter

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

DATASET_PATH = os.path.join(DATA_DIR, "cifrado_mlp.data")
MODEL_PATH = os.path.join(MODELS_DIR, "cipher_mlp.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "cipher_metrics.json")

LABELS = {
    0: "Texto plano",
    1: "Caesar Cipher",
    2: "ROT13",
    3: "Base64",
    4: "XOR",
}


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def safe_ratio(value, total: int) -> float:
    return float(value) / total if total else 0.0


def top_char_freqs(text: str):
    if not text:
        return 0.0, 0.0
    counts = Counter(text)
    freqs = sorted((count / len(text) for count in counts.values()), reverse=True)
    top1 = freqs[0] if freqs else 0.0
    top2 = freqs[1] if len(freqs) > 1 else 0.0
    return top1, top2


def extract_features(text: str):
    text = text or ""
    total = len(text)
    ascii_values = [ord(ch) for ch in text] or [0]

    mayus = sum(1 for ch in text if ch.isupper())
    minus = sum(1 for ch in text if ch.islower())
    nums = sum(1 for ch in text if ch.isdigit())
    espacios = sum(1 for ch in text if ch.isspace())
    alpha = sum(1 for ch in text if ch.isalpha())
    especiales = total - mayus - minus - nums - espacios

    top1, top2 = top_char_freqs(text)

    return [
        total,
        shannon_entropy(text),
        safe_ratio(mayus, total),
        safe_ratio(minus, total),
        safe_ratio(nums, total),
        safe_ratio(espacios, total),
        safe_ratio(especiales, total),
        min(ascii_values),
        max(ascii_values),
        float(np.mean(ascii_values)),
        float(np.var(ascii_values)),
        top1,
        top2,
        1.0 if text.endswith("=") else 0.0,
        safe_ratio(alpha, total),
        len(set(text)),
        safe_ratio(len(set(text)), total),
    ]


def load_dataset():
    return pd.read_csv(DATASET_PATH, header=None)


def evaluate_model(model, X_test, y_test, train_size: int):
    pred = model.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "report": classification_report(y_test, pred, output_dict=True),
        "labels": LABELS,
        "train_size": train_size,
        "test_size": int(len(y_test)),
    }


def train_and_save():
    df = load_dataset()

    # Col 0 = texto, cols 1..17 = features, col 18 = clase
    X = df.iloc[:, 1:18].astype(float)
    y = df.iloc[:, 18].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0005,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=42,
        ))
    ])

    model.fit(X_train, y_train)
    dump(model, MODEL_PATH)

    metrics = evaluate_model(model, X_test, y_test, train_size=len(X_train))
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def ensure_model_exists():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METRICS_PATH):
        return train_and_save()
    return None


def load_model():
    return load(MODEL_PATH)


def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return train_and_save()
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def shift_char(ch: str, shift: int) -> str:
    if "a" <= ch <= "z":
        return chr((ord(ch) - ord("a") - shift) % 26 + ord("a"))
    if "A" <= ch <= "Z":
        return chr((ord(ch) - ord("A") - shift) % 26 + ord("A"))
    return ch


COMMON_WORDS = [
    " de ", " la ", " el ", " que ", " y ", " en ", " los ", " las ",
    " un ", " una ", " para ", " con ", " por ", " del ",
    " modelo ", " datos ", " algoritmo ", " texto ",
    " aprendizaje ", " red ", " clasificación "
]


def spanish_score(text: str) -> float:
    if not text:
        return -1e9

    text_l = " " + text.lower() + " "
    score = 0.0

    score += sum(5 for word in COMMON_WORDS if word in text_l)

    printable_ratio = sum(
        32 <= ord(ch) <= 126 or ch in "áéíóúÁÉÍÓÚñÑüÜ\t\n\r"
        for ch in text
    ) / len(text)
    score += printable_ratio * 6

    alpha_space_ratio = sum(ch.isalpha() or ch.isspace() for ch in text) / len(text)
    score += alpha_space_ratio * 6

    score -= sum(ch in "{}[]<>|^~`" for ch in text) * 0.15
    return score


def decrypt_caesar_auto(text: str):
    best_shift = 0
    best_text = text
    best_score = -1e9

    for shift in range(1, 26):
        candidate = "".join(shift_char(ch, shift) for ch in text)
        score = spanish_score(candidate)
        if score > best_score:
            best_score = score
            best_shift = shift
            best_text = candidate

    return best_text, {"shift_detectado": best_shift}


def decrypt_rot13(text: str):
    return codecs.decode(text, "rot_13"), {}


def decrypt_base64(text: str):
    raw = text.strip()
    padding_needed = (-len(raw)) % 4
    raw += "=" * padding_needed

    decoded_bytes = base64.b64decode(raw, validate=False)

    try:
        decoded = decoded_bytes.decode("utf-8")
    except UnicodeDecodeError:
        decoded = decoded_bytes.decode("latin-1", errors="replace")

    return decoded, {}


def decrypt_xor_hex_auto(text: str):
    raw = bytes.fromhex(text.strip())

    best_text = text
    best_key = 0
    best_score = -1e9

    for key in range(256):
        candidate_bytes = bytes(b ^ key for b in raw)
        try:
            candidate = candidate_bytes.decode("utf-8")
        except UnicodeDecodeError:
            candidate = candidate_bytes.decode("latin-1", errors="replace")

        score = spanish_score(candidate)
        if score > best_score:
            best_score = score
            best_text = candidate
            best_key = key

    return best_text, {
        "xor_key_decimal": best_key,
        "xor_key_hex": hex(best_key)
    }


def decrypt_by_label(label: int, text: str):
    if label == 0:
        return text, {}
    if label == 1:
        return decrypt_caesar_auto(text)
    if label == 2:
        return decrypt_rot13(text)
    if label == 3:
        return decrypt_base64(text)
    if label == 4:
        return decrypt_xor_hex_auto(text)

    return text, {}


def analyze_text(model, text: str):
    features = extract_features(text)
    pred = int(model.predict([features])[0])

    probabilities = None
    confidence = None

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([features])[0]
        probabilities = {
            LABELS[i]: round(float(prob), 6)
            for i, prob in enumerate(probs)
        }
        confidence = round(float(max(probs)), 6)

    decrypted_text, details = decrypt_by_label(pred, text)

    return {
        "input_text": text,
        "features": features,
        "predicted_class": pred,
        "predicted_label": LABELS[pred],
        "confidence": confidence,
        "probabilities": probabilities,
        "decrypted_text": decrypted_text,
        "decryption_details": details,
    }