import base64
import codecs
import json
import math
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
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

VOWELS = set("aeiouáéíóúüAEIOUÁÉÍÓÚÜ")
CONSONANTS = set("bcdfghjklmnñpqrstvwxyzBCDFGHJKLMNÑPQRSTVWXYZ")
BASE64_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
HEX_CHARS = set("0123456789abcdefABCDEF")

COMMON_WORDS = [
    " de ", " la ", " el ", " que ", " y ", " en ", " los ", " las ",
    " un ", " una ", " para ", " con ", " por ", " del ", " se ", " al ",
    " modelo ", " datos ", " algoritmo ", " texto ", " aprendizaje ",
    " red ", " clasificación ", " sistema ", " información ", " automático "
]

COMMON_BIGRAMS = [
    "de", "la", "el", "en", "es", "ue", "os", "as", "ra", "ci",
    "qu", "co", "re", "nt", "te", "ar", "or", "to"
]


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


def bigram_ratio(text: str, bigrams):
    text_l = text.lower()
    if len(text_l) < 2:
        return 0.0
    total = len(text_l) - 1
    count = 0
    for bg in bigrams:
        count += text_l.count(bg)
    return count / total if total else 0.0


def unique_bigram_ratio(text: str):
    text_l = text.lower()
    if len(text_l) < 2:
        return 0.0
    bigrams = [text_l[i:i+2] for i in range(len(text_l) - 1)]
    return safe_ratio(len(set(bigrams)), len(bigrams))


def text_spanishness_basic(text: str):
    if not text:
        return 0.0

    text_l = " " + text.lower() + " "
    score = 0.0

    score += sum(1.5 for word in COMMON_WORDS if word in text_l)
    score += bigram_ratio(text_l, COMMON_BIGRAMS) * 12

    alpha_ratio = safe_ratio(sum(ch.isalpha() for ch in text), len(text))
    space_ratio = safe_ratio(sum(ch.isspace() for ch in text), len(text))
    vowel_ratio = safe_ratio(sum(ch in VOWELS for ch in text), len(text))
    printable_ratio = safe_ratio(
        sum(32 <= ord(ch) <= 126 or ch in "áéíóúÁÉÍÓÚñÑüÜ\t\n\r" for ch in text),
        len(text)
    )

    score += alpha_ratio * 4
    score += space_ratio * 2
    score += vowel_ratio * 4
    score += printable_ratio * 2

    score -= safe_ratio(sum(ch in "{}[]<>|^~`" for ch in text), len(text)) * 8
    score -= safe_ratio(sum(ch.isdigit() for ch in text), len(text)) * 2

    return score


def is_probable_base64(text: str):
    raw = text.strip()
    if not raw or len(raw) < 8:
        return False
    if any(ch.isspace() for ch in raw):
        return False
    if not set(raw).issubset(BASE64_CHARS):
        return False
    if len(raw) % 4 != 0:
        return False
    try:
        decoded = base64.b64decode(raw, validate=True)
        return len(decoded) > 0
    except Exception:
        return False


def is_probable_hex(text: str):
    raw = text.strip()
    return bool(raw) and len(raw) % 2 == 0 and all(ch in HEX_CHARS for ch in raw)


def shift_char(ch: str, shift: int) -> str:
    if "a" <= ch <= "z":
        return chr((ord(ch) - ord("a") - shift) % 26 + ord("a"))
    if "A" <= ch <= "Z":
        return chr((ord(ch) - ord("A") - shift) % 26 + ord("A"))
    return ch


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

    return best_text, {"shift_detectado": best_shift, "caesar_score": round(best_score, 6)}


def decrypt_rot13(text: str):
    return codecs.decode(text, "rot_13"), {}


def decrypt_base64(text: str):
    raw = text.strip()
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
        "xor_key_hex": hex(best_key),
        "xor_score": round(best_score, 6),
    }


def spanish_score(text: str) -> float:
    if not text:
        return -1e9
    return text_spanishness_basic(text)


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
    vowels = sum(1 for ch in text if ch in VOWELS)
    consonants = sum(1 for ch in text if ch in CONSONANTS)
    punctuation = sum(1 for ch in text if ch in ".,;:!?¡¿()-_\"'")

    base64_ratio = safe_ratio(sum(1 for ch in text if ch in BASE64_CHARS), total)
    hex_ratio = safe_ratio(sum(1 for ch in text if ch in HEX_CHARS), total)
    printable_ratio = safe_ratio(
        sum(32 <= ord(ch) <= 126 or ch in "áéíóúÁÉÍÓÚñÑüÜ\t\n\r" for ch in text),
        total
    )

    top1, top2 = top_char_freqs(text)

    original_spanish_score = spanish_score(text)
    rot13_candidate = codecs.decode(text, "rot_13")
    rot13_score = spanish_score(rot13_candidate)
    caesar_candidate, caesar_details = decrypt_caesar_auto(text)
    caesar_score = caesar_details["caesar_score"]

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

        # nuevas features
        safe_ratio(vowels, total),
        safe_ratio(consonants, total),
        safe_ratio(punctuation, total),
        base64_ratio,
        hex_ratio,
        printable_ratio,
        bigram_ratio(text, COMMON_BIGRAMS),
        unique_bigram_ratio(text),
        original_spanish_score,
        rot13_score,
        caesar_score,
        1.0 if is_probable_base64(text) else 0.0,
        1.0 if is_probable_hex(text) else 0.0,
    ]


def load_dataset():
    return pd.read_csv(DATASET_PATH, header=None)


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


def build_feature_matrix(texts):
    return np.array([extract_features(str(text)) for text in texts], dtype=float)


def train_and_save():
    df = load_dataset()

    texts = df.iloc[:, 0].astype(str)
    y = df.iloc[:, 18].astype(int)

    X = build_feature_matrix(texts)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.125,  # 10% del total
        random_state=42,
        stratify=y_train_full
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
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
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

    # reentrenar el mejor con el 80% completo
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
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
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


def get_model_probabilities(model, features):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([features])[0]
        return {i: float(prob) for i, prob in enumerate(probs)}
    pred = int(model.predict([features])[0])
    return {i: (1.0 if i == pred else 0.0) for i in LABELS.keys()}


def hybrid_decision(model, text: str, features):
    probs = get_model_probabilities(model, features)

    original_score = spanish_score(text)
    rot13_text = codecs.decode(text, "rot_13")
    rot13_score = spanish_score(rot13_text)
    caesar_text, caesar_details = decrypt_caesar_auto(text)
    caesar_score = caesar_details["caesar_score"]

    base64_hint = 0.0
    xor_hint = 0.0

    if is_probable_base64(text):
        try:
            decoded_b64, _ = decrypt_base64(text)
            base64_hint = 1.2 + max(0.0, spanish_score(decoded_b64)) * 0.04
        except Exception:
            base64_hint = 0.8

    if is_probable_hex(text):
        try:
            xor_text, xor_details = decrypt_xor_hex_auto(text)
            xor_hint = 1.0 + max(0.0, xor_details.get("xor_score", 0.0)) * 0.04
        except Exception:
            xor_hint = 0.5

    decision_scores = {
        0: probs.get(0, 0.0) + max(0.0, original_score) * 0.05,
        1: probs.get(1, 0.0) + max(0.0, caesar_score) * 0.05,
        2: probs.get(2, 0.0) + max(0.0, rot13_score) * 0.05,
        3: probs.get(3, 0.0) + base64_hint,
        4: probs.get(4, 0.0) + xor_hint,
    }

    best_label = max(decision_scores, key=decision_scores.get)

    details = {
        "decision_scores": {LABELS[k]: round(v, 6) for k, v in decision_scores.items()},
        "raw_model_probabilities": {LABELS[k]: round(v, 6) for k, v in probs.items()},
        "spanish_scores": {
            "texto_original": round(original_score, 6),
            "rot13_descifrado": round(rot13_score, 6),
            "caesar_descifrado": round(caesar_score, 6),
        }
    }

    return best_label, details


def analyze_text(model, text: str):
    features = extract_features(text)

    pred, hybrid_details = hybrid_decision(model, text, features)

    probabilities = get_model_probabilities(model, features)
    probabilities_named = {
        LABELS[i]: round(float(probabilities.get(i, 0.0)), 6)
        for i in LABELS.keys()
    }

    decrypted_text, details = decrypt_by_label(pred, text)

    final_details = {}
    final_details.update(details)
    final_details.update(hybrid_details)

    return {
        "input_text": text,
        "features": features,
        "predicted_class": pred,
        "predicted_label": LABELS[pred],
        "probabilities": probabilities_named,
        "decrypted_text": decrypted_text,
        "decryption_details": final_details,
    }