# Servicio principal de análisis

import codecs

from core.config import LABELS
from core.crypto_utils import (
    decrypt_base64,
    decrypt_by_label,
    decrypt_caesar_auto,
    decrypt_xor_hex_auto,
)
from core.features import extract_features, is_probable_base64, is_probable_hex, spanish_score
from core.training import load_model


# Modelo en memoria
model = None


# Inicializar modelo
def init_model():
    global model
    if model is None:
        model = load_model()
    return model


# Probabilidades del modelo
def get_model_probabilities(model, features):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([features])[0]
        return {i: float(prob) for i, prob in enumerate(probs)}

    pred = int(model.predict([features])[0])
    return {i: (1.0 if i == pred else 0.0) for i in LABELS.keys()}


# Decisión híbrida
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


# Analizar texto
def analyze_text(text: str):
    model = init_model()

    rot13_score = spanish_score(codecs.decode(text, "rot_13"))
    _, caesar_details = decrypt_caesar_auto(text)
    caesar_score = caesar_details["caesar_score"]

    features = extract_features(
        text,
        rot13_score=rot13_score,
        caesar_score=caesar_score
    )

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