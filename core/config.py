# Configuración general

import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

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

COMMON_WORDS = [
    " de ", " la ", " el ", " que ", " y ", " en ", " los ", " las ",
    " un ", " una ", " para ", " con ", " por ", " del ", " se ", " al ",
    " modelo ", " datos ", " algoritmo ", " texto ", " aprendizaje ",
    " red ", " clasificación ", " sistema ", " información ", " automático ",
    " aplicación ", " web ", " muestra ", " detectado ", " descifrado "
]

COMMON_BIGRAMS = [
    "de", "la", "el", "en", "es", "ue", "os", "as", "ra", "ci",
    "qu", "co", "re", "nt", "te", "ar", "or", "to"
]

VOWELS = set("aeiouáéíóúüAEIOUÁÉÍÓÚÜ")
CONSONANTS = set("bcdfghjklmnñpqrstvwxyzBCDFGHJKLMNÑPQRSTVWXYZ")
BASE64_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
HEX_CHARS = set("0123456789abcdefABCDEF")
VALID_TEXT_CHARS = set(
    "abcdefghijklmnñopqrstuvwxyzABCDEFGHIJKLMNÑOPQRSTUVWXYZ"
    "áéíóúÁÉÍÓÚüÜ "
    ".,;:!?¡¿()-_\"'0123456789"
)