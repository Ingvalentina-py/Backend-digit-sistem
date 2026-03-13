# Funciones de cifrado y descifrado

import base64
import codecs

from core.features import spanish_score


# Caesar
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

    return best_text, {
        "shift_detectado": best_shift,
        "caesar_score": round(best_score, 6),
    }


# ROT13
def decrypt_rot13(text: str):
    return codecs.decode(text, "rot_13"), {}


# Base64
def decrypt_base64(text: str):
    raw = text.strip()
    decoded_bytes = base64.b64decode(raw, validate=False)

    try:
        decoded = decoded_bytes.decode("utf-8")
    except UnicodeDecodeError:
        decoded = decoded_bytes.decode("latin-1", errors="replace")

    return decoded, {}


# Intenta decodificar bytes en varias codificaciones
def decode_candidate_bytes(candidate_bytes: bytes):
    candidates = []

    for encoding in ["utf-8", "latin-1"]:
        try:
            decoded = candidate_bytes.decode(encoding)
            candidates.append((decoded, encoding))
        except UnicodeDecodeError:
            continue

    if not candidates:
        decoded = candidate_bytes.decode("latin-1", errors="replace")
        candidates.append((decoded, "latin-1-replace"))

    return candidates


# XOR mejorado
def decrypt_xor_hex_auto(text: str):
    raw = bytes.fromhex(text.strip())

    best_text = text
    best_key = 0
    best_score = -1e9
    best_encoding = "unknown"

    for key in range(256):
        candidate_bytes = bytes(b ^ key for b in raw)

        for decoded, encoding in decode_candidate_bytes(candidate_bytes):
            score = spanish_score(decoded)

            if encoding == "utf-8":
                score += 1.5

            if score > best_score:
                best_score = score
                best_text = decoded
                best_key = key
                best_encoding = encoding

    return best_text, {
        "xor_key_decimal": best_key,
        "xor_key_hex": hex(best_key),
        "xor_score": round(best_score, 6),
        "xor_encoding": best_encoding,
    }


# Descifrado por clase
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