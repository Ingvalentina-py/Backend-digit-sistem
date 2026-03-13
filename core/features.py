# Extracción de características

import math
from collections import Counter

import numpy as np

from core.config import (
    BASE64_CHARS,
    COMMON_BIGRAMS,
    COMMON_WORDS,
    CONSONANTS,
    HEX_CHARS,
    VALID_TEXT_CHARS,
    VOWELS,
)


def safe_ratio(value, total: int) -> float:
    return float(value) / total if total else 0.0


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


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


# Score de parecido a español
def spanish_score(text: str) -> float:
    if not text:
        return -1e9

    total = len(text)
    text_l = " " + text.lower() + " "
    score = 0.0

    score += sum(1.5 for word in COMMON_WORDS if word in text_l)
    score += bigram_ratio(text_l, COMMON_BIGRAMS) * 12

    alpha_ratio = safe_ratio(sum(ch.isalpha() for ch in text), total)
    space_ratio = safe_ratio(sum(ch.isspace() for ch in text), total)
    vowel_ratio = safe_ratio(sum(ch in VOWELS for ch in text), total)
    printable_ratio = safe_ratio(
        sum(32 <= ord(ch) <= 126 or ch in "áéíóúÁÉÍÓÚñÑüÜ\t\n\r" for ch in text),
        total
    )
    valid_char_ratio = safe_ratio(sum(ch in VALID_TEXT_CHARS for ch in text), total)
    weird_ratio = safe_ratio(sum(ord(ch) < 32 and ch not in "\t\n\r" for ch in text), total)

    score += alpha_ratio * 4
    score += space_ratio * 2
    score += vowel_ratio * 4
    score += printable_ratio * 2
    score += valid_char_ratio * 5

    score -= weird_ratio * 20
    score -= safe_ratio(sum(ch in "{}[]<>|^~`" for ch in text), total) * 8
    score -= safe_ratio(sum(ch.isdigit() for ch in text), total) * 2

    return score


# Detección básica de Base64
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
    return True


# Detección básica de hexadecimal
def is_probable_hex(text: str):
    raw = text.strip()
    return bool(raw) and len(raw) % 2 == 0 and all(ch in HEX_CHARS for ch in raw)


# Features del texto
def extract_features(text: str, rot13_score: float, caesar_score: float):
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
    original_score = spanish_score(text)

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
        safe_ratio(vowels, total),
        safe_ratio(consonants, total),
        safe_ratio(punctuation, total),
        base64_ratio,
        hex_ratio,
        printable_ratio,
        bigram_ratio(text, COMMON_BIGRAMS),
        unique_bigram_ratio(text),
        original_score,
        rot13_score,
        caesar_score,
        1.0 if is_probable_base64(text) else 0.0,
        1.0 if is_probable_hex(text) else 0.0,
    ]