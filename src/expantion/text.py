from __future__ import annotations

import re
import unicodedata

from src.expantion.constants import WORD_RE, get_stopwords


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def normalize_key(text: str) -> str:
    return re.sub(r"\s+", " ", strip_accents(str(text or "").casefold())).strip()


def detect_language(text: str) -> str:
    normalized = normalize_key(text)
    english_markers = {"beach", "beaches", "hotel", "hotels", "where", "what", "tour", "tours", "museum"}
    spanish_markers = {"playa", "playas", "donde", "que", "museo", "museos", "excursion", "habana"}
    tokens = set(normalized.split())
    if tokens & english_markers and not tokens & spanish_markers:
        return "english"
    if tokens & spanish_markers and not tokens & english_markers:
        return "spanish"
    return "auto"


def tokenize_raw(text: str, *, language: str = "auto") -> list[str]:
    effective_language = detect_language(text) if language == "auto" else language
    stopwords = get_stopwords(effective_language)
    tokens: list[str] = []
    for match in WORD_RE.finditer(str(text or "")):
        token = normalize_key(match.group(0))
        if token and token not in stopwords:
            tokens.append(token)
    return tokens
