"""
Tokenization and stopword removal for IR preprocessing.

Steps implemented:
- split cleaned text into tokens
- remove empty tokens
- filter very short tokens (< 2 characters)
- remove stopwords (English/Spanish)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import unicodedata
from typing import Iterable, List, Optional, Set

# Minimal but solid English stopword list as a fallback when NLTK data is missing.
# This keeps the project runnable in environments where `nltk.corpus.stopwords`
# isn't downloaded yet.
_FALLBACK_EN_STOPWORDS: Set[str] = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

_FALLBACK_ES_STOPWORDS: Set[str] = {
    "a",
    "al",
    "algo",
    "algunas",
    "algunos",
    "ante",
    "antes",
    "como",
    "con",
    "contra",
    "cual",
    "cuales",
    "cualquier",
    "cuando",
    "de",
    "del",
    "desde",
    "donde",
    "dos",
    "el",
    "ella",
    "ellas",
    "ellos",
    "en",
    "entre",
    "era",
    "erais",
    "eran",
    "eras",
    "eres",
    "es",
    "esa",
    "esas",
    "ese",
    "eses",
    "eso",
    "esos",
    "esta",
    "estaba",
    "estabais",
    "estaban",
    "estabas",
    "estad",
    "estada",
    "estadas",
    "estado",
    "estados",
    "estais",
    "estamos",
    "estan",
    "estando",
    "estar",
    "estaremos",
    "estara",
    "estaran",
    "estaras",
    "estare",
    "estareis",
    "estaria",
    "estariais",
    "estarian",
    "estarias",
    "estas",
    "este",
    "estemos",
    "esto",
    "estos",
    "estoy",
    "fue",
    "fuera",
    "fuerais",
    "fueran",
    "fueras",
    "fueron",
    "fui",
    "fuimos",
    "fuiste",
    "fuisteis",
    "ha",
    "habeis",
    "habia",
    "habiais",
    "habian",
    "habias",
    "habida",
    "habidas",
    "habido",
    "habidos",
    "habiendo",
    "habla",
    "hablan",
    "haber",
    "habia",
    "habian",
    "habias",
    "habla",
    "hace",
    "hacen",
    "hacia",
    "haciendo",
    "han",
    "has",
    "hasta",
    "hay",
    "la",
    "las",
    "le",
    "les",
    "lo",
    "los",
    "mas",
    "me",
    "mi",
    "mis",
    "mucho",
    "muy",
    "no",
    "nos",
    "nosotros",
    "nuestra",
    "nuestras",
    "nuestro",
    "nuestros",
    "o",
    "os",
    "otra",
    "otras",
    "otro",
    "otros",
    "para",
    "pero",
    "poco",
    "por",
    "porque",
    "que",
    "quien",
    "quienes",
    "se",
    "sea",
    "sean",
    "ser",
    "si",
    "sin",
    "sobre",
    "sois",
    "somos",
    "son",
    "soy",
    "su",
    "sus",
    "tambien",
    "te",
    "teneis",
    "tenemos",
    "tener",
    "tengo",
    "ti",
    "tiene",
    "tienen",
    "toda",
    "todas",
    "todo",
    "todos",
    "tu",
    "tus",
    "un",
    "una",
    "uno",
    "unos",
    "unas",
    "usted",
    "ustedes",
    "vosotros",
    "vuestra",
    "vuestras",
    "vuestro",
    "vuestros",
    "y",
    "ya",
}


def _strip_diacritics(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in value if not unicodedata.combining(ch))


def _normalize_language(language: Optional[str]) -> str:
    if not language:
        return "english"
    lang = str(language).strip().lower()
    if lang in {"en", "eng", "english", "ingles", "ing"}:
        return "english"
    if lang in {"es", "spa", "spanish", "espanol", "espa\u00f1ol"}:
        return "spanish"
    return lang


def load_english_stopwords() -> Set[str]:
    """
    Load a set of English stopwords.

    Tries NLTK first; if the corpus isn't available, falls back to a built-in list.
    """

    try:
        from nltk.corpus import stopwords  # type: ignore

        try:
            words = stopwords.words("english")
        except LookupError:
            # NLTK data not downloaded; fall back to the bundled list.
            return set(_FALLBACK_EN_STOPWORDS)

        normalized: Set[str] = set()
        for w in words:
            if not w or not isinstance(w, str):
                continue
            w = w.strip().lower()
            w = w.replace("\u2019", "'").replace("\u2018", "'").replace("'", "")
            w = _strip_diacritics(w)
            if w:
                normalized.add(w)
        return normalized
    except Exception:
        return set(_FALLBACK_EN_STOPWORDS)


def load_spanish_stopwords() -> Set[str]:
    """
    Load a set of Spanish stopwords.

    Tries NLTK first; if the corpus isn't available, falls back to a built-in list.
    Stopwords are normalized to match the cleaner (lowercase + no diacritics).
    """

    try:
        from nltk.corpus import stopwords  # type: ignore

        try:
            words = stopwords.words("spanish")
        except LookupError:
            return set(_FALLBACK_ES_STOPWORDS)

        normalized: Set[str] = set()
        for w in words:
            if not w or not isinstance(w, str):
                continue
            w = _strip_diacritics(w.strip().lower())
            if w:
                normalized.add(w)
        return normalized
    except Exception:
        return set(_FALLBACK_ES_STOPWORDS)


def load_stopwords(language: str = "english") -> Set[str]:
    lang = _normalize_language(language)
    if lang == "english":
        return load_english_stopwords()
    if lang == "spanish":
        return load_spanish_stopwords()
    # Unknown language: return empty set (no stopword filtering).
    return set()


@dataclass
class Tokenizer:
    min_token_length: int = 2
    language: str = "english"
    stopwords: Optional[Set[str]] = field(default=None)

    def __post_init__(self) -> None:
        if self.stopwords is None:
            self.stopwords = load_stopwords(self.language)

    def tokenize(self, text: str) -> List[str]:
        """
        Split cleaned text on whitespace and apply basic length filtering.
        """

        if not text:
            return []

        tokens = []
        for tok in text.split():
            if not tok:
                continue
            if len(tok) < self.min_token_length:
                continue
            tokens.append(tok)
        return tokens

    def remove_stopwords(self, tokens: Iterable[str], stopwords: Optional[Set[str]] = None) -> List[str]:
        """
        Remove stopwords from a token stream.
        """

        stop_set = self.stopwords if stopwords is None else stopwords
        if not stop_set:
            return [t for t in tokens if t]
        return [t for t in tokens if t and t not in stop_set]

    def tokenize_and_filter(self, text: str) -> List[str]:
        """
        Tokenize and remove stopwords in one call (convenience).
        """

        return self.remove_stopwords(self.tokenize(text))
