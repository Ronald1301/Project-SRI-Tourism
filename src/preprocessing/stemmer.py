"""
Stemming utilities for IR preprocessing.

Uses NLTK stemmers:
- English: PorterStemmer
- Spanish: SnowballStemmer("spanish")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from nltk.stem import PorterStemmer, SnowballStemmer


@dataclass
class Stemmer:
    language: str = "english"
    stemmer: Optional[object] = field(default=None)

    def __post_init__(self) -> None:
        if self.stemmer is not None:
            return

        lang = (self.language or "english").strip().lower()
        if lang in {"en", "eng", "english", "ingles", "ing"}:
            self.stemmer = PorterStemmer()
            return
        if lang in {"es", "spa", "spanish", "espanol", "espa\u00f1ol"}:
            self.stemmer = SnowballStemmer("spanish")
            return

        # Default behavior: English Porter stemmer.
        self.stemmer = PorterStemmer()

    def stem(self, token: str) -> str:
        if not token:
            return ""
        # `stemmer` is set in __post_init__.
        return self.stemmer.stem(token)  # type: ignore[union-attr]

    def stem_tokens(self, tokens: Iterable[str]) -> List[str]:
        return [self.stem(t) for t in tokens if t]


def stem_tokens(tokens: Iterable[str], stemmer: Optional[Stemmer] = None) -> List[str]:
    """
    Convenience function for stemming a token list.
    """

    if stemmer is None:
        stemmer = Stemmer()
    return stemmer.stem_tokens(tokens)
