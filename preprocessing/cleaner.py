"""
Text cleaning utilities for the IR preprocessing stage.

Steps implemented (standard IR):
- lowercase
- strip HTML tags (if present)
- remove punctuation, numbers, special characters
- collapse multiple spaces
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]+")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class TextCleaner:
    """
    Normalizes raw text into a whitespace-separated, lowercase string of letters.
    """

    remove_diacritics: bool = True

    def clean(self, text: object) -> str:
        if text is None:
            return ""

        if not isinstance(text, str):
            text = str(text)

        # Decode HTML entities first, then remove tags.
        text = html.unescape(text)
        text = _HTML_TAG_RE.sub(" ", text)

        # Normalize common apostrophe variants and remove apostrophes.
        # This keeps contractions as a single token candidate: "don't" -> "dont".
        text = text.replace("\u2019", "'").replace("\u2018", "'")
        text = text.replace("'", "")

        if self.remove_diacritics:
            # Common IR normalization: remove accents/diacritics.
            # Example: "habitaci\u00f3n" -> "habitacion", "ni\u00f1o" -> "nino".
            text = unicodedata.normalize("NFKD", text)
            text = "".join(ch for ch in text if not unicodedata.combining(ch))

        text = text.lower()

        # Remove punctuation, numbers, and special characters in one pass.
        text = _NON_ALPHA_RE.sub(" ", text)

        # Remove extra whitespace.
        text = _WHITESPACE_RE.sub(" ", text).strip()
        return text


def clean_text(text: object, cleaner: Optional[TextCleaner] = None) -> str:
    """
    Convenience function for cleaning a single document.
    """

    if cleaner is None:
        cleaner = TextCleaner()
    return cleaner.clean(text)
