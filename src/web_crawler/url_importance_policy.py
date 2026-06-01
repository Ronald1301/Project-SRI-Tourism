"""Politica heuristica para decidir si una URL merece ser descargada."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class URLImportanceDecision:
    """Resultado de evaluar la relevancia de una URL."""

    is_important: bool
    score: float
    reason: str


@dataclass(frozen=True)
class URLImportancePolicy:
    """Asigna un score heuristico a cada URL candidata."""

    min_score: float = 1.2

    _BLOCK_EXTENSIONS = (
        ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
        ".zip", ".rar", ".mp3", ".mp4", ".avi", ".mov", ".css", ".js",
        ".xml", ".json",
    )
    _BLOCK_PATTERNS = (
        "/login",
        "/register",
        "/signin",
        "/signup",
        "/account",
        "/cart",
        "/checkout",
        "/wp-admin",
        "/wp-includes",
        "/tag/",
        "/author/",
        "/feed",
        "/rss",
        "spotify.com",
        "open.spotify.com",
        "music.apple.com",
        "apple.com/music",
        "soundcloud.com",
        "deezer.com",
        "tidal.com",
        "youtube.com/watch",
        "youtube.com/playlist",
        "facebook.com",
        "twitter.com",
        "instagram.com",
        "youtube.com",
        "linkedin.com",
    )
    _TOPIC_HINTS = (
        "turismo",
        "travel",
        "destino",
        "destinos",
        "hotel",
        "playa",
        "havana",
        "habana",
        "cuba",
        "vacaciones",
        "visitar",
        "que-hacer",
    )

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokeniza un texto en palabras alfanumericas.

        Args:
            text: Texto a tokenizar.

        Returns:
            set[str]: Conjunto de tokens normalizados.
        """
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def evaluate(
        self,
        *,
        url: str,
        query: str,
        title: str = "",
        snippet: str = "",
    ) -> URLImportanceDecision:
        """Evalua si la URL es relevante para la consulta.

        Args:
            url: URL candidata.
            query: Consulta del usuario.
            title: Titulo asociado a la URL.
            snippet: Fragmento o descripcion corta.

        Returns:
            URLImportanceDecision: Decision binaria con score y razon.
        """
        parsed = urlparse(url.strip())
        if parsed.scheme not in {"http", "https"}:
            return URLImportanceDecision(False, 0.0, "scheme_no_permitido")

        normalized_url = url.lower()
        if any(normalized_url.endswith(ext) for ext in self._BLOCK_EXTENSIONS):
            return URLImportanceDecision(False, 0.0, "extension_bloqueada")
        if any(pattern in normalized_url for pattern in self._BLOCK_PATTERNS):
            return URLImportanceDecision(False, 0.0, "patron_bloqueado")

        score = 0.0
        host = (parsed.netloc or "").lower()
        path = (parsed.path or "").lower()
        title_l = (title or "").lower()
        snippet_l = (snippet or "").lower()

        # Coincidencia semántica con la consulta.
        query_tokens = self._tokenize(query)
        if query_tokens:
            haystack_tokens = self._tokenize(" ".join([path, title_l, snippet_l]))
            overlap = len(query_tokens & haystack_tokens)
            score += overlap * 0.7

        # Señales temáticas de turismo.
        for hint in self._TOPIC_HINTS:
            if hint in path:
                score += 0.35
            if hint in title_l:
                score += 0.25
            if hint in snippet_l:
                score += 0.15

        # Señales de autoridad de dominio.
        if host.endswith(".gob.cu") or host.endswith(".gov"):
            score += 0.6
        if host.endswith(".travel") or host.endswith(".org"):
            score += 0.25

        # Penalización por URLs muy profundas (ruido típico).
        depth = len([part for part in path.split("/") if part.strip()])
        if depth >= 6:
            score -= 0.4

        important = score >= self.min_score
        return URLImportanceDecision(
            is_important=important,
            score=round(score, 4),
            reason="score_ok" if important else "score_bajo",
        )
