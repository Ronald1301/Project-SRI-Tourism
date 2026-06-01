from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from scipy import sparse

from src.indexing.tfidf_index import TFIDFIndex
from src.preprocessing.pipeline import PreprocessingPipeline
from src.retrieval.lsi_model import LSIModel

DomainDecision = Literal["IN_DOMAIN", "OUT_OF_DOMAIN", "UNCERTAIN"]

logger = logging.getLogger("src.retrieval.domain_detector")


DEFAULT_DOMAIN_KEYWORDS = {
    "alojamiento",
    "baracoa",
    "beach",
    "beaches",
    "cayo",
    "cayos",
    "cienfuegos",
    "cuba",
    "cultura",
    "destino",
    "destination",
    "excursion",
    "gastronomia",
    "guardalavaca",
    "habana",
    "havana",
    "heritage",
    "historia",
    "hostal",
    "hotel",
    "hoteles",
    "hotels",
    "malecon",
    "matanzas",
    "museo",
    "museos",
    "museum",
    "naturaleza",
    "nature",
    "patrimonio",
    "playa",
    "playas",
    "reserva",
    "restaurant",
    "restaurants",
    "restaurante",
    "restaurantes",
    "ruta",
    "santiago",
    "senderismo",
    "tour",
    "tourism",
    "tours",
    "travel",
    "transporte",
    "transport",
    "trinidad",
    "turismo",
    "varadero",
    "vedado",
    "vinales",
    "viaje",
}

TOURISM_INTENT_KEYWORDS = {
    "alojamiento",
    "beach",
    "beaches",
    "cayo",
    "cayos",
    "cultura",
    "destino",
    "destination",
    "excursion",
    "gastronomia",
    "heritage",
    "historia",
    "hostal",
    "hotel",
    "hoteles",
    "hotels",
    "malecon",
    "museo",
    "museos",
    "museum",
    "naturaleza",
    "nature",
    "patrimonio",
    "playa",
    "playas",
    "reserva",
    "restaurant",
    "restaurants",
    "restaurante",
    "restaurantes",
    "ruta",
    "senderismo",
    "tour",
    "tourism",
    "tours",
    "travel",
    "transporte",
    "transport",
    "turismo",
    "vedado",
    "viaje",
}

OUT_OF_DOMAIN_KEYWORDS = {
    "bolsa",
    "cambio",
    "champions",
    "clima",
    "criptomoneda",
    "criptomonedas",
    "dolar",
    "dolares",
    "dollar",
    "euro",
    "futbol",
    "gano",
    "juego",
    "llueve",
    "lluvia",
    "manana",
    "mlc",
    "moneda",
    "precio",
    "pronostico",
    "sol",
    "tasa",
    "temperature",
    "tiempo",
    "weather",
}


@dataclass(frozen=True)
class DomainThresholds:
    out_max_score: float = 0.15
    out_avg_score: float = 0.005
    out_lsi_similarity: float = 0.10
    in_max_score: float = 0.18
    in_avg_score: float = 0.015
    in_lsi_similarity: float = 0.22


class LLMClient(Protocol):
    def classify_domain(self, prompt: str) -> bool:
        """Return True for in-domain and False for out-of-domain."""


class OllamaDomainClient:
    def __init__(
        self,
        *,
        model: str | None = None,
        ollama_cmd: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        self.model = (model or os.getenv("DOMAIN_LLM_MODEL") or os.getenv("RAG_LLM_MODEL") or "phi3").strip()
        self.ollama_cmd = (ollama_cmd or os.getenv("DOMAIN_OLLAMA_CMD") or os.getenv("RAG_OLLAMA_CMD") or "ollama").strip()
        self.timeout_seconds = int(timeout_seconds or os.getenv("DOMAIN_LLM_TIMEOUT_SECONDS") or 30)

    def classify_domain(self, prompt: str) -> bool:
        if shutil.which(self.ollama_cmd) is None:
            raise RuntimeError(f"No se encontro el comando '{self.ollama_cmd}' en PATH.")

        result = subprocess.run(
            [self.ollama_cmd, "run", self.model],
            input=self._sanitize_for_subprocess(prompt),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=max(self.timeout_seconds, 5),
            check=False,
        )
        if result.returncode != 0:
            detail = " ".join((result.stderr or result.stdout or "sin detalle").split())
            raise RuntimeError(f"Ollama retorno {result.returncode}: {detail[:180]}")

        answer = " ".join((result.stdout or "").strip().upper().split())
        if answer.startswith("YES"):
            return True
        if answer.startswith("NO"):
            return False
        raise ValueError(f"Respuesta LLM invalida para clasificacion de dominio: {answer!r}")

    def _sanitize_for_subprocess(self, text: str) -> str:
        cleaned_chars: list[str] = []
        for char in str(text or ""):
            code = ord(char)
            is_private_use = 0xE000 <= code <= 0xF8FF
            is_unsafe_control = code < 32 and char not in "\n\r\t"
            cleaned_chars.append(" " if is_private_use or is_unsafe_control else char)
        return "".join(cleaned_chars)


class DomainDetector:
    def __init__(
        self,
        tfidf_index: TFIDFIndex,
        lsi_model: LSIModel,
        domain_keywords: set[str] | list[str] | tuple[str, ...] | None,
        llm_client: LLMClient | None = None,
        *,
        thresholds: DomainThresholds | None = None,
        language: str = "spanish",
    ) -> None:
        self.tfidf_index = tfidf_index
        self.lsi_model = lsi_model
        self.domain_keywords = set(domain_keywords or DEFAULT_DOMAIN_KEYWORDS)
        self.llm_client = llm_client
        self.thresholds = thresholds or DomainThresholds()
        self.preprocessing = PreprocessingPipeline(language=language)
        self.domain_keyword_tokens = self._normalize_keywords(self.domain_keywords)
        self.tourism_intent_tokens = self._normalize_keywords(TOURISM_INTENT_KEYWORDS)
        self.out_of_domain_tokens = self._normalize_keywords(OUT_OF_DOMAIN_KEYWORDS)
        self.domain_vocabulary = self._extract_domain_vocabulary()
        self._lsi_centroid = self._compute_lsi_centroid()

    @classmethod
    def from_searcher(
        cls,
        searcher: Any,
        *,
        domain_keywords: set[str] | list[str] | tuple[str, ...] | None = None,
        llm_client: LLMClient | None = None,
        thresholds: DomainThresholds | None = None,
    ) -> "DomainDetector":
        return cls(
            tfidf_index=searcher.tfidf_index,
            lsi_model=searcher.lsi_model,
            domain_keywords=domain_keywords or DEFAULT_DOMAIN_KEYWORDS,
            llm_client=llm_client,
            thresholds=thresholds,
            language=getattr(searcher, "language", "spanish"),
        )

    def compute_features(self, query: str) -> dict[str, float | int]:
        tokens = self.preprocessing.process_text(query or "")
        query_vector = self.tfidf_index.vectorize_query(tokens)
        tfidf_scores = self._tfidf_scores(query_vector)

        features = {
            "max_score": float(np.max(tfidf_scores)) if tfidf_scores.size else 0.0,
            "avg_score": float(np.mean(tfidf_scores)) if tfidf_scores.size else 0.0,
            "lsi_similarity": self._lsi_centroid_similarity(query_vector),
            "keyword_overlap": self._overlap(tokens, self.domain_keyword_tokens | self.domain_vocabulary),
            "tourism_intent_overlap": self._overlap(tokens, self.tourism_intent_tokens),
            "out_of_domain_overlap": self._overlap(tokens, self.out_of_domain_tokens),
        }
        logger.debug("Domain features | query=%r | features=%s", query, features)
        return features

    def fast_decision(self, features: dict[str, float | int]) -> DomainDecision:
        max_score = float(features.get("max_score", 0.0))
        avg_score = float(features.get("avg_score", 0.0))
        lsi_similarity = float(features.get("lsi_similarity", 0.0))
        keyword_overlap = int(features.get("keyword_overlap", 0))
        tourism_intent_overlap = int(features.get("tourism_intent_overlap", 0))
        out_of_domain_overlap = int(features.get("out_of_domain_overlap", 0))
        t = self.thresholds

        if out_of_domain_overlap > 0 and tourism_intent_overlap == 0:
            return "OUT_OF_DOMAIN"
        if keyword_overlap > 0:
            return "IN_DOMAIN"
        if max_score < t.out_max_score and avg_score < t.out_avg_score and lsi_similarity < t.out_lsi_similarity:
            return "OUT_OF_DOMAIN"
        if max_score > t.in_max_score and avg_score > t.in_avg_score and lsi_similarity > t.in_lsi_similarity:
            return "IN_DOMAIN"
        return "UNCERTAIN"

    def llm_decision(self, query: str) -> bool:
        if self.llm_client is None:
            raise RuntimeError("No hay llm_client configurado para fallback de dominio.")
        return self.llm_client.classify_domain(self._build_llm_prompt(query))

    def classify(self, query: str) -> DomainDecision:
        return self.explain(query)["decision"]  # type: ignore[return-value]

    def explain(self, query: str) -> dict[str, Any]:
        features = self.compute_features(query)
        decision = self.fast_decision(features)
        used_llm = False
        llm_result: bool | None = None
        llm_error: str | None = None

        if decision == "UNCERTAIN":
            if self.llm_client is not None:
                used_llm = True
                try:
                    llm_result = self.llm_decision(query)
                    decision = "IN_DOMAIN" if llm_result else "OUT_OF_DOMAIN"
                except Exception as exc:
                    llm_error = str(exc)
                    logger.warning("Domain LLM fallback failed | query=%r | error=%s", query, exc)
                    decision = "OUT_OF_DOMAIN"
            else:
                decision = "OUT_OF_DOMAIN"

        return {
            "query": query,
            "features": features,
            "decision": decision,
            "used_llm": used_llm,
            "llm_result": llm_result,
            "llm_error": llm_error,
            "thresholds": {
                "out_max_score": self.thresholds.out_max_score,
                "out_avg_score": self.thresholds.out_avg_score,
                "out_lsi_similarity": self.thresholds.out_lsi_similarity,
                "in_max_score": self.thresholds.in_max_score,
                "in_avg_score": self.thresholds.in_avg_score,
                "in_lsi_similarity": self.thresholds.in_lsi_similarity,
            },
        }

    def _tfidf_scores(self, query_vector: Any) -> np.ndarray:
        matrix = self.tfidf_index.matrix
        if matrix is None or getattr(matrix, "shape", (0, 0))[0] == 0:
            return np.array([], dtype=np.float32)
        if sparse.issparse(matrix):
            scores = query_vector @ matrix.T
            return np.asarray(scores.toarray()).ravel().astype(np.float32, copy=False)
        return np.asarray(query_vector).reshape(1, -1) @ np.asarray(matrix).T

    def _lsi_centroid_similarity(self, query_vector: Any) -> float:
        if self._lsi_centroid is None:
            return 0.0
        try:
            query_lsi = np.asarray(self.lsi_model.transform_query(query_vector), dtype=np.float32).reshape(-1)
        except Exception as exc:
            logger.debug("Could not transform query to LSI for domain detection: %s", exc)
            return 0.0
        return self._cosine(query_lsi, self._lsi_centroid)

    def _normalize_keywords(self, keywords: set[str] | list[str] | tuple[str, ...]) -> set[str]:
        normalized: set[str] = set()
        for keyword in keywords:
            normalized.update(self.preprocessing.process_text(keyword))
        return normalized

    def _extract_domain_vocabulary(self) -> set[str]:
        vocabulary = getattr(self.tfidf_index, "vocabulary", {}) or {}
        if not vocabulary:
            return set()
        return {token for token in self.domain_keyword_tokens if token in vocabulary}

    def _compute_lsi_centroid(self) -> np.ndarray | None:
        doc_vectors = getattr(self.lsi_model, "doc_vectors", None)
        if doc_vectors is None or len(doc_vectors) == 0:
            return None
        centroid = np.mean(np.asarray(doc_vectors, dtype=np.float32), axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            return None
        return centroid / norm

    def _build_llm_prompt(self, query: str) -> str:
        return "\n".join(
            [
                "Eres un clasificador de consultas.",
                "Dominio: turismo en Cuba.",
                "Responde SOLO YES o NO.",
                f"Consulta: {query}",
            ]
        )

    def _overlap(self, query_tokens: list[str], target_tokens: set[str]) -> int:
        return len(set(query_tokens) & target_tokens)

    def _cosine(self, left: np.ndarray, right: np.ndarray) -> float:
        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return float(np.dot(left, right) / (left_norm * right_norm))
