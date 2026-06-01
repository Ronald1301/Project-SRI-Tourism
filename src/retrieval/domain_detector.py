from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, Literal

import numpy as np
from scipy import sparse

from src.indexing.tfidf_index import TFIDFIndex
from src.preprocessing.pipeline import PreprocessingPipeline
from src.retrieval.lsi_model import LSIModel

DecisionLabel = Literal["IN_DOMAIN", "OUT_OF_DOMAIN", "UNCERTAIN"]

logger = logging.getLogger("src.retrieval.domain_detector")


@dataclass(frozen=True)
class DomainThresholds:
    """Thresholds used by the fast heuristic layer."""

    max_score_out: float = 0.03
    avg_score_out: float = 0.005
    lsi_similarity_out: float = 0.05
    max_score_in: float = 0.60
    lsi_similarity_in: float = 0.35

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "DomainThresholds":
        if not values:
            return cls()
        return cls(
            max_score_out=float(values.get("max_score_out", cls.max_score_out)),
            avg_score_out=float(values.get("avg_score_out", cls.avg_score_out)),
            lsi_similarity_out=float(values.get("lsi_similarity_out", cls.lsi_similarity_out)),
            max_score_in=float(values.get("max_score_in", cls.max_score_in)),
            lsi_similarity_in=float(values.get("lsi_similarity_in", cls.lsi_similarity_in)),
        )


class DomainDetector:
    """
    Hybrid domain detector for tourism-in-Cuba queries.

    It combines:
    - TF-IDF retrieval signals
    - LSI similarity against the corpus centroid
    - lexical overlap with domain keywords
    - optional Ollama fallback only for uncertain cases
    """

    def __init__(
        self,
        tfidf_index: TFIDFIndex,
        lsi_model: LSIModel,
        domain_keywords: Iterable[str] | None,
        llm_client: Any = None,
        *,
        language: str = "spanish",
        thresholds: Mapping[str, Any] | DomainThresholds | None = None,
        llm_model: str | None = None,
        debug: bool = False,
    ) -> None:
        self.tfidf_index = tfidf_index
        self.lsi_model = lsi_model
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.debug = bool(debug)
        self.thresholds = (
            thresholds
            if isinstance(thresholds, DomainThresholds)
            else DomainThresholds.from_mapping(thresholds)
        )
        self.pipeline = PreprocessingPipeline(language=language)

        if self.tfidf_index.matrix is None:
            raise ValueError("TF-IDF index must be built or loaded before using DomainDetector")
        if self.lsi_model.doc_vectors is None:
            raise ValueError("LSI model must be trained or loaded before using DomainDetector")

        self._doc_count = int(self.tfidf_index.matrix.shape[0])
        self._lsi_doc_count = int(self.lsi_model.doc_vectors.shape[0])
        self._corpus_centroid = self._build_lsi_centroid(self.lsi_model.doc_vectors)

        raw_keywords = tuple(
            keyword.strip()
            for keyword in (domain_keywords or ())
            if isinstance(keyword, str) and keyword.strip()
        )
        self.domain_keywords = raw_keywords
        self._domain_keyword_tokens = self._build_keyword_token_set(raw_keywords)
        self._non_domain_keyword_tokens = self._build_keyword_token_set(
            (
                "dolar",
                "dolares",
                "moneda",
                "cambio",
                "divisa",
                "bolsa",
                "inflacion",
                "salario",
                "noticias",
                "politica",
                "clima",
                "tiempo",
                "bitcoin",
                "economia",
                "precio",
            )
        )

        if not self.domain_keywords:
            logger.warning(
                "DomainDetector initialized without domain keywords; lexical overlap will stay disabled"
            )

        if self.debug:
            logger.setLevel(logging.DEBUG)

    def compute_features(self, query: str) -> dict[str, float | int]:
        query_tokens = self._tokenize_query(query)
        if not query_tokens:
            return {
                "max_score": 0.0,
                "avg_score": 0.0,
                "lsi_similarity": 0.0,
                "keyword_overlap": 0,
                "non_domain_hint": 0,
            }

        query_vector = self.tfidf_index.vectorize_query(query_tokens)
        tfidf_scores = self._compute_tfidf_scores(query_vector)
        max_score = float(np.max(tfidf_scores)) if tfidf_scores.size else 0.0
        non_zero_scores = tfidf_scores[tfidf_scores > 0.0]
        avg_score = float(np.mean(non_zero_scores)) if non_zero_scores.size else 0.0
        lsi_similarity = float(self._compute_lsi_similarity(query_vector))
        keyword_overlap = int(self._compute_keyword_overlap(query_tokens))
        non_domain_hint = int(self._compute_non_domain_hint(query_tokens))

        features: dict[str, float | int] = {
            "max_score": max_score,
            "avg_score": avg_score,
            "lsi_similarity": lsi_similarity,
            "keyword_overlap": keyword_overlap,
            "non_domain_hint": non_domain_hint,
        }

        if self.debug:
            logger.debug("Query features | query=%r | features=%s", query, features)

        return features

    def fast_decision(self, features: Mapping[str, float | int]) -> DecisionLabel:
        max_score = float(features.get("max_score", 0.0))
        avg_score = float(features.get("avg_score", 0.0))
        lsi_similarity = float(features.get("lsi_similarity", 0.0))
        keyword_overlap = int(features.get("keyword_overlap", 0))
        non_domain_hint = int(features.get("non_domain_hint", 0))

        if non_domain_hint > 0 and keyword_overlap == 0:
            return "OUT_OF_DOMAIN"

        if non_domain_hint > 0 and keyword_overlap > 0:
            return "UNCERTAIN"

        thresholds = self.thresholds
        if (
            max_score < thresholds.max_score_out
            and avg_score < thresholds.avg_score_out
            and lsi_similarity < thresholds.lsi_similarity_out
            and keyword_overlap == 0
        ):
            return "OUT_OF_DOMAIN"

        if (
            max_score > thresholds.max_score_in
            or lsi_similarity > thresholds.lsi_similarity_in
            or (keyword_overlap > 0 and non_domain_hint == 0)
        ):
            return "IN_DOMAIN"

        return "UNCERTAIN"

    def llm_decision(self, query: str) -> bool:
        if self.llm_client is None:
            raise ValueError("llm_client is required to call llm_decision")

        prompt = (
            "Eres un clasificador de consultas.\n"
            "Dominio: turismo en Cuba.\n"
            "Responde SOLO YES o NO.\n"
            f"Consulta: {query}"
        )
        response_text = self._invoke_llm(prompt)
        normalized = self._normalize_text(response_text)

        if normalized.startswith("yes"):
            return True
        if normalized.startswith("no"):
            return False

        match = re.search(r"\b(yes|no)\b", normalized, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower() == "yes"

        logger.warning("LLM returned an unexpected answer; falling back to NO | response=%r", response_text)
        return False

    def classify(self, query: str) -> DecisionLabel:
        features = self.compute_features(query)
        decision = self.fast_decision(features)
        if decision != "UNCERTAIN":
            return decision

        if self.llm_client is not None:
            try:
                return "IN_DOMAIN" if self.llm_decision(query) else "OUT_OF_DOMAIN"
            except Exception as exc:  # pragma: no cover - runtime fallback guard
                logger.warning("LLM fallback failed; using OUT_OF_DOMAIN | error=%s", exc)

        return "OUT_OF_DOMAIN"

    def explain(self, query: str) -> dict[str, Any]:
        features = self.compute_features(query)
        fast_decision = self.fast_decision(features)
        used_llm = False
        llm_result: bool | None = None

        if fast_decision == "UNCERTAIN" and self.llm_client is not None:
            used_llm = True
            try:
                llm_result = self.llm_decision(query)
                final_decision: DecisionLabel = "IN_DOMAIN" if llm_result else "OUT_OF_DOMAIN"
            except Exception as exc:  # pragma: no cover - runtime fallback guard
                logger.warning("LLM fallback failed during explain(); using OUT_OF_DOMAIN | error=%s", exc)
                final_decision = "OUT_OF_DOMAIN"
                llm_result = None
        else:
            final_decision = fast_decision if fast_decision != "UNCERTAIN" else "OUT_OF_DOMAIN"

        return {
            "query": query,
            "features": features,
            "fast_decision": fast_decision,
            "decision": final_decision,
            "used_llm": used_llm,
            "llm_result": llm_result,
            "thresholds": {
                "max_score_out": self.thresholds.max_score_out,
                "avg_score_out": self.thresholds.avg_score_out,
                "lsi_similarity_out": self.thresholds.lsi_similarity_out,
                "max_score_in": self.thresholds.max_score_in,
                "lsi_similarity_in": self.thresholds.lsi_similarity_in,
            },
        }

    def _tokenize_query(self, query: str) -> list[str]:
        if not query or not query.strip():
            return []
        return self.pipeline.process_text(query)

    def _compute_tfidf_scores(self, query_vector: sparse.csr_matrix) -> np.ndarray:
        if self.tfidf_index.matrix is None or self.tfidf_index.matrix.shape[0] == 0:
            return np.array([], dtype=np.float32)
        if query_vector is None or query_vector.shape[1] == 0:
            return np.zeros(self.tfidf_index.matrix.shape[0], dtype=np.float32)

        scores = self.tfidf_index.matrix @ query_vector.T
        if sparse.issparse(scores):
            scores = scores.toarray()
        return np.asarray(scores, dtype=np.float32).reshape(-1)

    def _compute_lsi_similarity(self, query_vector: sparse.csr_matrix) -> float:
        if self._corpus_centroid is None or self._corpus_centroid.size == 0:
            return 0.0
        if query_vector is None or query_vector.shape[1] == 0:
            return 0.0

        try:
            query_lsi = self.lsi_model.transform_query(query_vector)
        except Exception as exc:
            logger.debug("Unable to project query into LSI space: %s", exc)
            return 0.0
        return self._cosine_similarity(query_lsi, self._corpus_centroid)

    def _compute_keyword_overlap(self, query_tokens: Sequence[str]) -> int:
        if not self._domain_keyword_tokens:
            return 0

        query_token_set = {token for token in query_tokens if token}
        return int(len(query_token_set & self._domain_keyword_tokens))

    def _compute_non_domain_hint(self, query_tokens: Sequence[str]) -> int:
        if not self._non_domain_keyword_tokens:
            return 0

        query_token_set = {token for token in query_tokens if token}
        return int(len(query_token_set & self._non_domain_keyword_tokens))

    def _build_lsi_centroid(self, doc_vectors: np.ndarray | None) -> np.ndarray | None:
        if doc_vectors is None:
            return None
        vectors = np.asarray(doc_vectors, dtype=np.float32)
        if vectors.size == 0:
            return None
        centroid = np.mean(vectors, axis=0)
        centroid = np.asarray(centroid, dtype=np.float32).reshape(-1)
        return centroid if centroid.size else None

    def _build_keyword_token_set(self, keywords: Iterable[str]) -> frozenset[str]:
        tokens: set[str] = set()
        for keyword in keywords:
            tokens.update(self.pipeline.process_text(keyword))
        return frozenset(tokens)

    def _invoke_llm(self, prompt: str) -> str:
        client = self.llm_client

        if callable(client):
            result = client(prompt)
            return self._extract_text_from_response(result)

        model = self.llm_model or getattr(client, "model", None)

        if hasattr(client, "chat"):
            messages = [{"role": "user", "content": prompt}]
            try:
                if model is not None:
                    response = client.chat(model=model, messages=messages)
                else:
                    response = client.chat(messages=messages)
            except TypeError:
                response = client.chat(prompt)
            return self._extract_text_from_response(response)

        if hasattr(client, "generate"):
            try:
                if model is not None:
                    response = client.generate(model=model, prompt=prompt)
                else:
                    response = client.generate(prompt=prompt)
            except TypeError:
                response = client.generate(prompt)
            return self._extract_text_from_response(response)

        if isinstance(client, Mapping):
            response = client.get("response", "")
            return self._extract_text_from_response(response)

        raise TypeError("llm_client must be callable or expose chat/generate methods")

    def _extract_text_from_response(self, response: Any) -> str:
        if response is None:
            return ""
        if isinstance(response, str):
            return response
        if isinstance(response, Mapping):
            for key in ("response", "content", "text", "message"):
                value = response.get(key)
                if isinstance(value, str):
                    return value
                if isinstance(value, Mapping):
                    nested = value.get("content")
                    if isinstance(nested, str):
                        return nested
            return str(response)
        if isinstance(response, Sequence) and not isinstance(response, (bytes, bytearray, str)):
            parts = [self._extract_text_from_response(item) for item in response]
            return "\n".join(part for part in parts if part)
        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content
        message = getattr(response, "message", None)
        if message is not None:
            extracted = self._extract_text_from_response(message)
            if extracted:
                return extracted
        if hasattr(response, "response"):
            extracted = self._extract_text_from_response(getattr(response, "response"))
            if extracted:
                return extracted
        return str(response)

    def _cosine_similarity(self, vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        a = np.asarray(vector_a, dtype=np.float32).reshape(-1)
        b = np.asarray(vector_b, dtype=np.float32).reshape(-1)
        if a.size == 0 or b.size == 0:
            return 0.0

        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())
