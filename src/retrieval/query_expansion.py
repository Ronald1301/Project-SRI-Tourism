from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


DEFAULT_FEEDBACK_PATH = Path("data/feedback/query_feedback.json")
WORD_RE = re.compile(r"[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ]{3,}")

STOPWORDS = {
    "aqui",
    "algo",
    "ante",
    "como",
    "con",
    "cual",
    "cuando",
    "donde",
    "desde",
    "este",
    "esta",
    "estos",
    "estas",
    "para",
    "pero",
    "porque",
    "que",
    "sin",
    "sobre",
    "todo",
    "una",
    "uno",
    "unos",
    "unas",
    "ver",
    "visitar",
    "viaje",
    "viajar",
    "turismo",
    "turistico",
    "turistica",
    "cuba",
    "cubano",
    "cubana",
    "lugar",
    "lugares",
    "mejor",
    "mejores",
}

DOMAIN_SYNONYMS = {
    "playa": ["cayo", "balneario", "costa", "mar"],
    "playas": ["cayo", "balneario", "costa", "mar"],
    "hotel": ["alojamiento", "hostal", "reserva"],
    "hoteles": ["alojamiento", "hostal", "reserva"],
    "restaurante": ["gastronomia", "comida", "cocina"],
    "restaurantes": ["gastronomia", "comida", "cocina"],
    "habana": ["malecon", "vedado", "capital"],
    "vieja": ["casco historico", "centro historico"],
    "museo": ["galeria", "patrimonio", "cultura"],
    "museos": ["galeria", "patrimonio", "cultura"],
    "excursion": ["tour", "ruta", "recorrido"],
    "excursiones": ["tour", "ruta", "recorrido"],
    "naturaleza": ["parque", "sendero", "reserva"],
    "familia": ["ninos", "familiar", "actividades"],
}

IMPLICIT_EVENT_GROUPS = {
    "copy_url": "source_interaction",
    "copiar_url": "source_interaction",
    "open_source": "source_interaction",
    "abrir_fuente": "source_interaction",
    "source_interaction": "source_interaction",
}

IMPLICIT_WEIGHTS = {
    "source_interaction": 0.65,
    "view_result": 0.2,
    "dwell": 0.35,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def _normalize_key(text: str) -> str:
    return re.sub(r"\s+", " ", _strip_accents(str(text or "").casefold())).strip()


def _tokenize_raw(text: str) -> list[str]:
    tokens: list[str] = []
    for match in WORD_RE.finditer(str(text or "")):
        token = _normalize_key(match.group(0))
        if token and token not in STOPWORDS:
            tokens.append(token)
    return tokens


@dataclass
class ExpansionResult:
    original_query: str
    expanded_query: str
    terms: list[str]
    term_scores: dict[str, float]
    method: str = "hybrid"
    applied: bool = False
    selected_query: str | None = None
    selected_strategy: str = "raw"

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "original_query": self.original_query,
            "expanded_query": self.expanded_query,
            "selected_query": self.selected_query or self.expanded_query,
            "selected_strategy": self.selected_strategy,
            "terms": self.terms,
            "term_scores": self.term_scores,
            "method": self.method,
            "applied": self.applied,
        }


@dataclass
class FeedbackStore:
    path: Path = DEFAULT_FEEDBACK_PATH
    data: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.data = self._load()

    def _load(self) -> dict[str, list[dict[str, Any]]]:
        if not self.path.exists():
            return {"explicit": [], "implicit": []}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"explicit": [], "implicit": []}
        return {
            "explicit": list(payload.get("explicit") or []),
            "implicit": list(payload.get("implicit") or []),
        }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)

    def add_explicit(
        self,
        *,
        query: str,
        doc_id: str,
        relevance: int,
        expanded_query: str | None = None,
        search_mode: str | None = None,
    ) -> dict[str, Any]:
        event = {
            "query": query,
            "query_key": _normalize_key(query),
            "expanded_query": expanded_query,
            "doc_id": doc_id,
            "relevance": int(relevance),
            "search_mode": search_mode,
            "timestamp": _now_iso(),
        }
        self.data.setdefault("explicit", []).append(event)
        self._save()
        return event

    def add_implicit(
        self,
        *,
        query: str,
        doc_id: str,
        event: str,
        search_mode: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        event_group = IMPLICIT_EVENT_GROUPS.get(_normalize_key(event), _normalize_key(event))
        query_key = _normalize_key(query)
        doc_key = str(doc_id or "").strip()

        for item in self.data.get("implicit", []):
            if (
                item.get("query_key") == query_key
                and item.get("doc_id") == doc_key
                and item.get("event_group") == event_group
            ):
                return item, False

        payload = {
            "query": query,
            "query_key": query_key,
            "doc_id": doc_key,
            "event": event,
            "event_group": event_group,
            "weight": float(IMPLICIT_WEIGHTS.get(event_group, 0.2)),
            "search_mode": search_mode,
            "timestamp": _now_iso(),
        }
        self.data.setdefault("implicit", []).append(payload)
        self._save()
        return payload, True

    def query_feedback(self, query: str) -> dict[str, list[dict[str, Any]]]:
        query_key = _normalize_key(query)
        return {
            "explicit": [
                item
                for item in self.data.get("explicit", [])
                if item.get("query_key") == query_key
            ],
            "implicit": [
                item
                for item in self.data.get("implicit", [])
                if item.get("query_key") == query_key
            ],
        }


class QueryExpander:
    def __init__(
        self,
        searcher: Any,
        *,
        feedback_path: Path | str = DEFAULT_FEEDBACK_PATH,
        max_terms: int = 5,
    ) -> None:
        self.searcher = searcher
        self.feedback_store = FeedbackStore(Path(feedback_path))
        self.max_terms = max(int(max_terms), 1)

    def expand_query(
        self,
        query: str,
        *,
        method: str = "hybrid",
        top_documents: list[Any] | None = None,
        max_terms: int | None = None,
    ) -> ExpansionResult:
        query = str(query or "").strip()
        if not query:
            return ExpansionResult(query, query, [], {}, method=method, applied=False)

        candidates: dict[str, float] = {}
        original_terms = set(_tokenize_raw(query))
        feedback = self.feedback_store.query_feedback(query)
        top_docs = [self._document_payload(doc) for doc in (top_documents or [])[:3]]

        self._add_synonym_terms(query, candidates)
        self._add_pseudo_relevance_terms(top_docs, candidates)
        self._add_cooccurrence_terms(query, top_docs, candidates)
        self._add_feedback_terms(feedback, candidates)
        self._add_rocchio_terms(query, top_docs, feedback, candidates)

        ranked_terms = self._rank_terms(candidates, original_terms, max_terms or self.max_terms)
        expanded_query = " ".join([query] + ranked_terms).strip() if ranked_terms else query
        return ExpansionResult(
            original_query=query,
            expanded_query=expanded_query,
            terms=ranked_terms,
            term_scores={term: round(float(candidates.get(term, 0.0)), 4) for term in ranked_terms},
            method=method,
            applied=bool(ranked_terms),
            selected_query=expanded_query,
            selected_strategy="expanded" if ranked_terms else "raw",
        )

    def record_explicit_feedback(
        self,
        *,
        query: str,
        doc_id: str,
        relevance: int,
        expanded_query: str | None = None,
        search_mode: str | None = None,
    ) -> dict[str, Any]:
        return self.feedback_store.add_explicit(
            query=query,
            doc_id=doc_id,
            relevance=relevance,
            expanded_query=expanded_query,
            search_mode=search_mode,
        )

    def record_implicit_feedback(
        self,
        *,
        query: str,
        doc_id: str,
        event: str,
        search_mode: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        return self.feedback_store.add_implicit(
            query=query,
            doc_id=doc_id,
            event=event,
            search_mode=search_mode,
        )

    def _add_candidate(self, candidates: dict[str, float], term: str, score: float) -> None:
        term = _normalize_key(term)
        if not term or term in STOPWORDS or len(term) < 3:
            return
        candidates[term] = candidates.get(term, 0.0) + float(score)

    def _add_synonym_terms(self, query: str, candidates: dict[str, float]) -> None:
        for token in _tokenize_raw(query):
            for synonym in DOMAIN_SYNONYMS.get(token, []):
                self._add_candidate(candidates, synonym, 0.85)

    def _add_pseudo_relevance_terms(self, docs: list[dict[str, Any]], candidates: dict[str, float]) -> None:
        for rank, doc in enumerate(docs, start=1):
            rank_weight = 1.0 / rank
            title_tokens = _tokenize_raw(str(doc.get("title") or ""))
            body_tokens = _tokenize_raw(
                " ".join(
                    [
                        str(doc.get("summary") or ""),
                        str(doc.get("content_text") or doc.get("content") or ""),
                    ]
                )
            )
            for token in title_tokens[:24]:
                self._add_candidate(candidates, token, 0.7 * rank_weight)
            for token in self._top_frequency_terms(body_tokens, limit=12):
                self._add_candidate(candidates, token, 0.35 * rank_weight)

    def _add_cooccurrence_terms(
        self,
        query: str,
        docs: list[dict[str, Any]],
        candidates: dict[str, float],
    ) -> None:
        query_terms = set(_tokenize_raw(query))
        if not query_terms:
            return
        for rank, doc in enumerate(docs, start=1):
            tokens = _tokenize_raw(
                " ".join(
                    [
                        str(doc.get("title") or ""),
                        str(doc.get("summary") or ""),
                        str(doc.get("content_text") or doc.get("content") or ""),
                    ]
                )
            )
            rank_weight = 1.0 / rank
            for idx, token in enumerate(tokens):
                if token not in query_terms:
                    continue
                start = max(idx - 4, 0)
                end = min(idx + 5, len(tokens))
                for neighbor in tokens[start:end]:
                    if neighbor != token:
                        self._add_candidate(candidates, neighbor, 0.28 * rank_weight)

    def _add_feedback_terms(
        self,
        feedback: dict[str, list[dict[str, Any]]],
        candidates: dict[str, float],
    ) -> None:
        for item in feedback.get("explicit", []):
            doc_id = str(item.get("doc_id") or "")
            relevance = int(item.get("relevance") or 0)
            doc = self._document_by_id(doc_id)
            if not doc:
                continue
            score = 0.95 if relevance > 0 else -0.75
            for token in self._top_terms_for_doc(doc, limit=10):
                self._add_candidate(candidates, token, score)

        for item in feedback.get("implicit", []):
            doc_id = str(item.get("doc_id") or "")
            doc = self._document_by_id(doc_id)
            if not doc:
                continue
            score = float(item.get("weight") or 0.2)
            for token in self._top_terms_for_doc(doc, limit=6):
                self._add_candidate(candidates, token, score)

    def _add_rocchio_terms(
        self,
        query: str,
        top_docs: list[dict[str, Any]],
        feedback: dict[str, list[dict[str, Any]]],
        candidates: dict[str, float],
    ) -> None:
        matrix = getattr(getattr(self.searcher, "tfidf_index", None), "matrix", None)
        vocabulary = getattr(getattr(self.searcher, "tfidf_index", None), "vocabulary", None)
        if matrix is None or not vocabulary:
            return

        try:
            query_tokens = self.searcher.pipeline.process_text(query)
            raw_query_vector = self.searcher.tfidf_index.vectorize_query(query_tokens)
            query_vector = (
                raw_query_vector.toarray().ravel().astype(np.float32, copy=False)
                if sparse.issparse(raw_query_vector)
                else np.asarray(raw_query_vector, dtype=np.float32).reshape(-1)
            )
        except Exception:
            return

        positive_ids = [
            str(item.get("doc_id"))
            for item in feedback.get("explicit", [])
            if int(item.get("relevance") or 0) > 0
        ]
        negative_ids = [
            str(item.get("doc_id"))
            for item in feedback.get("explicit", [])
            if int(item.get("relevance") or 0) <= 0
        ]
        if not positive_ids:
            positive_ids = [str(doc.get("doc_id")) for doc in top_docs if doc.get("doc_id")]

        pos_vectors = self._vectors_for_doc_ids(positive_ids)
        neg_vectors = self._vectors_for_doc_ids(negative_ids)
        if not pos_vectors:
            return

        alpha = 1.0
        beta = 0.75 if feedback.get("explicit") else 0.25
        gamma = 0.15
        rocchio_vector = alpha * query_vector
        rocchio_vector = rocchio_vector + beta * np.mean(pos_vectors, axis=0)
        if neg_vectors:
            rocchio_vector = rocchio_vector - gamma * np.mean(neg_vectors, axis=0)

        inverse_vocab = {idx: term for term, idx in vocabulary.items()}
        stem_to_raw = self._stem_to_raw_map(top_docs)
        for idx in np.argsort(rocchio_vector)[::-1][:18]:
            value = float(rocchio_vector[idx])
            if value <= 0:
                continue
            stem = inverse_vocab.get(int(idx))
            if not stem:
                continue
            raw_term = stem_to_raw.get(stem, stem)
            self._add_candidate(candidates, raw_term, min(value, 1.0) * 0.8)

    def _rank_terms(
        self,
        candidates: dict[str, float],
        original_terms: set[str],
        limit: int,
    ) -> list[str]:
        ranked = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))
        selected: list[str] = []
        seen: set[str] = set()
        for term, score in ranked:
            key = _normalize_key(term)
            if score <= 0 or key in seen or key in original_terms or key in STOPWORDS:
                continue
            if any(key in other or other in key for other in seen):
                continue
            selected.append(term)
            seen.add(key)
            if len(selected) >= limit:
                break
        return selected

    def _top_frequency_terms(self, tokens: list[str], *, limit: int) -> list[str]:
        counts: dict[str, int] = {}
        for token in tokens:
            if token in STOPWORDS:
                continue
            counts[token] = counts.get(token, 0) + 1
        return [
            term
            for term, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
        ]

    def _top_terms_for_doc(self, doc: dict[str, Any], *, limit: int) -> list[str]:
        tokens = _tokenize_raw(
            " ".join(
                [
                    str(doc.get("title") or ""),
                    str(doc.get("summary") or ""),
                    str(doc.get("content_text") or doc.get("content") or ""),
                ]
            )
        )
        return self._top_frequency_terms(tokens, limit=limit)

    def _vectors_for_doc_ids(self, doc_ids: list[str]) -> list[np.ndarray]:
        index = getattr(self.searcher, "tfidf_index", None)
        matrix = getattr(index, "matrix", None)
        doc_id_to_index = getattr(index, "doc_id_to_index", {}) or {}
        vectors: list[np.ndarray] = []
        if matrix is None:
            return vectors
        for doc_id in doc_ids:
            row_idx = doc_id_to_index.get(doc_id)
            if row_idx is None:
                continue
            row = matrix[row_idx]
            vector = row.toarray().ravel() if sparse.issparse(row) else np.asarray(row, dtype=float).reshape(-1)
            vectors.append(vector.astype(np.float32, copy=False))
        return vectors

    def _stem_to_raw_map(self, docs: list[dict[str, Any]]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for doc in docs:
            raw_text = " ".join(
                [
                    str(doc.get("title") or ""),
                    str(doc.get("summary") or ""),
                    str(doc.get("content_text") or doc.get("content") or ""),
                ]
            )
            for raw_term in _tokenize_raw(raw_text):
                try:
                    stems = self.searcher.pipeline.process_text(raw_term)
                except Exception:
                    stems = []
                for stem in stems:
                    mapping.setdefault(stem, raw_term)
        return mapping

    def _document_by_id(self, doc_id: str) -> dict[str, Any]:
        source = getattr(self.searcher, "documents_by_id", {}) or {}
        return dict(source.get(doc_id) or {})

    def _document_payload(self, doc: Any) -> dict[str, Any]:
        if isinstance(doc, dict):
            payload = dict(doc)
        else:
            payload = {
                "doc_id": getattr(doc, "doc_id", ""),
                "title": getattr(doc, "title", ""),
                "summary": getattr(doc, "summary", ""),
                "content_text": getattr(doc, "content_text", ""),
                "url": getattr(doc, "url", ""),
                "score": getattr(doc, "score", 0.0),
            }
            metadata = getattr(doc, "metadata", None)
            if isinstance(metadata, dict):
                payload.update({key: value for key, value in metadata.items() if key not in payload})

        doc_id = str(payload.get("doc_id") or "").strip()
        if doc_id:
            stored = self._document_by_id(doc_id)
            for key, value in stored.items():
                payload.setdefault(key, value)
        return payload
