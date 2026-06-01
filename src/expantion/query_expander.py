from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from src.expantion.cache import CacheManager
from src.expantion.config import ConfigLoader, ExpansionConfig
from src.expantion.constants import DEFAULT_FEEDBACK_PATH, get_stopwords
from src.expantion.feedback_database import FeedbackDatabase
from src.expantion.logger import ExpansionLogger
from src.expantion.models import ExpansionResult
from src.expantion.ngrams import NGramExtractor
from src.expantion.synonyms import SynonymLoader
from src.expantion.text import detect_language, normalize_key, tokenize_raw


class QueryExpander:
    def __init__(
        self,
        searcher: Any,
        *,
        config_path: Path | str | None = None,
        synonyms_path: Path | str | None = None,
        feedback_db_path: Path | str | None = None,
        feedback_path: Path | str = DEFAULT_FEEDBACK_PATH,
        max_terms: int | None = None,
    ) -> None:
        del feedback_path
        self.searcher = searcher
        self.config: ExpansionConfig = ConfigLoader().load(config_path)
        self.max_terms = max(int(max_terms or self.config.max_terms), 1)
        self.acceptance_threshold = self.config.acceptance_threshold
        self.language = self.config.language
        self.stopwords = get_stopwords(self.language)
        self.synonyms = SynonymLoader(self.config).load(synonyms_path)
        self.feedback_store = FeedbackDatabase(feedback_db_path or self.config.feedback_db_path, self.config)
        self.logger = ExpansionLogger(self.config)
        self.cache = CacheManager(
            max_size=int(self.config.get("global.cache_max_size", 256)),
            ttl_seconds=int(self.config.get("global.cache_ttl_seconds", 600)),
            enabled=bool(self.config.get("global.cache_enabled", True)),
        )
        self.ngrams = NGramExtractor(
            enabled=bool(self.config.get("ngrams.enabled", True)),
            min_n=int(self.config.get("ngrams.min_n", 1)),
            max_n=int(self.config.get("ngrams.max_n", 3)),
            multipliers=dict(self.config.get("ngrams.multipliers", {"1": 1.0, "2": 1.2, "3": 1.5})),
        )

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

        top_docs = [self._document_payload(doc) for doc in (top_documents or [])[: self._top_documents_limit()]]
        cache_key = self._cache_key(query, method, top_docs, max_terms)
        cached = self.cache.get(cache_key)
        if isinstance(cached, ExpansionResult):
            self.logger.cache("hit", cache_key)
            return cached
        self.logger.cache("miss", cache_key)

        language = detect_language(query) if self.language == "auto" else self.language
        original_terms = set(tokenize_raw(query, language=language))
        feedback = self.feedback_store.query_feedback(query)
        candidates: dict[str, float] = {}
        trace: dict[str, Any] = {
            "language": language,
            "techniques": [],
            "top_documents": [doc.get("doc_id") for doc in top_docs],
        }

        # 1. PRF primero (establece baseline de documentos relevantes)
        self._add_pseudo_relevance_terms(top_docs, candidates, language, trace)

        # 2. Rocchio segundo (repondera candidatos con vectores TF-IDF)
        self._add_rocchio_terms(query, top_docs, feedback, candidates, language, trace)

        # 3. Sinónimos tercero (expande términos de consulta original)
        self._add_synonym_terms(query, candidates, language, trace)

        # 4. Co-ocurrencia último (encuentra contexto relacionado)
        self._add_cooccurrence_terms(query, top_docs, candidates, language, trace)

        # 5. Feedback (puede estar aquí o en cualquier orden)
        self._add_feedback_terms(feedback, candidates, language, trace)

        ranked_terms = self._rank_terms(candidates, original_terms, max_terms or self.max_terms, language)
        expanded_query = " ".join([query] + ranked_terms).strip() if ranked_terms else query
        result = ExpansionResult(
            original_query=query,
            expanded_query=expanded_query,
            terms=ranked_terms,
            term_scores={term: round(float(candidates.get(term, 0.0)), 4) for term in ranked_terms},
            method=method,
            applied=bool(ranked_terms),
            selected_query=expanded_query,
            selected_strategy="expanded" if ranked_terms else "raw",
            trace=trace,
        )
        self.cache.set(cache_key, result)
        return result

    def record_explicit_feedback(
        self,
        *,
        query: str,
        doc_id: str,
        relevance: int,
        expanded_query: str | None = None,
        search_mode: str | None = None,
    ) -> dict[str, Any]:
        self.cache.clear()
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
        payload, counted = self.feedback_store.add_implicit(
            query=query,
            doc_id=doc_id,
            event=event,
            search_mode=search_mode,
        )
        if counted:
            self.cache.clear()
        return payload, counted

    def _top_documents_limit(self) -> int:
        return max(int(self.config.get("global.top_documents_for_prf", 3)), 1)

    def _cache_key(
        self,
        query: str,
        method: str,
        docs: list[dict[str, Any]],
        max_terms: int | None,
    ) -> str:
        doc_ids = ",".join(str(doc.get("doc_id") or "") for doc in docs)
        return "|".join([normalize_key(query), method, doc_ids, str(max_terms or self.max_terms)])

    def _technique_enabled(self, name: str) -> bool:
        return bool(self.config.get(f"techniques.{name}.enabled", True))

    def _trace(self, trace: dict[str, Any], technique: str, before: int, candidates: dict[str, float]) -> None:
        after = len(candidates)
        trace.setdefault("techniques", []).append(
            {"name": technique, "new_candidates": max(after - before, 0), "total_candidates": after}
        )
        self.logger.trace_candidates(technique, candidates)

    def _add_candidate(
        self,
        candidates: dict[str, float],
        term: str,
        score: float,
        *,
        language: str,
    ) -> None:
        term = normalize_key(term)
        stopwords = get_stopwords(language)
        if not term or term in stopwords or len(term) < 3:
            return
        candidates[term] = candidates.get(term, 0.0) + float(score)

    def _add_synonym_terms(
        self,
        query: str,
        candidates: dict[str, float],
        language: str,
        trace: dict[str, Any],
    ) -> None:
        if not self._technique_enabled("synonyms"):
            return
        before = len(candidates)
        weight = float(self.config.get("techniques.synonyms.weight", 0.85))
        for token in tokenize_raw(query, language=language):
            for synonym in self.synonyms.get(token, []):
                self._add_candidate(candidates, synonym, weight, language=language)
        self._trace(trace, "synonyms", before, candidates)

    def _add_pseudo_relevance_terms(
        self,
        docs: list[dict[str, Any]],
        candidates: dict[str, float],
        language: str,
        trace: dict[str, Any],
    ) -> None:
        if not self._technique_enabled("pseudo_relevance"):
            return
        before = len(candidates)
        title_weight = float(self.config.get("techniques.pseudo_relevance.title_weight", 0.7))
        body_weight = float(self.config.get("techniques.pseudo_relevance.body_weight", 0.35))
        max_title_terms = int(self.config.get("techniques.pseudo_relevance.max_title_terms", 24))
        max_body_terms = int(self.config.get("techniques.pseudo_relevance.max_body_terms", 12))
        for rank, doc in enumerate(docs, start=1):
            rank_weight = 1.0 / rank
            title_tokens = tokenize_raw(str(doc.get("title") or ""), language=language)
            body_tokens = tokenize_raw(
                " ".join(
                    [
                        str(doc.get("summary") or ""),
                        str(doc.get("content_text") or doc.get("content") or ""),
                    ]
                ),
                language=language,
            )
            for term, multiplier in self.ngrams.extract(title_tokens, max_items=max_title_terms):
                self._add_candidate(candidates, term, title_weight * rank_weight * multiplier, language=language)
            for term, freq in self._top_frequency_terms_with_counts(body_tokens, limit=max_body_terms):
                self._add_candidate(candidates, term, body_weight * rank_weight * freq, language=language)
        self._trace(trace, "pseudo_relevance", before, candidates)

    def _add_cooccurrence_terms(
        self,
        query: str,
        docs: list[dict[str, Any]],
        candidates: dict[str, float],
        language: str,
        trace: dict[str, Any],
    ) -> None:
        if not self._technique_enabled("cooccurrence"):
            return
        before = len(candidates)
        query_terms = set(tokenize_raw(query, language=language))
        if not query_terms:
            return
        window_size = max(int(self.config.get("techniques.cooccurrence.window_size", 4)), 1)
        weight = float(self.config.get("techniques.cooccurrence.weight", 0.28))
        for rank, doc in enumerate(docs, start=1):
            tokens = tokenize_raw(
                " ".join(
                    [
                        str(doc.get("title") or ""),
                        str(doc.get("summary") or ""),
                        str(doc.get("content_text") or doc.get("content") or ""),
                    ]
                ),
                language=language,
            )
            rank_weight = 1.0 / rank
            for idx, token in enumerate(tokens):
                if token not in query_terms:
                    continue
                start = max(idx - window_size, 0)
                end = min(idx + window_size + 1, len(tokens))
                for neighbor in tokens[start:end]:
                    if neighbor != token:
                        self._add_candidate(candidates, neighbor, weight * rank_weight, language=language)
        self._trace(trace, "cooccurrence", before, candidates)

    def _add_feedback_terms(
        self,
        feedback: dict[str, list[dict[str, Any]]],
        candidates: dict[str, float],
        language: str,
        trace: dict[str, Any],
    ) -> None:
        if not self._technique_enabled("feedback"):
            return
        before = len(candidates)
        positive_weight = float(self.config.get("techniques.feedback.explicit_positive_weight", 0.95))
        negative_weight = float(self.config.get("techniques.feedback.explicit_negative_weight", -0.75))
        for item in feedback.get("explicit", []):
            doc = self._document_by_id(str(item.get("doc_id") or ""))
            if not doc:
                continue
            score = positive_weight if int(item.get("relevance") or 0) > 0 else negative_weight
            for token in self._top_terms_for_doc(doc, limit=10, language=language):
                self._add_candidate(candidates, token, score, language=language)

        for item in feedback.get("implicit", []):
            doc = self._document_by_id(str(item.get("doc_id") or ""))
            if not doc:
                continue
            score = float(item.get("weight") or self.config.get("techniques.feedback.implicit_default_weight", 0.2))
            for token in self._top_terms_for_doc(doc, limit=6, language=language):
                self._add_candidate(candidates, token, score, language=language)
        self._trace(trace, "feedback", before, candidates)

    def _add_rocchio_terms(
        self,
        query: str,
        top_docs: list[dict[str, Any]],
        feedback: dict[str, list[dict[str, Any]]],
        candidates: dict[str, float],
        language: str,
        trace: dict[str, Any],
    ) -> None:
        if not self._technique_enabled("rocchio"):
            return
        before = len(candidates)
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
        except Exception as exc:
            self.logger.rocchio_error(query, exc)
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

        try:
            pos_vectors = self._vectors_for_doc_ids(positive_ids)
            neg_vectors = self._vectors_for_doc_ids(negative_ids)
            if not pos_vectors:
                return
            alpha = float(self.config.get("techniques.rocchio.alpha", 1.0))
            beta = float(
                self.config.get(
                    "techniques.rocchio.beta_with_explicit"
                    if feedback.get("explicit")
                    else "techniques.rocchio.beta_without_explicit",
                    0.75 if feedback.get("explicit") else 0.25,
                )
            )
            gamma = float(self.config.get("techniques.rocchio.gamma", 0.15))
            rocchio_vector = alpha * query_vector
            rocchio_vector = rocchio_vector + beta * np.mean(pos_vectors, axis=0)
            if neg_vectors:
                rocchio_vector = rocchio_vector - gamma * np.mean(neg_vectors, axis=0)

            inverse_vocab = {idx: term for term, idx in vocabulary.items()}
            stem_to_raw = self._stem_to_raw_map(top_docs, language=language)
            candidate_weight = float(self.config.get("techniques.rocchio.candidate_weight", 0.8))
            max_vector_terms = int(self.config.get("techniques.rocchio.max_vector_terms", 18))
            for idx in np.argsort(rocchio_vector)[::-1][:max_vector_terms]:
                value = float(rocchio_vector[idx])
                if value <= 0:
                    continue
                stem = inverse_vocab.get(int(idx))
                if not stem:
                    continue
                raw_term = stem_to_raw.get(stem, stem)
                self._add_candidate(candidates, raw_term, min(value, 1.0) * candidate_weight, language=language)
        except Exception as exc:
            self.logger.rocchio_error(query, exc)
            return
        self._trace(trace, "rocchio", before, candidates)

    def _rank_terms(
        self,
        candidates: dict[str, float],
        original_terms: set[str],
        limit: int,
        language: str,
    ) -> list[str]:
        ranked = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))
        selected: list[str] = []
        seen: set[str] = set()
        stopwords = get_stopwords(language)
        for term, score in ranked:
            key = normalize_key(term)
            if score <= 0 or key in seen or key in original_terms or key in stopwords:
                continue
            if any(key in other or other in key for other in seen):
                continue
            selected.append(term)
            seen.add(key)
            if len(selected) >= limit:
                break
        return selected

    def _top_frequency_terms_with_counts(self, tokens: list[str], *, limit: int) -> list[tuple[str, float]]:
        counts: dict[str, float] = {}
        for term, multiplier in self.ngrams.extract(tokens):
            if term in self.stopwords:
                continue
            counts[term] = counts.get(term, 0.0) + multiplier
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
        max_count = max((count for _, count in ranked), default=1.0)
        return [(term, count / max_count) for term, count in ranked]

    def _top_terms_for_doc(self, doc: dict[str, Any], *, limit: int, language: str) -> list[str]:
        tokens = tokenize_raw(
            " ".join(
                [
                    str(doc.get("title") or ""),
                    str(doc.get("summary") or ""),
                    str(doc.get("content_text") or doc.get("content") or ""),
                ]
            ),
            language=language,
        )
        return [term for term, _ in self._top_frequency_terms_with_counts(tokens, limit=limit)]

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

    def _stem_to_raw_map(self, docs: list[dict[str, Any]], *, language: str) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for doc in docs:
            raw_text = " ".join(
                [
                    str(doc.get("title") or ""),
                    str(doc.get("summary") or ""),
                    str(doc.get("content_text") or doc.get("content") or ""),
                ]
            )
            for raw_term in tokenize_raw(raw_text, language=language):
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
