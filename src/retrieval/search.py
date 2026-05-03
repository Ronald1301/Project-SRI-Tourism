from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np

from src.indexing.tfidf_index import TFIDFIndex
from src.preprocessing.pipeline import PreprocessingPipeline
from src.retrieval.lsi_model import LSIModel
from src.vector_db.preset import resolve_documents_path

DEFAULT_TFIDF_MATRIX = "data/index/tfidf_matrix.npy"
DEFAULT_TFIDF_VOCAB = "data/index/vocabulary.json"
DEFAULT_TFIDF_META = "data/index/tfidf_meta.json"

DEFAULT_LSI_MODEL = "data/index/lsi_model.pkl"
DEFAULT_LSI_VECTORS = "data/index/doc_vectors.npy"
DEFAULT_LSI_META = "data/index/lsi_metadata.json"
DEFAULT_SCORE_THRESHOLD = 0.2
DEFAULT_INITIAL_POOL_SIZE = 20
DEFAULT_SNIPPET_LENGTH = 280
BASELINE_RETRIEVAL_MODEL = "tfidf+lsi"
RERANK_RETRIEVAL_MODEL = "tfidf+lsi+rerank"


def cosine_similarity(query_vector: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    if doc_matrix is None or doc_matrix.size == 0:
        return np.array([], dtype=float)

    query_vector = np.asarray(query_vector, dtype=float).reshape(-1)
    doc_matrix = np.asarray(doc_matrix, dtype=float)

    doc_norms = np.linalg.norm(doc_matrix, axis=1)
    query_norm = np.linalg.norm(query_vector)
    denom = doc_norms * query_norm

    sims = np.zeros(doc_matrix.shape[0], dtype=float)
    valid = denom > 0
    if np.any(valid):
        sims[valid] = (doc_matrix[valid] @ query_vector) / denom[valid]
    return sims


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _truncate_text(text: str, max_length: int = DEFAULT_SNIPPET_LENGTH) -> str:
    normalized = _normalize_whitespace(text)
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip() + "..."


class SemanticSearcher:
    def __init__(
        self,
        *,
        language: str = "spanish",
        tfidf_matrix_path: str = DEFAULT_TFIDF_MATRIX,
        tfidf_vocab_path: str = DEFAULT_TFIDF_VOCAB,
        tfidf_meta_path: str = DEFAULT_TFIDF_META,
        lsi_model_path: str = DEFAULT_LSI_MODEL,
        lsi_vectors_path: str = DEFAULT_LSI_VECTORS,
        lsi_meta_path: str = DEFAULT_LSI_META,
        documents_path: str | Path | None = None,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        initial_pool_size: int = DEFAULT_INITIAL_POOL_SIZE,
    ) -> None:
        self.pipeline = PreprocessingPipeline(language=language)
        self.language = language
        self.score_threshold = float(score_threshold)
        self.initial_pool_size = max(int(initial_pool_size), 1)

        self.tfidf_index = TFIDFIndex.load(
            tfidf_matrix_path,
            tfidf_vocab_path,
            tfidf_meta_path,
        )

        self.lsi_model = LSIModel.load(
            model_path=lsi_model_path,
            vectors_path=lsi_vectors_path,
            metadata_path=lsi_meta_path,
        )

        if self.lsi_model.doc_vectors is None:
            raise ValueError("LSI document vectors are missing. Train and save LSI first.")

        if self.tfidf_index.doc_ids:
            self.doc_ids = list(self.tfidf_index.doc_ids)
        else:
            self.doc_ids = [str(i) for i in range(self.lsi_model.doc_vectors.shape[0])]

        self.documents_path = self._resolve_documents_path(documents_path)
        self.documents_by_id = self._load_documents_by_id(self.documents_path)

    def _resolve_documents_path(self, documents_path: str | Path | None) -> Path | None:
        if documents_path:
            path = Path(documents_path)
            return path if path.exists() else None
        try:
            return resolve_documents_path()
        except FileNotFoundError:
            return None

    def _load_documents_by_id(self, documents_path: Path | None) -> dict[str, dict[str, Any]]:
        if documents_path is None or not documents_path.exists():
            return {}

        documents: dict[str, dict[str, Any]] = {}
        for item in _iter_jsonl(documents_path):
            doc_id = item.get("doc_id")
            if not isinstance(doc_id, str) or not doc_id.strip():
                continue

            title = str(item.get("title") or item.get("entity_name") or "").strip()
            content = str(item.get("content_text") or item.get("summary") or "").strip()
            full_text = "\n".join(part for part in [title, content] if part).strip()
            title_tokens = self.pipeline.process_text(title) if title else []
            content_tokens = self.pipeline.process_text(full_text) if full_text else []

            documents[doc_id] = {
                "doc_id": doc_id,
                "title": title,
                "content": content,
                "summary": item.get("summary"),
                "url": item.get("url"),
                "location": item.get("location"),
                "review_date": item.get("review_date"),
                "rating": item.get("rating"),
                "content_type": item.get("content_type"),
                "entity_name": item.get("entity_name"),
                "word_count": int(item.get("word_count") or 0),
                "source": "local",
                "metadata": item,
                "title_tokens": set(title_tokens),
                "content_tokens": set(content_tokens),
            }
        return documents

    def _extract_best_snippet(self, query: str, query_tokens: list[str], document: dict[str, Any]) -> str:
        summary = str(document.get("summary") or "").strip()
        if summary:
            return _truncate_text(summary)

        content = str(document.get("content") or "").strip()
        if not content:
            return ""

        query_token_set = set(query_tokens)
        paragraphs = [segment.strip() for segment in re.split(r"\n+", content) if segment.strip()]
        if not paragraphs:
            return _truncate_text(content)

        best_paragraph = paragraphs[0]
        best_score = -1.0
        for paragraph in paragraphs:
            paragraph_tokens = set(self.pipeline.process_text(paragraph))
            overlap = (
                len(query_token_set & paragraph_tokens) / max(len(query_token_set), 1)
                if query_token_set
                else 0.0
            )
            density_bonus = min(len(paragraph) / 240.0, 1.0) * 0.05
            score = overlap + density_bonus
            if score > best_score:
                best_score = score
                best_paragraph = paragraph

        if best_score <= 0 and query.strip():
            return _truncate_text(content)
        return _truncate_text(best_paragraph)

    def _build_result(
        self,
        *,
        doc_id: str,
        rank: int,
        final_score: float,
        lsi_score: float,
        lexical_overlap: float,
        title_match: float,
        length_signal: float,
        query: str,
        query_tokens: list[str],
        retrieval_model: str,
    ) -> dict[str, Any]:
        document = self.documents_by_id.get(doc_id, {})
        title = str(document.get("title") or document.get("entity_name") or "")
        content = str(document.get("content") or document.get("summary") or "")
        snippet = self._extract_best_snippet(query, query_tokens, document)

        return {
            "doc_id": doc_id,
            "rank": rank,
            "score": float(final_score),
            "lsi_score": float(lsi_score),
            "score_components": {
                "lsi_score": float(lsi_score),
                "lexical_overlap": float(lexical_overlap),
                "title_match": float(title_match),
                "length_signal": float(length_signal),
            },
            "title": title,
            "content": content,
            "snippet": snippet,
            "url": document.get("url"),
            "location": document.get("location"),
            "review_date": document.get("review_date"),
            "rating": document.get("rating"),
            "content_type": document.get("content_type"),
            "source": document.get("source", "local"),
            "retrieval_model": retrieval_model,
        }

    def _score_candidate(
        self,
        *,
        doc_id: str,
        lsi_score: float,
        query_tokens: list[str],
    ) -> tuple[float, float, float, float]:
        document = self.documents_by_id.get(doc_id, {})
        query_token_set = set(query_tokens)
        content_tokens = document.get("content_tokens") or set()
        title_tokens = document.get("title_tokens") or set()
        word_count = int(document.get("word_count") or 0)

        lexical_overlap = (
            len(query_token_set & content_tokens) / max(len(query_token_set), 1)
            if query_token_set
            else 0.0
        )
        title_match = (
            len(query_token_set & title_tokens) / max(len(query_token_set), 1)
            if query_token_set
            else 0.0
        )
        length_signal = min(word_count / 120.0, 1.0)
        final_score = (
            0.70 * float(lsi_score)
            + 0.20 * lexical_overlap
            + 0.08 * title_match
            + 0.02 * length_signal
        )
        return final_score, lexical_overlap, title_match, length_signal

    def _prepare_query(self, query: str) -> tuple[list[str], np.ndarray]:
        tokens = self.pipeline.process_text(query)
        query_vector = self.tfidf_index.vectorize_query(tokens)
        query_lsi = self.lsi_model.transform_query(query_vector)
        return tokens, query_lsi

    def search_baseline(self, query: str, top_k: int = 10) -> List[dict[str, Any]]:
        if not query or not query.strip():
            return []

        top_k = max(int(top_k), 0)
        if top_k == 0:
            return []

        tokens, query_lsi = self._prepare_query(query)
        similarities = cosine_similarity(query_lsi, self.lsi_model.doc_vectors)
        if similarities.size == 0:
            return []

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results: list[dict[str, Any]] = []
        for rank, idx in enumerate(top_indices, start=1):
            doc_id = self.doc_ids[idx]
            lsi_score = float(similarities[idx])
            results.append(
                self._build_result(
                    doc_id=doc_id,
                    rank=rank,
                    final_score=lsi_score,
                    lsi_score=lsi_score,
                    lexical_overlap=0.0,
                    title_match=0.0,
                    length_signal=0.0,
                    query=query,
                    query_tokens=tokens,
                    retrieval_model=BASELINE_RETRIEVAL_MODEL,
                )
            )
        return results

    def search(
        self,
        query: str,
        top_k: int = 10,
        *,
        score_threshold: float | None = None,
        initial_top_k: int | None = None,
    ) -> List[dict[str, Any]]:
        if not query or not query.strip():
            return []

        tokens, query_lsi = self._prepare_query(query)
        similarities = cosine_similarity(query_lsi, self.lsi_model.doc_vectors)
        if similarities.size == 0:
            return []

        top_k = max(int(top_k), 0)
        if top_k == 0:
            return []

        threshold = self.score_threshold if score_threshold is None else float(score_threshold)
        candidate_pool_size = max(
            top_k,
            self.initial_pool_size if initial_top_k is None else max(int(initial_top_k), 1),
        )
        top_indices = np.argsort(similarities)[::-1][:candidate_pool_size]

        reranked_results: list[dict[str, Any]] = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            lsi_score = float(similarities[idx])
            final_score, lexical_overlap, title_match, length_signal = self._score_candidate(
                doc_id=doc_id,
                lsi_score=lsi_score,
                query_tokens=tokens,
            )
            reranked_results.append(
                self._build_result(
                    doc_id=doc_id,
                    rank=0,
                    final_score=final_score,
                    lsi_score=lsi_score,
                    lexical_overlap=lexical_overlap,
                    title_match=title_match,
                    length_signal=length_signal,
                    query=query,
                    query_tokens=tokens,
                    retrieval_model=RERANK_RETRIEVAL_MODEL,
                )
            )

        reranked_results.sort(key=lambda item: item["score"], reverse=True)

        filtered_results: list[dict[str, Any]] = []
        for result in reranked_results:
            if result["score"] < threshold:
                continue
            filtered_results.append(result)
            if len(filtered_results) >= top_k:
                break

        for rank, result in enumerate(filtered_results, start=1):
            result["rank"] = rank

        return filtered_results
