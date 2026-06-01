from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.api.config import DEFAULT_DOMAIN_SYNONYMS, DEFAULT_QUERY_EXPANSION_CONFIG
from src.evaluation.constants import SUPPORTED_SYSTEMS
from src.retrieval.query_expansion import QueryExpander
from src.retrieval.search import (
    DEFAULT_LSI_META,
    DEFAULT_LSI_MODEL,
    DEFAULT_LSI_VECTORS,
    DEFAULT_TFIDF_MATRIX,
    DEFAULT_TFIDF_META,
    DEFAULT_TFIDF_VOCAB,
    SemanticSearcher,
)


@dataclass
class RunnableSystem:
    name: str
    label: str
    search: Callable[[str, int], list[dict[str, Any]]]


def missing_lsi_artifacts() -> list[str]:
    matrix_path = Path(DEFAULT_TFIDF_MATRIX)
    matrix_exists = matrix_path.exists() or matrix_path.with_suffix(".npy").exists()
    required_files = [
        DEFAULT_TFIDF_VOCAB,
        DEFAULT_TFIDF_META,
        DEFAULT_LSI_MODEL,
        DEFAULT_LSI_VECTORS,
        DEFAULT_LSI_META,
    ]
    missing = [path for path in required_files if not Path(path).exists()]
    if not matrix_exists:
        missing.insert(0, DEFAULT_TFIDF_MATRIX)
    return missing


def parse_systems(raw_systems: str | list[str] | None) -> list[str]:
    if raw_systems is None:
        return list(SUPPORTED_SYSTEMS)
    if isinstance(raw_systems, str):
        requested = [item.strip() for item in raw_systems.split(",") if item.strip()]
    else:
        requested = [str(item).strip() for item in raw_systems if str(item).strip()]

    if not requested or requested == ["all"] or "all" in requested:
        return list(SUPPORTED_SYSTEMS)

    invalid = [system for system in requested if system not in SUPPORTED_SYSTEMS]
    if invalid:
        raise ValueError(
            f"Sistemas no soportados: {', '.join(invalid)}. "
            f"Usa: {', '.join(SUPPORTED_SYSTEMS)} o all."
        )
    return requested


def build_systems(
    selected_systems: list[str],
    *,
    searcher: SemanticSearcher | None = None,
) -> tuple[dict[str, RunnableSystem], dict[str, str]]:
    runnable: dict[str, RunnableSystem] = {}
    skipped: dict[str, str] = {}

    lsi_systems = {"lsi_baseline", "lsi_refined", "lsi_expanded"} & set(selected_systems)
    if lsi_systems:
        missing = missing_lsi_artifacts()
        if missing:
            reason = "Faltan artefactos LSI/TF-IDF: " + ", ".join(missing)
            for name in lsi_systems:
                skipped[name] = reason
        else:
            searcher = searcher or SemanticSearcher()
            runnable["lsi_baseline"] = RunnableSystem(
                name="lsi_baseline",
                label="TF-IDF + LSI baseline",
                search=lambda query, top_k: searcher.search_baseline(query, top_k=top_k),
            )
            runnable["lsi_refined"] = RunnableSystem(
                name="lsi_refined",
                label="TF-IDF + LSI + reranking",
                search=lambda query, top_k: searcher.search(query, top_k=top_k),
            )
            expander = QueryExpander(
                searcher,
                config_path=DEFAULT_QUERY_EXPANSION_CONFIG,
                synonyms_path=DEFAULT_DOMAIN_SYNONYMS,
            )
            runnable["lsi_expanded"] = RunnableSystem(
                name="lsi_expanded",
                label="TF-IDF + LSI + expansion + reranking",
                search=lambda query, top_k: search_with_expansion(searcher, expander, query, top_k),
            )

    rag_modes = {"vectorial", "hybrid_search"} & set(selected_systems)
    if rag_modes:
        try:
            from src.RAG.rag_pipeline import RAGPipeline

            rag_searcher = searcher
            if "hybrid_search" in rag_modes and rag_searcher is None and not missing_lsi_artifacts():
                rag_searcher = SemanticSearcher()

            rag = RAGPipeline.from_preset(semantic_searcher=rag_searcher)
            runnable["vectorial"] = RunnableSystem(
                name="vectorial",
                label="Vectorial embeddings",
                search=lambda query, top_k: rag_retrieve(rag, query, top_k, "vectorial"),
            )
            runnable["hybrid_search"] = RunnableSystem(
                name="hybrid_search",
                label="Hybrid search RRF",
                search=lambda query, top_k: rag_retrieve(rag, query, top_k, "hybrid_search"),
            )
        except Exception as exc:
            for name in rag_modes:
                skipped[name] = f"No se pudo inicializar RAG/VectorDB: {exc}"

    return (
        {name: system for name, system in runnable.items() if name in selected_systems},
        skipped,
    )


def search_with_expansion(
    searcher: SemanticSearcher,
    expander: QueryExpander,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    preview = searcher.search(query, top_k=max(3, top_k))
    expansion = expander.expand_query(query, top_documents=preview[:3])
    selected_query = expansion.expanded_query if expansion.applied else query
    return searcher.search(selected_query, top_k=top_k)


def rag_retrieve(rag: Any, query: str, top_k: int, mode: str) -> list[dict[str, Any]]:
    documents = rag.retrieve(query, top_k=top_k, search_mode=mode, include_explanations=False)
    return [{"doc_id": doc.doc_id, "score": doc.score, "title": doc.title} for doc in documents]


def system_label(name: str) -> str:
    labels = {
        "lsi_baseline": "TF-IDF + LSI baseline",
        "lsi_refined": "TF-IDF + LSI + reranking",
        "lsi_expanded": "TF-IDF + LSI + expansion + reranking",
        "vectorial": "Vectorial embeddings",
        "hybrid_search": "Hybrid search RRF",
    }
    return labels.get(name, name)
