from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np # type: ignore

logger = logging.getLogger("src.RAG.rag_pipeline")

from src.RAG.rag_answer_generator import RAGAnswerGenerator
from src.indexing.tfidf_index import TFIDFIndex
from src.preprocessing.pipeline import PreprocessingPipeline
from src.retrieval.search import SemanticSearcher
from src.utils.file_manager import save_documents_to_jsonl
from src.vector_db.preset import OUTPUT_DIR, resolve_documents_path
from src.vector_db.vector_store import VectorDatabase
from src.web_crawler.insufficiency_policy import InsufficiencyPolicy
from src.web_crawler.web_search_client import DuckDuckGoWebSearchClient


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


@dataclass(frozen=True)
class RetrievedDocument:
    citation_id: int
    doc_id: str
    title: str
    url: str
    score: float
    summary: str
    content_text: str
    metadata: dict


@dataclass(frozen=True)
class CandidatePassage:
    citation_id: int
    title: str
    text: str
    score: float


@dataclass(frozen=True)
class RAGResult:
    query: str
    prompt: str
    answer: str
    documents: list[RetrievedDocument]


class DocumentRepository:
    def __init__(self, documents: dict[str, dict]) -> None:
        self.documents = documents

    @classmethod
    def from_jsonl(cls, path: Path) -> "DocumentRepository":
        documents: dict[str, dict] = {}
        for doc in iter_jsonl(path):
            doc_id = str(doc.get("doc_id") or "").strip()
            if not doc_id:
                continue
            documents[doc_id] = doc
        return cls(documents)

    def get(self, doc_id: str) -> dict | None:
        return self.documents.get(doc_id)


class RAGPipeline:
    SUPPORTED_SEARCH_MODES = {"vectorial", "lsi", "hybrid_search"}

    def __init__(
        self,
        vector_db: VectorDatabase,
        repository: DocumentRepository,
        *,
        language: str = "spanish",
        semantic_searcher: SemanticSearcher | None = None,
        insufficiency_policy: InsufficiencyPolicy | None = None,
        web_search_client: DuckDuckGoWebSearchClient | None = None,
        web_search_output_path: Path | None = None,
    ) -> None:
        self.vector_db = vector_db
        self.repository = repository
        self.semantic_searcher = semantic_searcher
        self.insufficiency_policy = insufficiency_policy or InsufficiencyPolicy()
        self.web_search_client = web_search_client or DuckDuckGoWebSearchClient()
        self.web_search_output_path = (
            Path(web_search_output_path)
            if web_search_output_path is not None
            else Path("data/raw/web_documents.jsonl")
        )
        self.answer_generator = RAGAnswerGenerator()
        self.preprocessing = PreprocessingPipeline(language=language)
        self._document_token_cache: dict[str, set[str]] = {}
        self._title_token_cache: dict[str, set[str]] = {}
        self._fallback_index: TFIDFIndex | None = None
        self._vector_search_available = self._has_local_vector_model()

    @classmethod
    def from_preset(
        cls,
        *,
        output_dir: Path = OUTPUT_DIR,
        language: str = "spanish",
        semantic_searcher: SemanticSearcher | None = None,
        insufficiency_policy: InsufficiencyPolicy | None = None,
        web_search_client: DuckDuckGoWebSearchClient | None = None,
        web_search_output_path: Path | None = None,
    ) -> "RAGPipeline":
        logger.info("Inicializando RAGPipeline | output_dir=%s | language=%s", output_dir, language)
        documents_path = resolve_documents_path()
        vector_db = VectorDatabase.load(Path(output_dir))
        repository = DocumentRepository.from_jsonl(documents_path)
        logger.info("VectorDB cargada: %d documentos | repositorio: %s", len(vector_db.doc_ids), documents_path)
        return cls(
            vector_db,
            repository,
            language=language,
            semantic_searcher=semantic_searcher,
            insufficiency_policy=insufficiency_policy,
            web_search_client=web_search_client,
            web_search_output_path=web_search_output_path,
        )

    def answer_query(
        self,
        query: str,
        top_k: int = 4,
        *,
        search_mode: str = "vectorial",
        include_explanations: bool = False,
    ) -> RAGResult:
        logger.info("answer_query | query=\"%s\" | mode=%s | top_k=%d", query, search_mode, top_k)
        local_documents = self.retrieve(
            query,
            top_k=top_k,
            search_mode=search_mode,
            include_explanations=include_explanations,
        )
        logger.info("Recuperacion local: %d documentos", len(local_documents))

        if self._is_retrieval_insufficient(local_documents):
            logger.info("Recuperacion local INSUFICIENTE — activando busqueda web")
            web_documents = self._retrieve_and_index_web_context(query, top_k=top_k)
            if web_documents:
                logger.info("Web search obtuvo %d documentos, re-consultando vectorial...", len(web_documents))
                augmented_documents = self.retrieve(
                    query,
                    top_k=top_k,
                    search_mode="vectorial",
                    include_explanations=False,
                )
                if augmented_documents:
                    documents = augmented_documents
                    logger.info("Re-consulta exitosa: %d documentos", len(documents))
                else:
                    documents = (local_documents + web_documents)[: max(int(top_k), 0)]
                    logger.info("Usando merge local+web: %d documentos", len(documents))
            else:
                documents = local_documents
                logger.info("Web search no retorno documentos, usando solo locales")
        else:
            documents = local_documents
            logger.info("Recuperacion local suficiente")

        logger.info("Generando respuesta con LLM...")
        prompt = self.answer_generator.build_prompt(query, documents)
        answer = self.answer_generator.generate(query, documents, prompt=prompt)
        logger.info("Respuesta generada | answer_len=%d", len(answer))
        return RAGResult(
            query=query,
            prompt=prompt,
            answer=answer,
            documents=documents,
        )

    def answer_with_lsi(
        self,
        query: str,
        lsi_results: list[dict],
        top_k: int = 4,
    ) -> RAGResult:
        documents = self._convert_lsi_results(lsi_results[:top_k])
        prompt = self.answer_generator.build_prompt(query, documents)
        answer = self.answer_generator.generate(query, documents, prompt=prompt)
        return RAGResult(
            query=query,
            prompt=prompt,
            answer=answer,
            documents=documents,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        *,
        search_mode: str = "vectorial",
        include_explanations: bool = False,
    ) -> list[RetrievedDocument]:
        mode = self._normalize_search_mode(search_mode)
        logger.info("retrieve | mode=%s | top_k=%d", mode, top_k)

        if mode == "vectorial":
            raw_results = self._search_vectorial_raw(query, top_k=top_k)
        elif mode == "lsi":
            logger.info("Ejecutando busqueda LSI...")
            raw_results = self._search_lsi_raw(
                query,
                top_k=top_k,
                include_explanations=include_explanations,
            )
        else:
            pool_k = max(int(top_k) * 4, int(top_k))
            logger.info("Ejecutando busqueda hibrida (pool_k=%d)...", pool_k)
            vector_results = self._search_vectorial_raw(query, top_k=pool_k)
            lsi_results = self._search_lsi_raw(
                query,
                top_k=pool_k,
                include_explanations=include_explanations,
            )
            raw_results = self._rrf_fuse(
                [vector_results, lsi_results],
                top_k=top_k,
                rrf_k=60,
            )
            logger.info("RRF fusion completa: %d resultados", len(raw_results))

        documents: list[RetrievedDocument] = []

        for citation_id, item in enumerate(raw_results, start=1):
            doc_id = str(item.get("doc_id") or "").strip()
            source_doc = self.repository.get(doc_id) or {}

            merged = dict(source_doc)
            for key, value in item.items():
                merged.setdefault(key, value)

            title = str(
                merged.get("title")
                or merged.get("entity_name")
                or merged.get("doc_id")
                or f"Documento {citation_id}"
            ).strip()
            summary = str(merged.get("summary") or "").strip()
            content_text = str(merged.get("content_text") or "").strip()
            url = str(merged.get("url") or "").strip()
            score = float(item.get("score", 0.0))

            documents.append(
                RetrievedDocument(
                    citation_id=citation_id,
                    doc_id=doc_id,
                    title=title,
                    url=url,
                    score=score,
                    summary=summary,
                    content_text=content_text,
                    metadata=merged,
                )
            )

        return documents

    def _normalize_search_mode(self, search_mode: str | None) -> str:
        mode = str(search_mode or "").strip().lower()
        aliases = {
            "hybrid": "hybrid_search",
            "hibrido": "hybrid_search",
            "híbrido": "hybrid_search",
        }
        mode = aliases.get(mode, mode)
        if mode not in self.SUPPORTED_SEARCH_MODES:
            raise ValueError(
                f"search_mode invalido: '{search_mode}'. "
                "Usa: 'lsi', 'vectorial' o 'hybrid_search'."
            )
        return mode

    def _search_vectorial_raw(self, query: str, top_k: int) -> list[dict]:
        if self._vector_search_available:
            try:
                logger.info("Busqueda vectorial en VectorDatabase...")
                results = self.vector_db.search(query, top_k=top_k)
                logger.info("Vectorial retorno %d resultados", len(results))
                return results
            except Exception:
                logger.warning("Busqueda vectorial fallo, usando fallback TF-IDF")
                self._vector_search_available = False
        logger.info("Vectorial no disponible, usando fallback TF-IDF")
        return self._tfidf_search(query, top_k=top_k)

    def _search_lsi_raw(
        self,
        query: str,
        top_k: int,
        *,
        include_explanations: bool,
    ) -> list[dict]:
        if self.semantic_searcher is None:
            logger.info("LSI no disponible: semantic_searcher es None")
            return []

        try:
            raw = self.semantic_searcher.search(
                query,
                top_k=top_k,
                include_explanations=include_explanations,
            )
        except Exception as exc:
            logger.warning("Busqueda LSI fallo: %s", exc)
            return []

        logger.info("LSI retorno %d resultados raw", len(raw))
        results: list[dict] = []
        for item in raw:
            payload: dict[str, Any] = {
                "doc_id": str(item.get("doc_id") or "").strip(),
                "title": item.get("title") or "",
                "score": float(item.get("score", 0.0)),
                "summary": item.get("snippet") or item.get("summary") or "",
                "url": item.get("url") or "",
            }
            if include_explanations and isinstance(item.get("explanation"), dict):
                payload["explanation"] = item["explanation"]
            if payload["doc_id"]:
                results.append(payload)
        return results

    def _rrf_fuse(
        self,
        result_lists: list[list[dict]],
        *,
        top_k: int,
        rrf_k: int = 60,
    ) -> list[dict]:
        if top_k <= 0:
            return []
        accumulator: dict[str, dict[str, Any]] = {}

        for result_list in result_lists:
            for rank, item in enumerate(result_list, start=1):
                doc_id = str(item.get("doc_id") or "").strip()
                if not doc_id:
                    continue

                contribution = 1.0 / (float(rrf_k) + float(rank))
                if doc_id not in accumulator:
                    base_payload = dict(item)
                    base_payload["score"] = 0.0
                    accumulator[doc_id] = base_payload

                accumulator[doc_id]["score"] = float(accumulator[doc_id]["score"]) + contribution

                if not accumulator[doc_id].get("title") and item.get("title"):
                    accumulator[doc_id]["title"] = item.get("title")
                if not accumulator[doc_id].get("url") and item.get("url"):
                    accumulator[doc_id]["url"] = item.get("url")
                if not accumulator[doc_id].get("summary") and item.get("summary"):
                    accumulator[doc_id]["summary"] = item.get("summary")

                if isinstance(item.get("explanation"), dict):
                    accumulator[doc_id]["explanation"] = item["explanation"]

        merged = sorted(
            accumulator.values(),
            key=lambda payload: float(payload.get("score", 0.0)),
            reverse=True,
        )
        return merged[:top_k]

    def _is_retrieval_insufficient(self, documents: list[RetrievedDocument]) -> bool:
        scores = [doc.score for doc in documents]
        result = self.insufficiency_policy.is_insufficient(
            [{"doc_id": doc.doc_id, "score": float(doc.score)} for doc in documents]
        )
        if result and documents:
            logger.info("Recuperacion insuficiente: scores=%s", [f"{s:.4f}" for s in scores])
        return result

    def _retrieve_and_index_web_context(self, query: str, *, top_k: int) -> list[RetrievedDocument]:
        max_results = max(int(top_k) * 2, int(top_k), 1)
        logger.info("Buscando en DuckDuckGo (max_results=%d)...", max_results)
        web_documents_raw = self.web_search_client.search(query, max_results=max_results)
        if not web_documents_raw:
            logger.info("Web search: sin resultados")
            return []

        logger.info("Web search obtuvo %d documentos raw", len(web_documents_raw))
        normalized_docs = self._normalize_web_documents(web_documents_raw)
        if not normalized_docs:
            logger.info("Web search: ningun documento paso la normalizacion")
            return []

        logger.info("Normalizados: %d documentos. Persistiendo...", len(normalized_docs))
        self._persist_web_documents(normalized_docs)
        logger.info("Indexando en VectorDatabase...")
        self._index_web_documents(normalized_docs)
        self._merge_web_documents_into_repository(normalized_docs)
        logger.info("Web context listo: %d documentos convertidos", min(len(normalized_docs), max(int(top_k), 0)))
        return self._convert_web_docs_to_retrieved(normalized_docs[: max(int(top_k), 0)])

    def _normalize_web_documents(self, documents: list[dict[str, object]]) -> list[dict[str, object]]:
        normalized: list[dict[str, object]] = []
        for item in documents:
            doc_id = str(item.get("doc_id") or "").strip()
            title = str(item.get("title") or item.get("entity_name") or "").strip()
            content_text = str(item.get("content_text") or "").strip()
            summary = str(item.get("summary") or "").strip()
            url = str(item.get("url") or "").strip()
            if not doc_id:
                continue
            if not (title or content_text or summary):
                continue
            payload = dict(item)
            payload["doc_id"] = doc_id
            payload["title"] = title or f"Documento {doc_id[:8]}"
            payload["content_text"] = content_text
            payload["summary"] = summary
            payload["url"] = url
            payload["source"] = "web"
            normalized.append(payload)
        return normalized

    def _persist_web_documents(self, documents: list[dict[str, object]]) -> None:
        try:
            output_path = self.web_search_output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_documents_to_jsonl(documents, output_path)
            logger.info("Documentos web persistidos en %s", output_path)
        except Exception as exc:
            logger.warning("Error persistiendo documentos web: %s", exc)

    def _index_web_documents(self, documents: list[dict[str, object]]) -> None:
        index_ready_docs: list[dict[str, object]] = []
        for item in documents:
            combined_text = "\n".join(
                [
                    str(item.get("title") or ""),
                    str(item.get("summary") or ""),
                    str(item.get("content_text") or ""),
                ]
            ).strip()
            tokens = self.preprocessing.process_text(combined_text)
            if not tokens:
                continue
            processed = dict(item)
            processed["content_text"] = " ".join(tokens)
            index_ready_docs.append(processed)

        if not index_ready_docs:
            logger.info("Index web: 0 documentos listos para indexar")
            return

        logger.info("Indexando %d documentos web en VectorDatabase...", len(index_ready_docs))
        try:
            self.vector_db.add_documents(
                index_ready_docs,
                store_fields=[
                    "url",
                    "title",
                    "summary",
                    "content_text",
                    "content_type",
                    "rating",
                    "review_date",
                    "location",
                    "source",
                ],
                persist=True,
                output_dir=Path(OUTPUT_DIR),
            )
            logger.info("Indexacion web completada")
        except Exception as exc:
            logger.warning("Error indexando documentos web: %s", exc)

    def _merge_web_documents_into_repository(self, documents: list[dict[str, object]]) -> None:
        for item in documents:
            doc_id = str(item.get("doc_id") or "").strip()
            if not doc_id:
                continue
            self.repository.documents[doc_id] = dict(item)
            self._document_token_cache.pop(doc_id, None)
            self._title_token_cache.pop(doc_id, None)

    def _convert_web_docs_to_retrieved(self, web_docs: list[dict[str, object]]) -> list[RetrievedDocument]:
        converted: list[RetrievedDocument] = []
        for citation_id, item in enumerate(web_docs, start=1):
            doc_id = str(item.get("doc_id") or "").strip()
            if not doc_id:
                continue
            converted.append(
                RetrievedDocument(
                    citation_id=citation_id,
                    doc_id=doc_id,
                    title=str(item.get("title") or item.get("entity_name") or f"Documento {citation_id}"),
                    url=str(item.get("url") or ""),
                    score=float(item.get("score", 0.0)),
                    summary=str(item.get("summary") or ""),
                    content_text=str(item.get("content_text") or ""),
                    metadata=dict(item),
                )
            )
        return converted

    def _convert_lsi_results(self, lsi_results: list[dict]) -> list[RetrievedDocument]:
        documents: list[RetrievedDocument] = []
        for citation_id, item in enumerate(lsi_results, start=1):
            doc_id = str(item.get("doc_id") or "").strip()
            source_doc = self.repository.get(doc_id) or {}

            merged = dict(source_doc)
            for key, value in item.items():
                merged.setdefault(key, value)

            title = str(
                merged.get("title")
                or merged.get("entity_name")
                or merged.get("doc_id")
                or f"Documento {citation_id}"
            ).strip()
            summary = str(merged.get("summary") or "").strip()
            content_text = str(merged.get("content") or merged.get("content_text") or "").strip()
            url = str(merged.get("url") or "").strip()
            score = float(item.get("score", 0.0))

            documents.append(
                RetrievedDocument(
                    citation_id=citation_id,
                    doc_id=doc_id,
                    title=title,
                    url=url,
                    score=score,
                    summary=summary,
                    content_text=content_text,
                    metadata=merged,
                )
            )
        return documents

    def _tfidf_search(self, query: str, top_k: int = 4) -> list[dict]:
        index = self._get_fallback_index()
        query_tokens = self.preprocessing.process_text(query)
        if not query_tokens or index.matrix is None or index.matrix.size == 0:
            return []
        focus_tokens = self._focus_query_tokens(set(query_tokens))

        query_vector = index.vectorize_query(query_tokens)
        doc_matrix = np.asarray(index.matrix, dtype=float)
        doc_norms = np.linalg.norm(doc_matrix, axis=1)
        query_norm = np.linalg.norm(query_vector)
        denom = doc_norms * query_norm

        similarities = np.zeros(doc_matrix.shape[0], dtype=float)
        valid = denom > 0
        if np.any(valid):
            similarities[valid] = (doc_matrix[valid] @ query_vector) / denom[valid]

        candidate_count = max(int(top_k) * 6, 12)
        top_indices = np.argsort(similarities)[::-1][:candidate_count]
        reranked_results: list[dict] = []
        for index_pos in top_indices:
            base_score = float(similarities[index_pos])
            if base_score <= 0:
                continue

            doc_id = index.doc_ids[index_pos]
            source_doc = self.repository.get(doc_id) or {}
            title = str(source_doc.get("title") or source_doc.get("entity_name") or "").strip()
            url = str(source_doc.get("url") or "").strip()
            summary = str(source_doc.get("summary") or "").strip()
            content_text = str(source_doc.get("content_text") or "").strip()
            title_tokens = self._get_title_tokens(doc_id, title)
            doc_tokens = self._get_document_tokens(doc_id, title, summary, content_text)
            focus_overlap = len(focus_tokens & doc_tokens)
            title_focus_overlap = len(focus_tokens & title_tokens)
            score = base_score + (title_focus_overlap * 1.1) + (focus_overlap * 0.18)

            reranked_results.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "url": url,
                    "score": score,
                }
            )

        reranked_results.sort(key=lambda item: item["score"], reverse=True)
        return reranked_results[: max(int(top_k), 0)]

    def build_prompt(self, query: str, documents: list[RetrievedDocument]) -> str:
        if not documents:
            context_block = "No se recuperaron documentos relevantes."
        else:
            context_parts: list[str] = []
            for doc in documents:
                snippet = self._best_passages_for_document(doc, query, limit=1)
                excerpt = snippet[0].text if snippet else self._fallback_excerpt(doc)
                context_parts.append(
                    "\n".join(
                        [
                            f"[{doc.citation_id}] Titulo: {doc.title}",
                            f"URL: {doc.url or 'N/D'}",
                            f"Score: {doc.score:.4f}",
                            f"Contexto: {excerpt}",
                        ]
                    )
                )
            context_block = "\n\n".join(context_parts)

        return "\n".join(
            [
                "Eres un asistente RAG especializado en turismo.",
                "Responde unicamente con la evidencia del contexto recuperado.",
                "Reglas:",
                "1. No inventes hechos, precios, fechas o ubicaciones que no aparezcan en el contexto.",
                "2. Si la evidencia es insuficiente o parcial, dilo explicitamente.",
                "3. Integra detalles concretos del contexto y ancla cada idea con citas [1], [2], etc.",
                "4. Prioriza claridad, sintesis y fidelidad al contexto recuperado.",
                "5. Responde en espanol.",
                "",
                "Consulta del usuario:",
                query.strip(),
                "",
                "Contexto recuperado:",
                context_block,
                "",
                "Formato esperado:",
                "- Un parrafo breve que responda la consulta.",
                "- Uno o dos detalles complementarios si aportan valor.",
                "- No extrapoles mas alla de la evidencia.",
            ]
        )

    def generate_answer(self, query: str, documents: list[RetrievedDocument]) -> str:
        if not documents:
            return (
                "No encontre documentos suficientemente relevantes para responder con evidencia "
                f"a la consulta: {query}."
            )

        passages = self._select_passages(query, documents, max_passages=3)
        if not passages:
            titles = ", ".join(f"[{doc.citation_id}] {doc.title}" for doc in documents[:3])
            return (
                f"Recupere documentos relacionados con la consulta, entre ellos {titles}, "
                "pero el contenido disponible no alcanza para construir una respuesta mas detallada."
            )

        answer_parts: list[str] = []
        prefixes = [
            "Segun la informacion recuperada, ",
            "Ademas, ",
            "Tambien, ",
        ]

        for index, passage in enumerate(passages):
            prefix = prefixes[index] if index < len(prefixes) else ""
            answer_parts.append(
                self._with_citation(prefix, passage.text, passage.citation_id)
            )

        if len(documents) > 1:
            sources = ", ".join(
                f"[{doc.citation_id}] {doc.title}" for doc in documents[: min(3, len(documents))]
            )
            answer_parts.append(
                f"Las fuentes principales consultadas fueron {sources}."
            )

        return " ".join(answer_parts)

    def _select_passages(
        self,
        query: str,
        documents: list[RetrievedDocument],
        *,
        max_passages: int,
    ) -> list[CandidatePassage]:
        ranked: list[CandidatePassage] = []
        for doc in documents:
            ranked.extend(self._best_passages_for_document(doc, query, limit=1))

        ranked.sort(key=lambda item: item.score, reverse=True)
        if ranked:
            return ranked[:max_passages]

        fallbacks: list[CandidatePassage] = []
        for doc in documents[:max_passages]:
            excerpt = self._fallback_excerpt(doc)
            if not excerpt:
                continue
            fallbacks.append(
                CandidatePassage(
                    citation_id=doc.citation_id,
                    title=doc.title,
                    text=excerpt,
                    score=doc.score,
                )
            )
        return fallbacks

    def _best_passages_for_document(
        self,
        doc: RetrievedDocument,
        query: str,
        *,
        limit: int,
    ) -> list[CandidatePassage]:
        query_tokens = set(self.preprocessing.process_text(query))
        focus_tokens = self._focus_query_tokens(query_tokens)
        active_query_tokens = focus_tokens or query_tokens
        title_tokens = self._get_title_tokens(doc.doc_id, doc.title)
        title_focus_overlap = len(active_query_tokens & title_tokens)
        candidates: list[CandidatePassage] = []
        seen: set[str] = set()

        for raw_text in self._candidate_segments(doc):
            normalized_text = self._normalize_whitespace(raw_text)
            if not self._is_useful_segment(normalized_text, doc.title):
                continue
            normalized_key = normalized_text.casefold()
            if normalized_key in seen:
                continue
            seen.add(normalized_key)

            segment_tokens = set(self.preprocessing.process_text(normalized_text))
            if (
                active_query_tokens
                and title_focus_overlap == 0
                and not (active_query_tokens & segment_tokens)
            ):
                continue

            score = self._score_segment(
                active_query_tokens,
                segment_tokens,
                doc.score,
                len(normalized_text),
                title_focus_overlap,
            )
            candidates.append(
                CandidatePassage(
                    citation_id=doc.citation_id,
                    title=doc.title,
                    text=normalized_text,
                    score=score,
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[:limit]

    def _get_fallback_index(self) -> TFIDFIndex:
        if self._fallback_index is None:
            documents: dict[str, list[str]] = {}
            for doc_id, source_doc in self.repository.documents.items():
                title = str(source_doc.get("title") or source_doc.get("entity_name") or "").strip()
                summary = str(source_doc.get("summary") or "").strip()
                content_text = str(source_doc.get("content_text") or "").strip()

                title_tokens = self.preprocessing.process_text(title)
                summary_tokens = self.preprocessing.process_text(summary)
                content_tokens = self.preprocessing.process_text(content_text)

                weighted_tokens = (title_tokens * 3) + (summary_tokens * 2) + content_tokens
                if weighted_tokens:
                    documents[doc_id] = weighted_tokens

            index = TFIDFIndex()
            index.build(documents)
            self._fallback_index = index
        return self._fallback_index

    def _candidate_segments(self, doc: RetrievedDocument) -> Iterable[str]:
        if doc.summary:
            yield doc.summary

        text = doc.content_text
        if not text:
            return

        paragraphs = [part.strip() for part in re.split(r"\n+", text) if part.strip()]
        for paragraph in paragraphs:
            if len(paragraph) <= 360:
                yield paragraph
                continue

            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) >= 50:
                    yield sentence

    def _get_document_tokens(
        self,
        doc_id: str,
        title: str,
        summary: str,
        content_text: str,
    ) -> set[str]:
        if doc_id not in self._document_token_cache:
            combined = "\n".join(part for part in [title, summary, content_text] if part)
            self._document_token_cache[doc_id] = set(self.preprocessing.process_text(combined))
        return self._document_token_cache[doc_id]

    def _get_title_tokens(self, doc_id: str, title: str) -> set[str]:
        if doc_id not in self._title_token_cache:
            self._title_token_cache[doc_id] = set(self.preprocessing.process_text(title))
        return self._title_token_cache[doc_id]

    def _score_segment(
        self,
        query_tokens: set[str],
        segment_tokens: set[str],
        doc_score: float,
        segment_length: int,
        title_focus_overlap: int,
    ) -> float:
        overlap = len(query_tokens & segment_tokens)
        coverage = overlap / max(len(query_tokens), 1)
        density = overlap / max(len(segment_tokens), 1)
        length_bonus = min(segment_length, 220) / 220.0 * 0.2
        short_penalty = 0.2 if segment_length < 70 else 0.0
        return (
            float(doc_score)
            + (coverage * 1.2)
            + (density * 0.35)
            + min(overlap, 4) * 0.12
            + (title_focus_overlap * 0.15)
            + length_bonus
            - short_penalty
        )

    def _fallback_excerpt(self, doc: RetrievedDocument) -> str:
        for raw_text in self._candidate_segments(doc):
            normalized = self._normalize_whitespace(raw_text)
            if self._is_useful_segment(normalized, doc.title):
                text = normalized
                break
        else:
            text = doc.summary or doc.content_text or doc.title

        cleaned = self._normalize_whitespace(text)
        if len(cleaned) <= 280:
            return cleaned
        return cleaned[:277].rstrip() + "..."

    def _with_citation(self, prefix: str, text: str, citation_id: int) -> str:
        cleaned = self._normalize_whitespace(text).rstrip(".!?")
        return f"{prefix}{cleaned} [{citation_id}]."

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        return text.strip("-• ")

    def _is_useful_segment(self, text: str, title: str) -> bool:
        normalized = self._normalize_whitespace(text)
        lowered = normalized.casefold()
        title_lower = self._normalize_whitespace(title).casefold()

        if len(normalized) < 40:
            return False
        if lowered in {"visitar cuba", "organizacion de agencias cubanas", title_lower}:
            return False
        if any(
            phrase in lowered
            for phrase in (
                "visitar cuba es una organizacion de agencias cubanas",
                "si eres una agencia o tour operador",
                "datos personales y de contacto",
                "estoy de acuerdo con la politica de privacidad",
                "mapa interactivo",
            )
        ):
            return False
        if lowered.startswith("ubicacion"):
            return False
        if lowered.startswith("calle ") and "cuba" in lowered and normalized.count(",") >= 2:
            return False
        if normalized.count(":") >= 4:
            return False
        return True

    def _focus_query_tokens(self, query_tokens: set[str]) -> set[str]:
        generic_tokens = {
            "cub",
            "turism",
            "viaj",
            "destin",
            "vacacion",
            "hotel",
            "ciud",
            "lug",
        }
        focused = {token for token in query_tokens if token not in generic_tokens}
        return focused or query_tokens

    def _has_local_vector_model(self) -> bool:
        model_name = str(self.vector_db.model_name or "").strip()
        if not model_name:
            logger.warning("No se encontro modelo vectorial local disponible")
            return False

        model_path = Path(model_name)
        if model_path.exists():
            return True

        model_slug = model_name.replace("/", "--")
        cache_candidates = [
            Path.home() / ".cache" / "torch" / "sentence_transformers" / model_name,
            Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_slug}",
            Path.home() / ".cache" / "huggingface" / "hub" / f"models--sentence-transformers--{model_slug}",
        ]
        for candidate in cache_candidates:
            if candidate.exists():
                logger.info("Modelo vectorial encontrado en cache: %s", candidate)
                return True

        # Fallback robusto para variaciones de mayúsculas/minúsculas en el nombre del modelo.
        hub_dir = Path.home() / ".cache" / "huggingface" / "hub"
        if hub_dir.exists():
            target_suffix = model_slug.casefold()
            for candidate in hub_dir.glob("models--*"):
                name = candidate.name.casefold()
                if name.endswith(target_suffix):
                    logger.info("Modelo vectorial encontrado (fallback): %s", candidate)
                    return True

        logger.warning("Modelo vectorial %s no encontrado en cache", model_name)
        return False
