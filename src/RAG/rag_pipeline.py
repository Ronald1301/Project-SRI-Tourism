from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np # type: ignore
from sklearn.metrics.pairwise import cosine_similarity as sparse_cosine_similarity

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
    """Itera un archivo JSONL y devuelve cada linea como diccionario.

    Args:
        path: Ruta del archivo JSONL a leer.

    Yields:
        dict: Cada registro JSON decodificado.
    """
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


@dataclass(frozen=True)
class RetrievedDocument:
    """Documento recuperado para construir el prompt y la respuesta final."""
    citation_id: int
    doc_id: str
    title: str
    url: str
    score: float
    summary: str
    content_text: str
    metadata: dict
    vector_score: float | None = None
    lsi_score: float | None = None


@dataclass(frozen=True)
class CandidatePassage:
    """Fragmento candidato usado para seleccionar evidencia relevante."""
    citation_id: int
    title: str
    text: str
    score: float


@dataclass(frozen=True)
class RAGResult:
    """Resultado completo de una consulta RAG."""
    query: str
    prompt: str
    answer: str
    documents: list[RetrievedDocument]
    events: list[dict[str, Any]] = field(default_factory=list)


class DocumentRepository:
    """Repositorio en memoria para acceder a los documentos base del sistema."""

    def __init__(self, documents: dict[str, dict]) -> None:
        """Inicializa el repositorio de documentos.

        Args:
            documents: Mapa `doc_id -> documento` cargado en memoria.

        Returns:
            None
        """
        self.documents = documents

    @classmethod
    def from_jsonl(cls, path: Path) -> "DocumentRepository":
        """Construye el repositorio a partir de un archivo JSONL.

        Args:
            path: Ruta del archivo JSONL con documentos persistidos.

        Returns:
            DocumentRepository: Repositorio en memoria indexado por `doc_id`.
        """
        documents: dict[str, dict] = {}
        for doc in iter_jsonl(path):
            doc_id = str(doc.get("doc_id") or "").strip()
            if not doc_id:
                continue
            documents[doc_id] = doc
        return cls(documents)

    def get(self, doc_id: str) -> dict | None:
        """Obtiene un documento por su identificador.

        Args:
            doc_id: Identificador del documento.

        Returns:
            dict | None: Documento asociado o `None` si no existe.
        """
        return self.documents.get(doc_id)


class RAGPipeline:
    """Orquesta la recuperacion, la ampliacion web y la generacion RAG."""

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
        """Inicializa el pipeline RAG completo.

        Args:
            vector_db: Base vectorial ya cargada.
            repository: Repositorio de documentos fuente.
            language: Idioma de trabajo del preprocesamiento.
            semantic_searcher: Recuperador semantico/LSI opcional.
            insufficiency_policy: Politica para decidir si falta evidencia.
            web_search_client: Cliente de busqueda web para ampliacion dinamica.
            web_search_output_path: Ruta donde se persisten los documentos web.

        Returns:
            None
        """
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
        """Construye el pipeline RAG a partir de la configuracion por defecto.

        Args:
            output_dir: Directorio donde esta persistida la base vectorial.
            language: Idioma de preprocesamiento.
            semantic_searcher: Recuperador semantico/LSI opcional.
            insufficiency_policy: Politica de suficiencia de evidencia opcional.
            web_search_client: Cliente de busqueda web opcional.
            web_search_output_path: Ruta de persistencia para documentos web.

        Returns:
            RAGPipeline: Pipeline listo para responder consultas.
        """
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
        include_explanations: bool = False,
        web_query: str | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        document_ranker: Callable[[str, list[RetrievedDocument]], list[RetrievedDocument]] | None = None,
        generate_answer: bool = True,
    ) -> RAGResult:
        """Responde una consulta ejecutando recuperacion hibrida y generacion.

        Args:
            query: Consulta del usuario.
            top_k: Numero maximo de documentos finales a recuperar.
            include_explanations: Si se deben incluir explicaciones en la parte LSI.
            web_query: Consulta original que se debe usar para la ampliacion web.
            generate_answer: Si es `True`, se usa Ollama para generar la respuesta.
                Si es `False`, se devuelve una respuesta local construida solo con la evidencia recuperada.

        Returns:
            RAGResult: Resultado final con prompt, respuesta y documentos usados.
        """
        logger.info("answer_query | query=\"%s\" | hybrid | top_k=%d", query, top_k)
        user_query = query.strip()
        web_query_text = (web_query or query).strip()
        events: list[dict[str, Any]] = []
        self._emit_event(
            events,
            "request_received",
            "Consulta recibida por el backend.",
            stage="request",
            progress=0.0,
            data={"top_k": int(top_k)},
            event_sink=event_sink,
        )
        self._emit_event(
            events,
            "retrieval_started",
            "Iniciando recuperacion local.",
            stage="retrieval",
            progress=0.15,
            data={"query": query},
            event_sink=event_sink,
        )
        local_documents = self.retrieve(
            query,
            top_k=top_k,
            include_explanations=include_explanations,
        )
        if document_ranker is not None:
            try:
                ranked_documents = document_ranker(user_query, list(local_documents))
                if isinstance(ranked_documents, list):
                    local_documents = ranked_documents
            except Exception as exc:
                logger.warning("document_ranker fallo; se conservaran los documentos sin sesgo: %s", exc)
        logger.info("Recuperacion local: %d documentos", len(local_documents))
        self._emit_event(
            events,
            "retrieval_completed",
            "Recuperacion local completada.",
            stage="retrieval",
            progress=0.45,
            data={
                "documents": len(local_documents),
                "top_score": float(local_documents[0].score) if local_documents else 0.0,
            },
            event_sink=event_sink,
        )

        if self._is_retrieval_insufficient(local_documents):
            logger.info("Recuperacion local INSUFICIENTE — activando busqueda web")
            self._emit_event(
                events,
                "insufficiency_detected",
                "La evidencia local fue insuficiente; se activara busqueda web.",
                stage="analysis",
                progress=0.55,
                data={"documents": len(local_documents)},
                event_sink=event_sink,
            )
            self._emit_event(
                events,
                "web_search_started",
                "Iniciando ampliacion web.",
                stage="web_search",
                progress=0.65,
                data={"query": web_query_text},
                event_sink=event_sink,
            )
            web_documents = self._retrieve_and_index_web_context(web_query_text, top_k=top_k)
            if web_documents:
                documents = self._merge_retrieved_documents(local_documents, web_documents, top_k=top_k)
                logger.info(
                    "Usando merge local+web: %d locales + %d web -> %d documentos",
                    len(local_documents),
                    len(web_documents),
                    len(documents),
                )
            else:
                documents = local_documents
                logger.info("Web search no retorno documentos, usando solo locales")
        else:
            documents = local_documents
            logger.info("Recuperacion local suficiente")

        if document_ranker is not None:
            try:
                ranked_documents = document_ranker(user_query, list(documents))
                if isinstance(ranked_documents, list):
                    documents = ranked_documents
            except Exception as exc:
                logger.warning("document_ranker fallo sobre documentos finales; se conserva el orden actual: %s", exc)

        prompt = self.answer_generator.build_prompt(user_query, documents)
        self._emit_event(
            events,
            "generation_started",
            "Iniciando generacion final.",
            stage="generation",
            progress=0.85,
            data={"documents": len(documents), "mode": "llm" if generate_answer else "local"},
            event_sink=event_sink,
        )
        if generate_answer:
            logger.info("Generando respuesta con LLM...")
            answer = self.answer_generator.generate(user_query, documents, prompt=prompt)
        else:
            logger.info("Generacion con LLM desactivada; usando respuesta local")
            self._emit_event(
                events,
                "generation_skipped",
                "La generacion con LLM fue desactivada; se devolvio una respuesta local.",
                stage="generation",
                progress=0.9,
                data={"documents": len(documents), "mode": "local"},
                event_sink=event_sink,
            )
            answer = self.generate_answer(user_query, documents)
        logger.info("Respuesta generada | answer_len=%d", len(answer))
        self._emit_event(
            events,
            "request_completed",
            "Consulta completada.",
            stage="done",
            progress=1.0,
            data={"documents": len(documents), "answer_length": len(answer), "generation_mode": "llm" if generate_answer else "local"},
            event_sink=event_sink,
        )
        return RAGResult(
            query=user_query,
            prompt=prompt,
            answer=answer,
            documents=documents,
            events=events,
        )

    def answer_with_lsi(
        self,
        query: str,
        lsi_results: list[dict],
        top_k: int = 4,
    ) -> RAGResult:
        """Construye una respuesta usando resultados LSI ya obtenidos.

        Args:
            query: Consulta original del usuario.
            lsi_results: Resultados recuperados por el componente LSI.
            top_k: Numero maximo de documentos a usar.

        Returns:
            RAGResult: Resultado listo para la capa de presentacion.
        """
        documents = self._convert_lsi_results(lsi_results[:top_k])
        prompt = self.answer_generator.build_prompt(query, documents)
        answer = self.answer_generator.generate(query, documents, prompt=prompt)
        return RAGResult(
            query=query,
            prompt=prompt,
            answer=answer,
            documents=documents,
            events=[],
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        include_explanations: bool = False,
    ) -> list[RetrievedDocument]:
        """Recupera documentos combinando la señal vectorial y la LSI.

        Args:
            query: Consulta del usuario.
            top_k: Numero de documentos finales a devolver.
            include_explanations: Si la recuperacion LSI debe incluir explicaciones.

        Returns:
            list[RetrievedDocument]: Lista ordenada de documentos recuperados.
        """
        logger.info("retrieve | hybrid | top_k=%d", top_k)
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
                    vector_score=float(item.get("vector_score")) if item.get("vector_score") is not None else None,
                    lsi_score=float(item.get("lsi_score")) if item.get("lsi_score") is not None else None,
                    summary=summary,
                    content_text=content_text,
                    metadata=merged,
                )
            )

        return documents

    def _search_vectorial_raw(self, query: str, top_k: int) -> list[dict]:
        """Ejecuta la recuperacion vectorial primaria o su respaldo TF-IDF.

        Args:
            query: Consulta del usuario.
            top_k: Numero maximo de resultados a recuperar.

        Returns:
            list[dict]: Resultados crudos de recuperacion.
        """
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
        """Ejecuta la recuperacion LSI y normaliza su salida.

        Args:
            query: Consulta del usuario.
            top_k: Numero maximo de resultados.
            include_explanations: Indica si se incluyen explicaciones en el payload.

        Returns:
            list[dict]: Resultados LSI normalizados.
        """
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
        """Fusiona varias listas de resultados con Reciprocal Rank Fusion.

        Args:
            result_lists: Listas ordenadas de resultados parciales.
            top_k: Numero maximo de resultados fusionados a devolver.
            rrf_k: Constante de suavizado de RRF.

        Returns:
            list[dict]: Lista fusionada y reordenada por score acumulado.
        """
        if top_k <= 0:
            return []
        accumulator: dict[str, dict[str, Any]] = {}

        for result_list in result_lists:
            for rank, item in enumerate(result_list, start=1):
                doc_id = str(item.get("doc_id") or "").strip()
                if not doc_id:
                    continue

                contribution = 1.0 / (float(rrf_k) + float(rank))
                source_score = float(item.get("score", 0.0))
                if doc_id not in accumulator:
                    base_payload = dict(item)
                    base_payload["rrf_score"] = 0.0
                    base_payload["semantic_score"] = source_score
                    accumulator[doc_id] = base_payload

                accumulator[doc_id]["rrf_score"] = float(accumulator[doc_id]["rrf_score"]) + contribution
                accumulator[doc_id]["semantic_score"] = min(
                    float(accumulator[doc_id].get("semantic_score", source_score)),
                    source_score,
                )
                accumulator[doc_id]["score"] = float(accumulator[doc_id]["semantic_score"])

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
            key=lambda payload: float(payload.get("rrf_score", 0.0)),
            reverse=True,
        )
        return merged[:top_k]

    def _emit_event(
        self,
        events: list[dict[str, Any]],
        event_type: str,
        message: str,
        *,
        stage: str,
        progress: float | None = None,
        data: dict[str, Any] | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Anade un evento estructurado al flujo de trazas del RAG.

        Args:
            events: Lista mutable donde se acumulan los eventos.
            event_type: Nombre tecnico del evento.
            message: Mensaje breve para la interfaz.
            stage: Fase funcional del proceso.
            progress: Progreso aproximado en el rango `[0, 1]`.
            data: Datos adicionales opcionales.

        Returns:
            None
        """
        payload: dict[str, Any] = {
            "event_type": event_type,
            "message": message,
            "stage": stage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if progress is not None:
            payload["progress"] = max(0.0, min(float(progress), 1.0))
        if data:
            payload["data"] = data
        events.append(payload)
        if event_sink is not None:
            event_sink(dict(payload))

    def _is_retrieval_insufficient(self, documents: list[RetrievedDocument]) -> bool:
        """Evalua si la evidencia recuperada es insuficiente para responder.

        Args:
            documents: Documentos ya recuperados por el pipeline.

        Returns:
            bool: `True` si la politica considera que falta evidencia.
        """
        scores = [doc.score for doc in documents]
        result = self.insufficiency_policy.is_insufficient(
            [{"doc_id": doc.doc_id, "score": float(doc.score)} for doc in documents]
        )
        if result and documents:
            logger.info("Recuperacion insuficiente: scores=%s", [f"{s:.4f}" for s in scores])
        return result

    def _retrieve_and_index_web_context(self, query: str, *, top_k: int) -> list[RetrievedDocument]:
        """Busca en la web, persiste e indexa los nuevos documentos recuperados.

        Args:
            query: Consulta del usuario.
            top_k: Numero de documentos finales que se desean alimentar al RAG.

        Returns:
            list[RetrievedDocument]: Documentos web normalizados y listos para generar.
        """
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
        logger.info("Web context listo: %d documentos convertidos", len(normalized_docs))
        return self._convert_web_docs_to_retrieved(normalized_docs)

    def _merge_retrieved_documents(
        self,
        local_documents: list[RetrievedDocument],
        web_documents: list[RetrievedDocument],
        *,
        top_k: int,
    ) -> list[RetrievedDocument]:
        """Fusiona resultados locales y web en una unica lista ordenada.

        Args:
            local_documents: Documentos recuperados desde el corpus local.
            web_documents: Documentos recuperados desde la busqueda web.
            top_k: Numero maximo de resultados finales a devolver.

        Returns:
            list[RetrievedDocument]: Lista fusionada, deduplicada y ordenada por score.
        """
        merged_by_id: dict[str, RetrievedDocument] = {}

        def _ingest(document: RetrievedDocument) -> None:
            existing = merged_by_id.get(document.doc_id)
            if existing is None:
                merged_by_id[document.doc_id] = document
                return

            merged_metadata = dict(existing.metadata)
            merged_metadata.update(document.metadata or {})

            if document.score >= existing.score:
                merged_by_id[document.doc_id] = RetrievedDocument(
                    citation_id=existing.citation_id,
                    doc_id=document.doc_id,
                    title=document.title or existing.title,
                    url=document.url or existing.url,
                    score=document.score,
                    summary=document.summary or existing.summary,
                    content_text=document.content_text or existing.content_text,
                    metadata=merged_metadata,
                )
            else:
                merged_by_id[document.doc_id] = RetrievedDocument(
                    citation_id=existing.citation_id,
                    doc_id=existing.doc_id,
                    title=existing.title or document.title,
                    url=existing.url or document.url,
                    score=existing.score,
                    summary=existing.summary or document.summary,
                    content_text=existing.content_text or document.content_text,
                    metadata=merged_metadata,
                )

        for document in local_documents:
            _ingest(document)
        for document in web_documents:
            _ingest(document)

        merged_documents = sorted(
            merged_by_id.values(),
            key=lambda item: float(item.score),
            reverse=True,
        )
        return merged_documents[: max(int(top_k), 0)]

    def _normalize_web_documents(self, documents: list[dict[str, object]]) -> list[dict[str, object]]:
        """Normaliza documentos web para su persistencia e indexacion.

        Args:
            documents: Documentos recuperados desde la busqueda web.

        Returns:
            list[dict[str, object]]: Documentos filtrados y normalizados.
        """
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
        """Persiste documentos web en formato JSONL.

        Args:
            documents: Documentos web normalizados.

        Returns:
            None
        """
        try:
            output_path = self.web_search_output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_documents_to_jsonl(documents, output_path)
            logger.info("Documentos web persistidos en %s", output_path)
        except Exception as exc:
            logger.warning("Error persistiendo documentos web: %s", exc)

    def _index_web_documents(self, documents: list[dict[str, object]]) -> None:
        """Indexa documentos web nuevos en la base vectorial.

        Args:
            documents: Documentos web ya normalizados.

        Returns:
            None
        """
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
        """Incorpora documentos web al repositorio local en memoria.

        Args:
            documents: Documentos web normalizados.

        Returns:
            None
        """
        for item in documents:
            doc_id = str(item.get("doc_id") or "").strip()
            if not doc_id:
                continue
            self.repository.documents[doc_id] = dict(item)
            self._document_token_cache.pop(doc_id, None)
            self._title_token_cache.pop(doc_id, None)

    def _convert_web_docs_to_retrieved(self, web_docs: list[dict[str, object]]) -> list[RetrievedDocument]:
        """Convierte documentos web normalizados al formato de recuperacion.

        Args:
            web_docs: Documentos web normalizados.

        Returns:
            list[RetrievedDocument]: Documentos listos para el generador de respuesta.
        """
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
        """Convierte resultados LSI al formato uniforme de recuperacion.

        Args:
            lsi_results: Resultados devueltos por el recuperador LSI.

        Returns:
            list[RetrievedDocument]: Documentos uniformados para el generador.
        """
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
        """Ejecuta una busqueda de respaldo basada en TF-IDF.

        Args:
            query: Consulta del usuario.
            top_k: Numero maximo de resultados a devolver.

        Returns:
            list[dict]: Resultados rerankeados por similitud TF-IDF.
        """
        index = self._get_fallback_index()
        query_tokens = self.preprocessing.process_text(query)
        if not query_tokens or index.matrix is None or index.matrix.size == 0:
            return []
        focus_tokens = self._focus_query_tokens(set(query_tokens))

        query_vector = index.vectorize_query(query_tokens)
        doc_matrix = index.matrix
        similarities = sparse_cosine_similarity(doc_matrix, query_vector).ravel()

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
        """Construye el prompt RAG usando los documentos recuperados.

        Args:
            query: Consulta del usuario.
            documents: Documentos recuperados y ordenados por relevancia.

        Returns:
            str: Prompt final para el generador de respuesta.
        """
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
        """Genera una respuesta a partir de la evidencia ya seleccionada.

        Args:
            query: Consulta del usuario.
            documents: Documentos recuperados que sirven como evidencia.

        Returns:
            str: Respuesta textual final.
        """
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
        """Selecciona los fragmentos mas utiles para responder la consulta.

        Args:
            query: Consulta del usuario.
            documents: Documentos recuperados.
            max_passages: Maximo de pasajes a devolver.

        Returns:
            list[CandidatePassage]: Fragmentos candidatos ordenados por relevancia.
        """
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
        """Obtiene los pasajes mas utiles de un documento concreto.

        Args:
            doc: Documento recuperado.
            query: Consulta del usuario.
            limit: Numero maximo de pasajes a devolver.

        Returns:
            list[CandidatePassage]: Pasajes del documento ordenados por score.
        """
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
        """Construye o reutiliza el indice TF-IDF de respaldo.

        Args:
            No recibe argumentos directos.

        Returns:
            TFIDFIndex: Indice TF-IDF ya construido.
        """
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
        """Genera segmentos textuales candidatos dentro de un documento.

        Args:
            doc: Documento recuperado.

        Yields:
            str: Segmentos textuales potencialmente utiles.
        """
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
        """Obtiene y cachea los tokens de un documento completo.

        Args:
            doc_id: Identificador del documento.
            title: Titulo del documento.
            summary: Resumen del documento.
            content_text: Contenido principal del documento.

        Returns:
            set[str]: Conjunto de tokens procesados.
        """
        if doc_id not in self._document_token_cache:
            combined = "\n".join(part for part in [title, summary, content_text] if part)
            self._document_token_cache[doc_id] = set(self.preprocessing.process_text(combined))
        return self._document_token_cache[doc_id]

    def _get_title_tokens(self, doc_id: str, title: str) -> set[str]:
        """Obtiene y cachea los tokens del titulo de un documento.

        Args:
            doc_id: Identificador del documento.
            title: Titulo del documento.

        Returns:
            set[str]: Conjunto de tokens del titulo.
        """
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
        """Calcula una puntuacion heuristica para un segmento textual.

        Args:
            query_tokens: Tokens de la consulta.
            segment_tokens: Tokens del segmento candidato.
            doc_score: Puntaje base del documento.
            segment_length: Longitud del segmento.
            title_focus_overlap: Superposicion con tokens importantes del titulo.

        Returns:
            float: Puntuacion compuesta del segmento.
        """
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
        """Genera un extracto de respaldo si no hay mejores pasajes.

        Args:
            doc: Documento recuperado.

        Returns:
            str: Texto corto representativo del documento.
        """
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
        """Anade una cita numerica a un fragmento de texto.

        Args:
            prefix: Prefijo narrativo.
            text: Texto base a citar.
            citation_id: Identificador numerico de la cita.

        Returns:
            str: Texto con la cita al final.
        """
        cleaned = self._normalize_whitespace(text).rstrip(".!?")
        return f"{prefix}{cleaned} [{citation_id}]."

    def _normalize_whitespace(self, text: str) -> str:
        """Normaliza espacios en blanco y signos decorativos.

        Args:
            text: Texto original.

        Returns:
            str: Texto normalizado.
        """
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        return text.strip("-• ")

    def _is_useful_segment(self, text: str, title: str) -> bool:
        """Determina si un segmento aporta evidencia util al RAG.

        Args:
            text: Segmento a evaluar.
            title: Titulo del documento asociado.

        Returns:
            bool: `True` si el segmento es util, `False` en caso contrario.
        """
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
        """Elimina tokens demasiado genericos para enfocar la consulta.

        Args:
            query_tokens: Tokens originales de la consulta.

        Returns:
            set[str]: Tokens filtrados o los originales si todos eran genericos.
        """
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
        """Verifica si el modelo vectorial local esta disponible en disco o cache.

        Args:
            No recibe argumentos directos.

        Returns:
            bool: `True` si el modelo esta disponible localmente.
        """
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
