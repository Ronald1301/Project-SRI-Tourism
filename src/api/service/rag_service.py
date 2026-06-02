import logging
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.RAG.rag_pipeline import RAGPipeline, RetrievedDocument
from src.api.config import DEFAULT_DOMAIN_LLM_MODEL, DEFAULT_OLLAMA_TIMEOUT_SECONDS
from src.api.config import DEFAULT_DOMAIN_DETECTION_CONFIG, DEFAULT_DOMAIN_SYNONYMS, DEFAULT_QUERY_EXPANSION_CONFIG
from src.retrieval.domain_detector import DomainDetector
from src.retrieval.query_expansion import QueryExpander
from src.retrieval.search import SemanticSearcher
from src.api.config import (
    DEFAULT_TFIDF_MATRIX,
    DEFAULT_TFIDF_VOCAB,
    DEFAULT_TFIDF_META,
    DEFAULT_LSI_MODEL,
    DEFAULT_LSI_VECTORS,
    DEFAULT_LSI_META,
    DEFAULT_DOCUMENTS,
)

logger = logging.getLogger("src.api.service.rag_service")


class RAGSearchService:
    def __init__(self):
        logger.info("Inicializando RAGSearchService...")
        self.searcher = SemanticSearcher(
            tfidf_matrix_path=DEFAULT_TFIDF_MATRIX,
            tfidf_vocab_path=DEFAULT_TFIDF_VOCAB,
            tfidf_meta_path=DEFAULT_TFIDF_META,
            lsi_model_path=DEFAULT_LSI_MODEL,
            lsi_vectors_path=DEFAULT_LSI_VECTORS,
            lsi_meta_path=DEFAULT_LSI_META,
            documents_path=DEFAULT_DOCUMENTS,
        )
        self.domain_detector = DomainDetector(
            tfidf_index=self.searcher.tfidf_index,
            lsi_model=self.searcher.lsi_model,
            domain_keywords=self._build_domain_keywords(),
            llm_client=self._build_ollama_client(),
            llm_model=DEFAULT_DOMAIN_LLM_MODEL,
            language="spanish",
            thresholds=self._load_domain_thresholds(DEFAULT_DOMAIN_DETECTION_CONFIG),
        )
        self.rag_pipeline = RAGPipeline.from_preset(semantic_searcher=self.searcher)
        self.query_expander = QueryExpander(
            self.searcher,
            config_path=DEFAULT_QUERY_EXPANSION_CONFIG,
            synonyms_path=DEFAULT_DOMAIN_SYNONYMS,
        )
        logger.info("RAGSearchService listo")

    def search(
        self,
        query: str,
        top_k: int = 5,
        include_explanations: bool = False,
        generate_answer: bool = True,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[list[RetrievedDocument], str | None, dict, dict[str, Any], list[dict[str, Any]]]:
        logger.info("service.search | query=\"%s\" | hybrid | top_k=%d", query, top_k)
        events: list[dict[str, Any]] = [
            self._stage_event("checking_domain", "Analizando consulta...", progress=0.1)
        ]
        if event_sink is not None:
            event_sink(dict(events[-1]))
        domain_info = self._detect_domain(query)
        if domain_info.get("status") == "OUT_OF_DOMAIN":
            logger.info("Consulta fuera de dominio; se omite retrieval | query=\"%s\"", query)
            out_event = self._stage_event(
                "out_of_domain",
                "La consulta no pertenece al dominio del sistema.",
                progress=1.0,
                data={"domain": domain_info},
            )
            events.append(out_event)
            if event_sink is not None:
                event_sink(dict(out_event))
            done_event = self._stage_event("done", "Proceso finalizado.", progress=1.0)
            events.append(done_event)
            if event_sink is not None:
                event_sink(dict(done_event))
            return [], None, {}, domain_info, events

        local_event = self._stage_event("searching_local", "Buscando en base de datos local...", progress=0.35)
        events.append(local_event)
        if event_sink is not None:
            event_sink(dict(local_event))
        preview_k = max(int(top_k), 3)
        raw_preview = self.rag_pipeline.retrieve(
            query,
            top_k=preview_k,
            include_explanations=include_explanations,
        )
        raw_preview = self._apply_feedback_bias(query, raw_preview)
        if not raw_preview:
            web_event = self._stage_event("searching_web", "Buscando en la web...", progress=0.7)
            events.append(web_event)
            if event_sink is not None:
                event_sink(dict(web_event))
        expansion = self.query_expander.expand_query(
            query,
            method="hybrid",
            top_documents=raw_preview[:3],
        )
        selected_query = query

        if expansion.applied:
            expanded_preview = self.rag_pipeline.retrieve(
                expansion.expanded_query,
                top_k=preview_k,
                include_explanations=include_explanations,
            )
            expanded_preview = self._apply_feedback_bias(query, expanded_preview)
            if self._should_use_expanded_results(raw_preview, expanded_preview):
                selected_query = expansion.expanded_query
                expansion.selected_strategy = "expanded"
            else:
                expansion.selected_strategy = "raw_fallback"
            expansion.selected_query = selected_query

        rag_result = self.rag_pipeline.answer_query(
            query=selected_query,
            top_k=top_k,
            include_explanations=include_explanations,
            web_query=query,
            event_sink=event_sink,
            document_ranker=self._apply_feedback_bias,
            generate_answer=generate_answer,
        )
        done_event = self._stage_event("done", "Busqueda completada.", progress=1.0)
        events.append(done_event)
        if event_sink is not None:
            event_sink(dict(done_event))
        logger.info("service.search completo | %d documentos", len(rag_result.documents))
        return rag_result.documents, rag_result.answer, expansion.to_dict(), domain_info, events

    def add_explicit_feedback(
        self,
        *,
        query: str,
        doc_id: str,
        relevance: int,
        expanded_query: str | None = None,
    ) -> dict:
        return self.query_expander.record_explicit_feedback(
            query=query,
            doc_id=doc_id,
            relevance=relevance,
            expanded_query=expanded_query,
        )

    def add_implicit_feedback(
        self,
        *,
        query: str,
        doc_id: str,
        event: str,
    ) -> tuple[dict, bool]:
        return self.query_expander.record_implicit_feedback(
            query=query,
            doc_id=doc_id,
            event=event,
        )

    def _should_use_expanded_results(
        self,
        raw_documents: list[RetrievedDocument],
        expanded_documents: list[RetrievedDocument],
    ) -> bool:
        if not expanded_documents:
            return False
        if not raw_documents:
            return True

        raw_top_score = max(float(doc.score) for doc in raw_documents[:3])
        expanded_top_score = max(float(doc.score) for doc in expanded_documents[:3])
        raw_ids = {doc.doc_id for doc in raw_documents[:3]}
        expanded_ids = {doc.doc_id for doc in expanded_documents[:3]}
        has_overlap = bool(raw_ids & expanded_ids)
        acceptance_threshold = float(getattr(self.query_expander.settings, "acceptance_threshold", 0.65))

        if not has_overlap and expanded_top_score < (raw_top_score * acceptance_threshold):
            logger.info(
                "Expansion descartada: score expandido debil | raw=%.4f expanded=%.4f",
                raw_top_score,
                expanded_top_score,
            )
            return False
        return True

    def _detect_domain(self, query: str) -> dict[str, Any]:
        explanation = self.domain_detector.is_in_domain(query)
        status = str(explanation.get("decision") or "OUT_OF_DOMAIN")
        message: str | None = None

        if status == "OUT_OF_DOMAIN":
            message = (
                "Tu consulta parece estar fuera del dominio de turismo en Cuba. "
                "Prueba con destinos, hoteles, playas, excursiones o lugares de la isla."
            )
        elif status == "UNCERTAIN":
            message = (
                "La consulta quedo incierta para el detector de dominio; se usara el flujo "
                "normal solo si el clasificador local la acepta."
            )

        return {
            "query": query,
            "status": status,
            "fast_decision": explanation.get("fast_decision"),
            "used_llm": bool(explanation.get("used_llm", False)),
            "llm_result": explanation.get("llm_result"),
            "message": message,
            "model": DEFAULT_DOMAIN_LLM_MODEL,
            "confidence": explanation.get("confidence"),
            "scores": explanation.get("scores", {}),
            "features": explanation.get("features", {}),
        }

    def _build_domain_keywords(self) -> list[str]:
        keywords: set[str] = {
            "turismo",
            "cuba",
            "cubana",
            "cubano",
            "varadero",
            "habana",
            "la habana",
            "trinidad",
            "matanzas",
            "pinar del rio",
            "baracoa",
            "santiago de cuba",
            "cayo guillermo",
            "cayo coco",
            "pico turquino",
            "turquino",
            "sierra maestra",
            "topes de collantes",
            "playa",
            "playas",
            "hotel",
            "hoteles",
            "alojamiento",
            "resort",
            "excursion",
            "excursiones",
            "viaje",
            "viajes",
            "destino",
            "destinos",
            "bar",
            "discoteca"
        }

        for document in self.searcher.documents_by_id.values():
            for field in ("title", "entity_name", "location"):
                value = str(document.get(field) or "").strip()
                if not value:
                    continue
                keywords.update(self.searcher.pipeline.process_text(value))

        cleaned = sorted({keyword for keyword in keywords if keyword})
        logger.info("Domain keywords preparadas | total=%d", len(cleaned))
        return cleaned

    def _build_ollama_client(self):
        try:
            import ollama
        except ImportError:
            logger.warning("Ollama no esta instalado; el fallback LLM quedara desactivado")
            return None

        try:
            return ollama.Client(timeout=DEFAULT_OLLAMA_TIMEOUT_SECONDS)
        except Exception as exc:  # pragma: no cover - depende del entorno local
            logger.warning("No se pudo inicializar Ollama; el fallback LLM quedara desactivado: %s", exc)
            return None

    def _load_domain_thresholds(self, path: str | Path | None) -> dict[str, Any] | None:
        if not path:
            return None
        threshold_path = Path(path)
        if not threshold_path.exists():
            return None
        try:
            payload = json.loads(threshold_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("No se pudo leer la configuracion de domain detection (%s): %s", threshold_path, exc)
            return None
        return payload if isinstance(payload, dict) else None

    def _apply_feedback_bias(self, query: str, documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
        if not documents:
            return documents

        try:
            feedback = self.query_expander.feedback_store.query_feedback(query)
        except Exception as exc:
            logger.warning("No se pudo leer feedback para sesgar el ranking: %s", exc)
            return documents

        explicit_rows = feedback.get("explicit", []) or []
        implicit_rows = feedback.get("implicit", []) or []
        negative_doc_ids = {
            str(item.get("doc_id") or "").strip()
            for item in explicit_rows
            if int(item.get("relevance") or 0) <= 0 and str(item.get("doc_id") or "").strip()
        }
        positive_bias: dict[str, float] = {}

        for item in explicit_rows:
            doc_id = str(item.get("doc_id") or "").strip()
            if not doc_id:
                continue
            relevance = int(item.get("relevance") or 0)
            if relevance > 0:
                positive_bias[doc_id] = positive_bias.get(doc_id, 0.0) + (0.25 * float(relevance))

        for item in implicit_rows:
            doc_id = str(item.get("doc_id") or "").strip()
            if not doc_id:
                continue
            weight = float(item.get("weight") or 0.0)
            positive_bias[doc_id] = positive_bias.get(doc_id, 0.0) + (0.15 * weight)

        biased_documents: list[RetrievedDocument] = []
        for doc in documents:
            if doc.doc_id in negative_doc_ids:
                continue
            adjusted_score = float(doc.score) + positive_bias.get(doc.doc_id, 0.0)
            biased_documents.append(
                RetrievedDocument(
                    citation_id=doc.citation_id,
                    doc_id=doc.doc_id,
                    title=doc.title,
                    url=doc.url,
                    score=adjusted_score,
                    summary=doc.summary,
                    content_text=doc.content_text,
                    metadata=doc.metadata,
                )
            )

        biased_documents.sort(key=lambda item: item.score, reverse=True)
        return biased_documents

    def _stage_event(
        self,
        stage: str,
        message: str,
        *,
        progress: float | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "event_type": "processing",
            "stage": stage,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "progress": progress,
            "data": data,
        }


_rag_service: RAGSearchService | None = None


def get_rag_service() -> RAGSearchService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGSearchService()
    return _rag_service
