import logging

from src.RAG.rag_pipeline import RAGPipeline, RetrievedDocument
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
        self.rag_pipeline = RAGPipeline.from_preset(semantic_searcher=self.searcher)
        self.query_expander = QueryExpander(self.searcher)
        logger.info("RAGSearchService listo")

    def search(
        self,
        query: str,
        search_mode: str = "lsi",
        top_k: int = 5,
        include_explanations: bool = False,
    ) -> tuple[list[RetrievedDocument], str, dict]:
        logger.info("service.search | query=\"%s\" | mode=%s | top_k=%d", query, search_mode, top_k)
        preview_k = max(int(top_k), 3)
        raw_preview = self.rag_pipeline.retrieve(
            query,
            top_k=preview_k,
            search_mode=search_mode,
            include_explanations=include_explanations,
        )
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
                search_mode=search_mode,
                include_explanations=include_explanations,
            )
            if self._should_use_expanded_results(raw_preview, expanded_preview):
                selected_query = expansion.expanded_query
                expansion.selected_strategy = "expanded"
            else:
                expansion.selected_strategy = "raw_fallback"
            expansion.selected_query = selected_query

        rag_result = self.rag_pipeline.answer_query(
            query=selected_query,
            top_k=top_k,
            search_mode=search_mode,
            include_explanations=include_explanations,
            answer_query_text=query,
        )
        logger.info("service.search completo | %d documentos", len(rag_result.documents))
        return rag_result.documents, rag_result.answer, expansion.to_dict()

    def add_explicit_feedback(
        self,
        *,
        query: str,
        doc_id: str,
        relevance: int,
        expanded_query: str | None = None,
        search_mode: str | None = None,
    ) -> dict:
        return self.query_expander.record_explicit_feedback(
            query=query,
            doc_id=doc_id,
            relevance=relevance,
            expanded_query=expanded_query,
            search_mode=search_mode,
        )

    def add_implicit_feedback(
        self,
        *,
        query: str,
        doc_id: str,
        event: str,
        search_mode: str | None = None,
    ) -> tuple[dict, bool]:
        return self.query_expander.record_implicit_feedback(
            query=query,
            doc_id=doc_id,
            event=event,
            search_mode=search_mode,
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

        if not has_overlap and expanded_top_score < (raw_top_score * 0.65):
            logger.info(
                "Expansion descartada: score expandido debil | raw=%.4f expanded=%.4f",
                raw_top_score,
                expanded_top_score,
            )
            return False
        return True


_rag_service: RAGSearchService | None = None


def get_rag_service() -> RAGSearchService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGSearchService()
    return _rag_service
