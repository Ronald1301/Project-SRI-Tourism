import logging

from src.RAG.rag_pipeline import RAGPipeline, RetrievedDocument
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
        logger.info("RAGSearchService listo")

    def search(
        self,
        query: str,
        search_mode: str = "lsi",
        top_k: int = 5,
        include_explanations: bool = False,
    ) -> tuple[list[RetrievedDocument], str]:
        logger.info("service.search | query=\"%s\" | mode=%s | top_k=%d", query, search_mode, top_k)
        rag_result = self.rag_pipeline.answer_query(
            query=query,
            top_k=top_k,
            search_mode=search_mode,
            include_explanations=include_explanations,
        )
        logger.info("service.search completo | %d documentos", len(rag_result.documents))
        return rag_result.documents, rag_result.answer


_rag_service: RAGSearchService | None = None


def get_rag_service() -> RAGSearchService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGSearchService()
    return _rag_service
