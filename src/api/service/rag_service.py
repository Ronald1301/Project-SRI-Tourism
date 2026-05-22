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


class RAGSearchService:
    def __init__(self):
        self.searcher = SemanticSearcher(
            tfidf_matrix_path=DEFAULT_TFIDF_MATRIX,
            tfidf_vocab_path=DEFAULT_TFIDF_VOCAB,
            tfidf_meta_path=DEFAULT_TFIDF_META,
            lsi_model_path=DEFAULT_LSI_MODEL,
            lsi_vectors_path=DEFAULT_LSI_VECTORS,
            lsi_meta_path=DEFAULT_LSI_META,
            documents_path=DEFAULT_DOCUMENTS,
        )
        self.rag_pipeline = RAGPipeline.from_preset()

    def search(
        self,
        query: str,
        search_mode: str = "lsi",
        top_k: int = 5,
        include_explanations: bool = False,
    ) -> tuple[list[RetrievedDocument], str]:
        if search_mode == "lsi":
            raw_results = self.searcher.search(
                query,
                top_k=top_k,
                include_explanations=include_explanations,
            )
        else:
            raw_results = self.searcher.search_baseline(
                query,
                top_k=top_k,
                include_explanations=include_explanations,
            )

        lsi_results: list[dict[str, object]] = []
        for result in raw_results:
            payload: dict[str, object] = {
                "doc_id": result["doc_id"],
                "title": result["title"],
                "score": result["score"],
                "summary": result.get("snippet") or result.get("summary") or "",
                "url": result.get("url") or "",
            }
            explanation = result.get("explanation")
            if include_explanations and isinstance(explanation, dict):
                payload["explanation"] = explanation
            lsi_results.append(payload)

        rag_result = self.rag_pipeline.answer_with_lsi(query, lsi_results, top_k=top_k)
        return rag_result.documents, rag_result.answer


_rag_service: RAGSearchService | None = None


def get_rag_service() -> RAGSearchService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGSearchService()
    return _rag_service
