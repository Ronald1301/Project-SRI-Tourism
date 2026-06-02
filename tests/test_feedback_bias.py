import unittest
from types import SimpleNamespace

from src.RAG.rag_pipeline import RetrievedDocument
from src.api.service.rag_service import RAGSearchService


class FeedbackBiasTests(unittest.TestCase):
    def test_negative_feedback_removes_document_and_boosts_positive(self):
        service = RAGSearchService.__new__(RAGSearchService)
        service.query_expander = SimpleNamespace(
            feedback_store=SimpleNamespace(
                query_feedback=lambda query: {
                    "explicit": [
                        {"doc_id": "doc_bad", "relevance": 0},
                        {"doc_id": "doc_good", "relevance": 1},
                    ],
                    "implicit": [
                        {"doc_id": "doc_good", "weight": 0.5},
                    ],
                }
            )
        )

        docs = [
            RetrievedDocument(
                citation_id=1,
                doc_id="doc_bad",
                title="Malo",
                url="",
                score=0.9,
                summary="",
                content_text="",
                metadata={},
            ),
            RetrievedDocument(
                citation_id=2,
                doc_id="doc_mid",
                title="Medio",
                url="",
                score=0.8,
                summary="",
                content_text="",
                metadata={},
            ),
            RetrievedDocument(
                citation_id=3,
                doc_id="doc_good",
                title="Bueno",
                url="",
                score=0.7,
                summary="",
                content_text="",
                metadata={},
            ),
        ]

        ranked = service._apply_feedback_bias("consulta", docs)

        self.assertNotIn("doc_bad", [doc.doc_id for doc in ranked])
        self.assertEqual(ranked[0].doc_id, "doc_good")
        self.assertGreater(ranked[0].score, 0.7)


if __name__ == "__main__":
    unittest.main()
