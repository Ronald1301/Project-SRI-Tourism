import unittest
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from src.evaluation.experiment_runner import evaluate_query, summarize
from src.evaluation.systems import search_with_expansion
from src.retrieval.domain_detector import DomainDetector


class AuditIntegrationTests(unittest.TestCase):
    def _build_detector(self) -> DomainDetector:
        class FakeTfidfIndex:
            def __init__(self) -> None:
                self.matrix = sparse.csr_matrix(np.array([[1.0], [0.0]], dtype=np.float32))

            def vectorize_query(self, tokens):
                token_set = set(tokens)
                value = 1.0 if token_set & {"hotel", "hoteles", "varadero", "playa", "playas"} else 0.0
                return sparse.csr_matrix(np.array([[value]], dtype=np.float32))

        class FakeLsiModel:
            def __init__(self) -> None:
                self.doc_vectors = np.array([[1.0], [0.0]], dtype=np.float32)

            def transform_query(self, query_vector):
                value = float(query_vector.toarray().ravel()[0])
                return np.array([value], dtype=np.float32)

        return DomainDetector(
            tfidf_index=FakeTfidfIndex(),
            lsi_model=FakeLsiModel(),
            domain_keywords=["hotel", "varadero", "playa"],
            llm_client=None,
            language="spanish",
        )

    def test_domain_detector_blocks_out_of_domain_query(self):
        detector = self._build_detector()
        result = detector.is_in_domain("precio del dolar")

        self.assertEqual(result["decision"], "OUT_OF_DOMAIN")
        self.assertFalse(result["used_llm"])
        self.assertIn("confidence", result)
        self.assertIn("scores", result)

    def test_domain_detector_accepts_tourism_query(self):
        detector = self._build_detector()
        result = detector.is_in_domain("hoteles en varadero")

        self.assertEqual(result["decision"], "IN_DOMAIN")
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertIn("heuristic", result["scores"])

    def test_query_expansion_changes_selected_query(self):
        class FakeSearcher:
            def search(self, query: str, top_k: int):
                return [{"doc_id": query, "score": 1.0}]

        fake_searcher = FakeSearcher()
        fake_expander = SimpleNamespace(
            expand_query=lambda query, top_documents=None: SimpleNamespace(
                expanded_query=f"{query} expandida",
                applied=True,
            )
        )

        results = search_with_expansion(fake_searcher, fake_expander, "playas", 3)
        self.assertEqual(results[0]["doc_id"], "playas expandida")

    def test_evaluation_reports_required_metrics(self):
        qrels = {
            "queries": [
                {
                    "query_id": "q1",
                    "query": "playas en cuba",
                    "relevant_doc_ids": ["d1", "d2"],
                }
            ]
        }

        class FakeSystem:
            label = "fake baseline"

            @staticmethod
            def search(query: str, top_k: int):
                return [
                    {"doc_id": "d1", "score": 1.0},
                    {"doc_id": "d3", "score": 0.9},
                    {"doc_id": "d2", "score": 0.8},
                    {"doc_id": "d4", "score": 0.7},
                    {"doc_id": "d5", "score": 0.6},
                    {"doc_id": "d6", "score": 0.5},
                    {"doc_id": "d7", "score": 0.4},
                    {"doc_id": "d8", "score": 0.3},
                    {"doc_id": "d9", "score": 0.2},
                    {"doc_id": "d10", "score": 0.1},
                ]

        query_judgment = SimpleNamespace(
            query_id="q1",
            query="playas en cuba",
            relevant_ids={"d1", "d2"},
            judgments={"d1": 3, "d2": 1},
        )
        retrieved_ids = ["d1", "d3", "d2", "d4", "d5", "d6", "d7", "d8", "d9", "d10"]

        row = evaluate_query(query_judgment, retrieved_ids, top_k=5)
        summary, statistics = summarize([row], top_k=5)

        self.assertEqual(row["precision_at_3"], 2 / 3)
        self.assertIn("precision_at_5", row)
        self.assertIn("recall_at_10", row)
        self.assertIn("ndcg_at_5", row)
        self.assertIn("mrr", row)
        self.assertIn("precision_at_3", summary)
        self.assertIn("precision_at_3", statistics)
        self.assertIn("ci95", statistics["precision_at_3"])


if __name__ == "__main__":
    unittest.main()
