import unittest
from unittest.mock import patch

from src.retrieval.evaluate import (
    average_precision,
    evaluate_systems,
    f1_at_k,
    ndcg_at_k,
    precision_at_k,
    r_precision,
    recall_at_k,
    reciprocal_rank_at_k,
    render_markdown_report,
)


class EvaluationMetricTests(unittest.TestCase):
    def test_core_metrics_binary_relevance(self):
        retrieved = ["d1", "d2", "d3", "d4"]
        relevant = {"d2", "d4", "d5"}

        self.assertAlmostEqual(precision_at_k(retrieved, relevant, 3), 1 / 3)
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, 3), 1 / 3)
        self.assertAlmostEqual(f1_at_k(retrieved, relevant, 3), 1 / 3)
        self.assertAlmostEqual(reciprocal_rank_at_k(retrieved, relevant, 4), 1 / 2)
        self.assertAlmostEqual(r_precision(retrieved, relevant), 1 / 3)

    def test_average_precision(self):
        retrieved = ["d1", "d2", "d3", "d4"]
        relevant = {"d2", "d4"}

        expected = ((1 / 2) + (2 / 4)) / 2
        self.assertAlmostEqual(average_precision(retrieved, relevant), expected)

    def test_ndcg_uses_graded_relevance(self):
        retrieved = ["d2", "d1", "d3"]
        judgments = {"d1": 3, "d2": 2, "d3": 1}

        score = ndcg_at_k(retrieved, judgments, 3)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_report_marks_missing_systems_as_skipped(self):
        qrels = {
            "queries": [
                {
                    "query_id": "q1",
                    "query": "playas",
                    "relevant_doc_ids": ["d1"],
                }
            ]
        }

        with patch("src.retrieval.evaluate.missing_lsi_artifacts", return_value=["missing"]):
            report = evaluate_systems(qrels=qrels, top_k=3, systems="lsi_baseline", include_markdown=True)
        self.assertIn("lsi_baseline", report["systems"])
        self.assertEqual(report["systems"]["lsi_baseline"]["status"], "skipped")
        markdown = render_markdown_report(report)
        self.assertIn("Reporte de evaluacion IR", markdown)
        self.assertIn("Resumen por sistema", markdown)


if __name__ == "__main__":
    unittest.main()
