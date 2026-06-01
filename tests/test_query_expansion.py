import json
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.retrieval.query_expansion import QueryExpander, STOPWORDS


def build_dummy_searcher() -> SimpleNamespace:
    return SimpleNamespace(
        tfidf_index=SimpleNamespace(matrix=None, vocabulary=None, doc_id_to_index={}),
        pipeline=SimpleNamespace(process_text=lambda text: str(text).split()),
        documents_by_id={
            "doc-1": {
                "doc_id": "doc-1",
                "title": "Casco historico de La Habana",
                "summary": "Patrimonio historico y cultural",
                "content_text": "casco historico habana vieja",
            }
        },
    )


class QueryExpansionTests(unittest.TestCase):
    def test_bidirectional_synonyms_and_ngrams_are_added(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "feedback.db"
            expander = QueryExpander(build_dummy_searcher(), feedback_db_path=db_path)

            result = expander.expand_query(
                "casco historico",
                top_documents=[
                    {
                        "doc_id": "doc-1",
                        "title": "Casco historico de La Habana",
                        "summary": "Patrimonio historico y cultural",
                        "content_text": "casco historico habana vieja",
                    }
                ],
            )

            self.assertTrue(result.applied)
            self.assertLessEqual(len(result.terms), 5)
            self.assertTrue(all(score >= 0.1 for score in result.term_scores.values()))
            self.assertTrue(any(term in {"centro_historico", "habana_vieja"} for term in result.terms))
            self.assertLessEqual(len(expander.domain_synonyms.get("casco_historico", [])), 2)
            self.assertEqual(
                [item.get("technique") for item in result.trace],
                ["pseudo_relevance", "rocchio", "synonyms", "cooccurrence", "feedback", "ngrams"],
            )
            self.assertTrue(any(item.get("technique") == "ngrams" for item in result.trace))

    def test_cache_returns_cached_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "feedback.db"
            expander = QueryExpander(build_dummy_searcher(), feedback_db_path=db_path)

            first = expander.expand_query("playas varadero")
            second = expander.expand_query("playas varadero")

            self.assertFalse(first.cached)
            self.assertTrue(second.cached)
            self.assertEqual(first.expanded_query, second.expanded_query)

    def test_score_floor_filters_weak_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "feedback.db"
            expander = QueryExpander(build_dummy_searcher(), feedback_db_path=db_path)
            result = expander.expand_query("playas varadero")

            self.assertTrue(all(score >= 0.1 for score in result.term_scores.values()))

    def test_expansion_terms_do_not_include_stopword_fragments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "feedback.db"
            expander = QueryExpander(build_dummy_searcher(), feedback_db_path=db_path)

            result = expander.expand_query(
                "habana",
                top_documents=[
                    {
                        "doc_id": "doc-1",
                        "title": "Casco historico de La Habana",
                        "summary": "Patrimonio historico y cultural",
                        "content_text": "casco historico habana vieja",
                    }
                ],
            )

            self.assertTrue(result.applied)
            self.assertTrue(result.expanded_query.startswith("habana"))
            self.assertTrue(
                all(
                    all(part not in STOPWORDS and len(part) >= 3 for part in term.split("_"))
                    for term in result.terms
                )
            )
            self.assertNotIn("la_habana", result.terms)

    def test_original_query_text_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "feedback.db"
            expander = QueryExpander(build_dummy_searcher(), feedback_db_path=db_path)

            query = "hoteles de las playas"
            result = expander.expand_query(
                query,
                top_documents=[
                    {
                        "doc_id": "doc-1",
                        "title": "Casco historico de La Habana",
                        "summary": "Patrimonio historico y cultural",
                        "content_text": "casco historico habana vieja",
                    }
                ],
            )

            self.assertTrue(result.expanded_query.startswith(query))
            self.assertTrue(
                all(part not in {"de", "las"} for term in result.terms for part in term.split("_"))
            )

    def test_feedback_migrates_from_json_to_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            db_path = tmpdir_path / "feedback.db"
            legacy_path = tmpdir_path / "legacy_feedback.json"
            legacy_payload = {
                "explicit": [
                    {
                        "query": "hoteles en varadero",
                        "query_key": "hoteles en varadero",
                        "doc_id": "doc-1",
                        "relevance": 1,
                        "search_mode": "hybrid",
                        "timestamp": "2026-06-01T00:00:00+00:00",
                    }
                ],
                "implicit": [
                    {
                        "query": "hoteles en varadero",
                        "query_key": "hoteles en varadero",
                        "doc_id": "doc-1",
                        "event": "copy_url",
                        "event_group": "source_interaction",
                        "weight": 0.65,
                        "search_mode": "hybrid",
                        "timestamp": "2026-06-01T00:00:00+00:00",
                    }
                ],
            }
            legacy_path.write_text(json.dumps(legacy_payload, ensure_ascii=False), encoding="utf-8")

            script_path = Path("scripts/migrate_feedback_json_to_sqlite.py").resolve()
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--db",
                    str(db_path),
                    "--legacy",
                    str(legacy_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            expander = QueryExpander(build_dummy_searcher(), feedback_db_path=db_path)
            feedback = expander.feedback_store.query_feedback("hoteles en varadero")

            self.assertEqual(len(feedback["explicit"]), 1)
            self.assertEqual(len(feedback["implicit"]), 1)
            self.assertTrue(db_path.exists())
            self.assertIn("Registros explicit importados: 1", completed.stdout)
            self.assertIn("Registros implicit importados: 1", completed.stdout)


if __name__ == "__main__":
    unittest.main()
