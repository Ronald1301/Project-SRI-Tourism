from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.expantion.cache import CacheManager
from src.expantion.config import ConfigLoader
from src.expantion.feedback_database import FeedbackDatabase
from src.expantion.ngrams import NGramExtractor
from src.expantion.query_expander import QueryExpander
from src.expantion.synonyms import SynonymLoader


class FakeSearcher:
    def __init__(self) -> None:
        self.documents_by_id = {
            "d1": {
                "doc_id": "d1",
                "title": "Playas de La Habana",
                "summary": "Playas del Este, costa, mar y actividades familiares.",
                "content_text": "La Habana tiene playas cercanas, hoteles y restaurantes.",
            }
        }
        self.tfidf_index = type("FakeIndex", (), {"matrix": None, "vocabulary": {}})()
        self.pipeline = type("FakePipeline", (), {"process_text": lambda _, text: str(text).split()})()


class QueryExpansionEnhancedTests(unittest.TestCase):
    def test_load_config(self):
        config = ConfigLoader().load()

        self.assertEqual(config.max_terms, 5)
        self.assertAlmostEqual(config.acceptance_threshold, 0.75)
        self.assertTrue(config.get("ngrams.enabled"))

    def test_load_synonyms_bidirectional(self):
        config = ConfigLoader().load()
        synonyms = SynonymLoader(config).load()

        self.assertIn("playa", synonyms.get("cayo", []))
        self.assertIn("beach", synonyms.get("playa", []))
        self.assertGreater(sum(len(values) for values in synonyms.values()), 70)

    def test_ngram_extraction(self):
        extractor = NGramExtractor(enabled=True, min_n=1, max_n=3, multipliers={"1": 1.0, "2": 1.2, "3": 1.5})
        grams = extractor.extract(["habana", "vieja", "museo"])

        self.assertIn(("habana", 1.0), grams)
        self.assertIn(("habana vieja", 1.2), grams)
        self.assertIn(("habana vieja museo", 1.5), grams)

    def test_feedback_database_crud_and_implicit_dedup(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = FeedbackDatabase(Path(tmp) / "feedback.db", ConfigLoader().load())
            explicit = db.add_explicit(query="playas", doc_id="d1", relevance=1)
            first, counted_first = db.add_implicit(query="playas", doc_id="d1", event="copy_url")
            second, counted_second = db.add_implicit(query="playas", doc_id="d1", event="open_source")
            feedback = db.query_feedback("playas")

        self.assertGreater(explicit["id"], 0)
        self.assertTrue(counted_first)
        self.assertFalse(counted_second)
        self.assertEqual(first["event_group"], second["event_group"])
        self.assertEqual(len(feedback["explicit"]), 1)
        self.assertEqual(len(feedback["implicit"]), 1)

    def test_cache_lru(self):
        cache = CacheManager(max_size=1, ttl_seconds=60)
        cache.set("a", 1)
        cache.set("b", 2)

        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("b"), 2)

    def test_expander_supports_english_and_spanish(self):
        with tempfile.TemporaryDirectory() as tmp:
            expander = QueryExpander(FakeSearcher(), feedback_db_path=Path(tmp) / "feedback.db")
            spanish = expander.expand_query(
                "playas en la habana",
                top_documents=[FakeSearcher().documents_by_id["d1"]],
            )
            english = expander.expand_query(
                "beaches near Havana",
                top_documents=[FakeSearcher().documents_by_id["d1"]],
            )

        self.assertTrue(spanish.applied)
        self.assertTrue(english.applied)
        self.assertIn("language", spanish.trace)
        self.assertIn("language", english.trace)

    def test_rocchio_fallback_does_not_break_expansion(self):
        with tempfile.TemporaryDirectory() as tmp:
            expander = QueryExpander(FakeSearcher(), feedback_db_path=Path(tmp) / "feedback.db")
            result = expander.expand_query("cayo", top_documents=[FakeSearcher().documents_by_id["d1"]], max_terms=8)

        self.assertTrue(result.applied)
        self.assertTrue(any(term.startswith("playa") for term in result.terms))

    def test_acceptance_threshold_75_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            expander = QueryExpander(FakeSearcher(), feedback_db_path=Path(tmp) / "feedback.db")

        self.assertAlmostEqual(expander.acceptance_threshold, 0.75)


if __name__ == "__main__":
    unittest.main()
