import unittest

from src.indexing.tfidf_index import TFIDFIndex
from src.retrieval.domain_detector import DomainDetector, DomainThresholds
from src.retrieval.lsi_model import LSIModel


class FakeLLMClient:
    def __init__(self, result: bool):
        self.result = result
        self.called = False

    def classify_domain(self, prompt: str) -> bool:
        self.called = True
        self.prompt = prompt
        return self.result


def build_detector(*, llm_client=None, thresholds=None):
    documents = {
        "d1": ["hotel", "varader", "play", "cub"],
        "d2": ["haban", "viej", "muse", "turism"],
        "d3": ["sender", "naturalez", "cay", "cub"],
        "d4": ["restaur", "trinidad", "viaj", "cultur"],
    }
    tfidf = TFIDFIndex(min_df=1, max_df=1.0).build(documents)
    lsi = LSIModel(n_components=2).train(tfidf.matrix)
    return DomainDetector(
        tfidf,
        lsi,
        {"hotel", "varadero", "playa", "habana", "museo", "turismo", "cuba"},
        llm_client=llm_client,
        thresholds=thresholds,
    )


class DomainDetectorTests(unittest.TestCase):
    def test_in_domain_by_lexical_signal(self):
        detector = build_detector()
        explanation = detector.explain("hoteles en varadero")

        self.assertEqual(explanation["decision"], "IN_DOMAIN")
        self.assertGreaterEqual(explanation["features"]["keyword_overlap"], 1)
        self.assertFalse(explanation["used_llm"])

    def test_out_of_domain_by_weak_ir_and_no_keywords(self):
        detector = build_detector()
        explanation = detector.explain("hola")

        self.assertEqual(explanation["decision"], "OUT_OF_DOMAIN")
        self.assertEqual(explanation["features"]["keyword_overlap"], 0)
        self.assertFalse(explanation["used_llm"])

    def test_currency_weather_and_sports_are_out_of_domain(self):
        detector = build_detector()

        for query in ["precio del dolar", "hay sol manana", "quien gano el juego"]:
            with self.subTest(query=query):
                explanation = detector.explain(query)
                self.assertEqual(explanation["decision"], "OUT_OF_DOMAIN")
                self.assertGreaterEqual(explanation["features"]["out_of_domain_overlap"], 1)

    def test_place_plus_weather_without_tourism_intent_is_out(self):
        detector = build_detector()
        explanation = detector.explain("weather in havana")

        self.assertEqual(explanation["decision"], "OUT_OF_DOMAIN")
        self.assertGreaterEqual(explanation["features"]["out_of_domain_overlap"], 1)

    def test_tourism_intent_wins_over_generic_price_word(self):
        detector = build_detector()
        explanation = detector.explain("precio de hoteles en varadero")

        self.assertEqual(explanation["decision"], "IN_DOMAIN")
        self.assertGreaterEqual(explanation["features"]["tourism_intent_overlap"], 1)

    def test_llm_is_used_only_for_uncertain_cases(self):
        llm = FakeLLMClient(True)
        thresholds = DomainThresholds(
            out_max_score=-1.0,
            out_avg_score=-1.0,
            out_lsi_similarity=-1.0,
            in_max_score=2.0,
            in_avg_score=2.0,
            in_lsi_similarity=2.0,
        )
        detector = build_detector(llm_client=llm, thresholds=thresholds)
        explanation = detector.explain("consulta ambigua")

        self.assertTrue(llm.called)
        self.assertTrue(explanation["used_llm"])
        self.assertEqual(explanation["decision"], "IN_DOMAIN")
        self.assertIn("Responde SOLO YES o NO", llm.prompt)


if __name__ == "__main__":
    unittest.main()
