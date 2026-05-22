from __future__ import annotations

from typing import Any, Mapping

from src.retrieval.ranking_signals import (
    DEFAULT_CONTENT_TYPE_VALUES,
    DEFAULT_RECENCY_HALF_LIFE_DAYS,
    QuerySignalContext,
    build_query_signal_context,
    compute_signal_values,
)

DEFAULT_RERANK_CONFIG = {
    "weights": {
        "phrase_match": 0.08,
        "recency": 0.03,
        "content_type_boost": 0.04,
        "authority": 0.04,
        "specificity": 0.07,
    },
    "recency_half_life_days": DEFAULT_RECENCY_HALF_LIFE_DAYS,
    "content_type_values": dict(DEFAULT_CONTENT_TYPE_VALUES),
}


class SignalReranker:
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        merged = _merge_config(config)
        self.weights: dict[str, float] = merged["weights"]
        self.recency_half_life_days: float = float(merged["recency_half_life_days"])
        self.content_type_values: dict[str, float] = merged["content_type_values"]

    def prepare_query(self, query: str, query_tokens: list[str]) -> QuerySignalContext:
        return build_query_signal_context(query=query, query_tokens=query_tokens)

    def score(
        self,
        *,
        base_score: float,
        query_context: QuerySignalContext,
        document: Mapping[str, Any],
        debug: bool = False,
    ) -> dict[str, Any]:
        signals = compute_signal_values(
            query_context=query_context,
            document=document,
            recency_half_life_days=self.recency_half_life_days,
            content_type_values=self.content_type_values,
        )

        # final_score = base_score + sum(weight_i * signal_i)
        signal_total = (
            (self.weights["phrase_match"] * signals.phrase_match)
            + (self.weights["recency"] * signals.recency)
            + (self.weights["content_type_boost"] * signals.content_type_boost)
            + (self.weights["authority"] * signals.authority)
            + (self.weights["specificity"] * signals.specificity)
        )
        final_score = float(base_score) + float(signal_total)

        components = {
            "lsi_score": float(base_score),
            "lexical_overlap": float(signals.lexical_overlap),
            "title_match": float(signals.title_match),
            "length_signal": float(signals.length_signal),
            "phrase_match": float(signals.phrase_match),
            "recency": float(signals.recency),
            "content_type_boost": float(signals.content_type_boost),
            "authority": float(signals.authority),
            "specificity": float(signals.specificity),
            "signal_total": float(signal_total),
        }

        payload: dict[str, Any] = {
            "final_score": final_score,
            "score_components": components,
        }
        if debug:
            payload["score_debug"] = {
                "base_score": float(base_score),
                "phrase": float(signals.phrase_match),
                "recency": float(signals.recency),
                "type": float(signals.content_type_boost),
                "authority": float(signals.authority),
                "specificity": float(signals.specificity),
                "final_score": float(final_score),
            }
        return payload


def _merge_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = {
        "weights": dict(DEFAULT_RERANK_CONFIG["weights"]),
        "recency_half_life_days": float(DEFAULT_RERANK_CONFIG["recency_half_life_days"]),
        "content_type_values": dict(DEFAULT_RERANK_CONFIG["content_type_values"]),
    }
    if not config:
        return merged

    incoming_weights = config.get("weights")
    if isinstance(incoming_weights, Mapping):
        for key in merged["weights"]:
            if key in incoming_weights:
                merged["weights"][key] = float(incoming_weights[key])

    if "recency_half_life_days" in config:
        merged["recency_half_life_days"] = float(config["recency_half_life_days"])

    incoming_type_values = config.get("content_type_values")
    if isinstance(incoming_type_values, Mapping):
        for key, value in incoming_type_values.items():
            merged["content_type_values"][str(key)] = float(value)

    return merged
