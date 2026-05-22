from __future__ import annotations

from typing import Any, Literal, Mapping, TypedDict

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


_WEIGHTED_SIGNAL_NAMES = (
    "phrase_match",
    "recency",
    "content_type_boost",
    "authority",
    "specificity",
)

_NORMALIZED_SIGNAL_FIELDS = (
    "phrase_match",
    "recency",
    "content_type_boost",
    "authority",
    "specificity",
    "lexical_overlap",
    "title_match",
    "length_signal",
)


class ExplanationComponent(TypedDict):
    name: str
    value: float
    weight: float
    contribution: float
    category: Literal["base", "signal"]


class ExplanationAdjustment(TypedDict):
    name: str
    value: float
    contribution: float


class ExplanationNormalization(TypedDict):
    name: str
    method: str
    fields: list[str]


class ExplanationExactMatches(TypedDict):
    full_query_phrase: bool


class RankingExplanation(TypedDict):
    schema_version: str
    final_score: float
    base_score: float
    signal_total: float
    components: list[ExplanationComponent]
    boosts: list[ExplanationAdjustment]
    penalties: list[ExplanationAdjustment]
    normalizations: list[ExplanationNormalization]
    weights: dict[str, float]
    signals: dict[str, float]
    exact_matches: ExplanationExactMatches


class RankingPayload(TypedDict, total=False):
    final_score: float
    score_components: dict[str, float]
    score_debug: dict[str, float]
    explanation: RankingExplanation


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
        include_explanation: bool = False,
    ) -> RankingPayload:
        signals = compute_signal_values(
            query_context=query_context,
            document=document,
            recency_half_life_days=self.recency_half_life_days,
            content_type_values=self.content_type_values,
        )

        weighted_signal_values = {
            "phrase_match": float(signals.phrase_match),
            "recency": float(signals.recency),
            "content_type_boost": float(signals.content_type_boost),
            "authority": float(signals.authority),
            "specificity": float(signals.specificity),
        }

        # final_score = base_score + sum(weight_i * signal_i)
        signal_total = sum(
            float(self.weights[name]) * float(weighted_signal_values[name])
            for name in _WEIGHTED_SIGNAL_NAMES
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

        payload: RankingPayload = {
            "final_score": final_score,
            "score_components": components,
        }
        if include_explanation:
            payload["explanation"] = self._build_explanation(
                base_score=float(base_score),
                final_score=float(final_score),
                signal_total=float(signal_total),
                weighted_signal_values=weighted_signal_values,
                all_signal_values={
                    "phrase_match": float(signals.phrase_match),
                    "recency": float(signals.recency),
                    "content_type_boost": float(signals.content_type_boost),
                    "authority": float(signals.authority),
                    "specificity": float(signals.specificity),
                    "lexical_overlap": float(signals.lexical_overlap),
                    "title_match": float(signals.title_match),
                    "length_signal": float(signals.length_signal),
                },
            )
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

    def build_baseline_explanation(self, *, base_score: float) -> RankingExplanation:
        safe_score = float(base_score)
        return {
            "schema_version": "ranking_explanation.v1",
            "final_score": safe_score,
            "base_score": safe_score,
            "signal_total": 0.0,
            "components": [
                {
                    "name": "semantic_lsi",
                    "value": safe_score,
                    "weight": 1.0,
                    "contribution": safe_score,
                    "category": "base",
                }
            ],
            "boosts": [],
            "penalties": [],
            "normalizations": [],
            "weights": {},
            "signals": {},
            "exact_matches": {
                "full_query_phrase": False,
            },
        }

    def _build_explanation(
        self,
        *,
        base_score: float,
        final_score: float,
        signal_total: float,
        weighted_signal_values: Mapping[str, float],
        all_signal_values: Mapping[str, float],
    ) -> RankingExplanation:
        components: list[ExplanationComponent] = [
            {
                "name": "semantic_lsi",
                "value": float(base_score),
                "weight": 1.0,
                "contribution": float(base_score),
                "category": "base",
            }
        ]
        boosts: list[ExplanationAdjustment] = []
        penalties: list[ExplanationAdjustment] = []

        for name in _WEIGHTED_SIGNAL_NAMES:
            value = float(weighted_signal_values.get(name, 0.0))
            weight = float(self.weights.get(name, 0.0))
            contribution = weight * value
            components.append(
                {
                    "name": name,
                    "value": value,
                    "weight": weight,
                    "contribution": contribution,
                    "category": "signal",
                }
            )
            if contribution > 0.0:
                boosts.append(
                    {
                        "name": name,
                        "value": value,
                        "contribution": contribution,
                    }
                )
            elif contribution < 0.0:
                penalties.append(
                    {
                        "name": name,
                        "value": value,
                        "contribution": contribution,
                    }
                )

        return {
            "schema_version": "ranking_explanation.v1",
            "final_score": float(final_score),
            "base_score": float(base_score),
            "signal_total": float(signal_total),
            "components": components,
            "boosts": boosts,
            "penalties": penalties,
            "normalizations": [
                {
                    "name": "signal_clamp01",
                    "method": "clamp01",
                    "fields": list(_NORMALIZED_SIGNAL_FIELDS),
                },
                {
                    "name": "recency_exponential_decay",
                    "method": "half_life_days",
                    "fields": ["recency"],
                },
            ],
            "weights": {name: float(self.weights.get(name, 0.0)) for name in _WEIGHTED_SIGNAL_NAMES},
            "signals": {name: float(value) for name, value in all_signal_values.items()},
            "exact_matches": {
                "full_query_phrase": float(all_signal_values.get("phrase_match", 0.0)) >= 0.999,
            },
        }


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
