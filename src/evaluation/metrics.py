from __future__ import annotations

import math
import random
from statistics import mean, pstdev
from typing import Any


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return _hit_count_at_k(retrieved_ids, relevant_ids, k) / float(k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0 or not relevant_ids:
        return 0.0
    return _hit_count_at_k(retrieved_ids, relevant_ids, k) / float(len(relevant_ids))


def f1_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    precision = precision_at_k(retrieved_ids, relevant_ids, k)
    recall = recall_at_k(retrieved_ids, relevant_ids, k)
    if precision + recall == 0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def reciprocal_rank_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0 or not relevant_ids:
        return 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / float(rank)
    return 0.0


def average_precision(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0

    hits = 0
    cumulative_precision = 0.0
    seen: set[str] = set()
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in seen:
            continue
        seen.add(doc_id)
        if doc_id not in relevant_ids:
            continue
        hits += 1
        cumulative_precision += hits / float(rank)

    return cumulative_precision / float(len(relevant_ids))


def r_precision(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    return precision_at_k(retrieved_ids, relevant_ids, len(relevant_ids))


def ndcg_at_k(retrieved_ids: list[str], judgments: dict[str, float], k: int) -> float:
    if k <= 0 or not judgments:
        return 0.0

    dcg = 0.0
    seen: set[str] = set()
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in seen:
            continue
        seen.add(doc_id)
        relevance = float(judgments.get(doc_id, 0.0))
        if relevance > 0:
            dcg += ((2.0**relevance) - 1.0) / math.log2(rank + 1)

    ideal_relevances = sorted(
        [float(value) for value in judgments.values() if float(value) > 0],
        reverse=True,
    )[:k]
    idcg = sum(
        ((2.0**relevance) - 1.0) / math.log2(rank + 1)
        for rank, relevance in enumerate(ideal_relevances, start=1)
    )
    if idcg == 0:
        return 0.0
    return dcg / idcg


def extract_doc_ids(results: list[Any]) -> list[str]:
    doc_ids: list[str] = []
    for item in results:
        if isinstance(item, dict):
            doc_id = item.get("doc_id")
        else:
            doc_id = getattr(item, "doc_id", None)
        doc_id = str(doc_id or "").strip()
        if doc_id:
            doc_ids.append(doc_id)
    return doc_ids


def hit_count_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> int:
    return _hit_count_at_k(retrieved_ids, relevant_ids, k)


def metric_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def metric_stddev(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return 0.0
    return pstdev(values)


def bootstrap_mean_ci(
    values: list[float],
    *,
    confidence: float = 0.95,
    resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        single = float(values[0])
        return single, single

    rng = random.Random(seed)
    sample_means: list[float] = []
    n = len(values)
    for _ in range(max(int(resamples), 1)):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        sample_means.append(mean(sample))

    sample_means.sort()
    alpha = max(0.0, min(1.0, 1.0 - float(confidence))) / 2.0
    lower_idx = max(int(alpha * len(sample_means)), 0)
    upper_idx = min(int((1.0 - alpha) * len(sample_means)) - 1, len(sample_means) - 1)
    return float(sample_means[lower_idx]), float(sample_means[upper_idx])


def _hit_count_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> int:
    hits = 0
    seen: set[str] = set()
    for doc_id in retrieved_ids[:k]:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        if doc_id in relevant_ids:
            hits += 1
    return hits
