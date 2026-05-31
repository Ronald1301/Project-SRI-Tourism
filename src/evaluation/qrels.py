from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QueryJudgment:
    query_id: str
    query: str
    relevant_ids: set[str]
    judgments: dict[str, float]


def load_qrels(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_qrels(qrels: dict[str, Any]) -> list[QueryJudgment]:
    normalized: list[QueryJudgment] = []
    for index, entry in enumerate(qrels.get("queries", []), start=1):
        query = str(entry.get("query") or "").strip()
        if not query:
            continue

        query_id = str(entry.get("query_id") or f"q{index}")
        raw_judgments = entry.get("judgments") or {}
        judgments = {
            str(doc_id): float(relevance)
            for doc_id, relevance in raw_judgments.items()
            if str(doc_id).strip() and float(relevance) > 0
        }

        relevant_ids = {
            str(doc_id).strip()
            for doc_id in entry.get("relevant_doc_ids") or []
            if str(doc_id).strip()
        }
        for doc_id in relevant_ids:
            judgments.setdefault(doc_id, 1.0)
        relevant_ids.update(doc_id for doc_id, value in judgments.items() if value > 0)

        normalized.append(
            QueryJudgment(
                query_id=query_id,
                query=query,
                relevant_ids=relevant_ids,
                judgments=judgments,
            )
        )
    return normalized


def validate_qrels(qrels: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    queries = qrels.get("queries")
    if not isinstance(queries, list) or not queries:
        return ["El archivo qrels no contiene una lista de consultas en 'queries'."]

    for index, entry in enumerate(queries, start=1):
        query_id = str(entry.get("query_id") or f"q{index}")
        query = str(entry.get("query") or "").strip()
        relevant_doc_ids = [str(doc_id).strip() for doc_id in entry.get("relevant_doc_ids") or []]
        judgments = entry.get("judgments") or {}

        if not query:
            warnings.append(f"{query_id}: consulta vacia.")
        if not relevant_doc_ids and not judgments:
            warnings.append(f"{query_id}: no tiene documentos relevantes.")
        if any(not doc_id for doc_id in relevant_doc_ids):
            warnings.append(f"{query_id}: contiene doc_id vacios.")

        duplicates = sorted({doc_id for doc_id in relevant_doc_ids if relevant_doc_ids.count(doc_id) > 1})
        if duplicates:
            warnings.append(f"{query_id}: doc_ids repetidos: {', '.join(duplicates)}.")

    return warnings
