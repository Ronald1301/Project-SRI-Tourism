from __future__ import annotations

from statistics import mean
from typing import Any

from src.evaluation.constants import BASELINE_SYSTEM
from src.evaluation.metrics import (
    average_precision,
    bootstrap_mean_ci,
    extract_doc_ids,
    f1_at_k,
    ndcg_at_k,
    metric_mean,
    metric_stddev,
    precision_at_k,
    r_precision,
    recall_at_k,
    reciprocal_rank_at_k,
)
from src.evaluation.qrels import QueryJudgment, normalize_qrels, validate_qrels
from src.evaluation.report_generator import render_markdown_report
from src.evaluation.systems import build_systems, parse_systems, system_label
from src.retrieval.search import SemanticSearcher

EVALUATION_KS = (3, 5, 10)


def evaluate_searcher(searcher: SemanticSearcher, qrels: dict[str, Any], top_k: int) -> dict[str, Any]:
    """Backward-compatible REC-01 evaluator used by older commands/tests."""
    return evaluate_systems(
        qrels=qrels,
        top_k=top_k,
        systems=["lsi_baseline", "lsi_refined"],
        searcher=searcher,
        include_markdown=False,
    )


def evaluate_systems(
    *,
    qrels: dict[str, Any],
    top_k: int,
    systems: list[str] | str | None = None,
    searcher: SemanticSearcher | None = None,
    include_markdown: bool = True,
) -> dict[str, Any]:
    selected_systems = parse_systems(systems)
    judgments = normalize_qrels(qrels)
    validation_warnings = validate_qrels(qrels)
    runnable_systems, skipped = build_systems(selected_systems, searcher=searcher)

    systems_report: dict[str, Any] = {}
    for name in selected_systems:
        if name in skipped:
            systems_report[name] = {
                "system": name,
                "label": system_label(name),
                "status": "skipped",
                "reason": skipped[name],
                "summary": empty_summary(top_k),
                "statistics": {},
                "queries": [],
            }
            continue

        runnable = runnable_systems[name]
        rows: list[dict[str, Any]] = []
        for query_judgment in judgments:
            try:
                results = runnable.search(query_judgment.query, max(top_k, max(EVALUATION_KS)))
            except Exception as exc:
                validation_warnings.append(
                    f"{name}/{query_judgment.query_id}: error recuperando resultados: {exc}"
                )
                results = []

            retrieved_ids = extract_doc_ids(results)
            if not retrieved_ids:
                validation_warnings.append(f"{name}/{query_judgment.query_id}: sin resultados.")

            rows.append(
                evaluate_query(
                    query_judgment=query_judgment,
                    retrieved_ids=retrieved_ids,
                    top_k=top_k,
                )
            )

        summary, statistics = summarize(rows, top_k)
        systems_report[name] = {
            "system": name,
            "label": runnable.label,
            "status": "ok",
            "summary": summary,
            "statistics": statistics,
            "queries": rows,
        }

    report = {
        "name": "IR evaluation report",
        "description": (
            "Evaluacion offline con Precision, Recall, F1, MAP, MRR, NDCG y R-Precision "
            "sobre consultas de turismo."
        ),
        "query_count": len(judgments),
        "top_k": int(top_k),
        "systems_requested": selected_systems,
        "baseline_system": BASELINE_SYSTEM,
        "validation": {
            "warnings": sorted(set(validation_warnings)),
            "skipped_systems": skipped,
        },
        "systems": systems_report,
        "delta_vs_baseline": compute_deltas(systems_report, baseline=BASELINE_SYSTEM),
        "analysis": build_analysis(systems_report, baseline=BASELINE_SYSTEM),
    }
    if include_markdown:
        report["markdown"] = render_markdown_report(report)
    return report


def evaluate_query(query_judgment: QueryJudgment, retrieved_ids: list[str], top_k: int) -> dict[str, Any]:
    relevant_ids = query_judgment.relevant_ids
    return {
        "query_id": query_judgment.query_id,
        "query": query_judgment.query,
        "relevant_count": len(relevant_ids),
        "retrieved_count": len(retrieved_ids),
        "retrieved_doc_ids": retrieved_ids,
        "precision_at_3": precision_at_k(retrieved_ids, relevant_ids, 3),
        "precision_at_5": precision_at_k(retrieved_ids, relevant_ids, 5),
        "recall_at_5": recall_at_k(retrieved_ids, relevant_ids, 5),
        "recall_at_10": recall_at_k(retrieved_ids, relevant_ids, 10),
        "map": average_precision(retrieved_ids, relevant_ids),
        "ndcg_at_5": ndcg_at_k(retrieved_ids, query_judgment.judgments, 5),
        "mrr": reciprocal_rank_at_k(retrieved_ids, relevant_ids, len(retrieved_ids)),
        "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, top_k),
        "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, top_k),
        "f1_at_k": f1_at_k(retrieved_ids, relevant_ids, top_k),
        "ap": average_precision(retrieved_ids, relevant_ids),
        "mrr_at_k": reciprocal_rank_at_k(retrieved_ids, relevant_ids, top_k),
        "ndcg_at_k": ndcg_at_k(retrieved_ids, query_judgment.judgments, top_k),
        "r_precision": r_precision(retrieved_ids, relevant_ids),
    }


def summarize(rows: list[dict[str, Any]], top_k: int) -> tuple[dict[str, Any], dict[str, Any]]:
    metric_keys = [
        "precision_at_3",
        "precision_at_5",
        "recall_at_5",
        "recall_at_10",
        "map",
        "ndcg_at_5",
        "mrr",
        "precision_at_k",
        "recall_at_k",
        "f1_at_k",
        "ap",
        "mrr_at_k",
        "ndcg_at_k",
        "r_precision",
    ]
    summary = {metric: metric_mean([float(row.get(metric, 0.0)) for row in rows]) for metric in metric_keys}
    summary["k"] = float(top_k)

    statistics = {}
    for metric in metric_keys:
        values = [float(row.get(metric, 0.0)) for row in rows]
        ci_low, ci_high = bootstrap_mean_ci(values)
        statistics[metric] = {
            "mean": metric_mean(values),
            "stddev": metric_stddev(values),
            "ci95": {"low": ci_low, "high": ci_high},
        }

    return summary, statistics


def empty_summary(top_k: int) -> dict[str, float]:
    return {
        "precision_at_3": 0.0,
        "precision_at_5": 0.0,
        "recall_at_5": 0.0,
        "recall_at_10": 0.0,
        "map": 0.0,
        "ndcg_at_5": 0.0,
        "mrr": 0.0,
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "f1_at_k": 0.0,
        "ap": 0.0,
        "mrr_at_k": 0.0,
        "ndcg_at_k": 0.0,
        "r_precision": 0.0,
        "k": float(top_k),
    }


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rows]
    return mean(values) if values else 0.0


def compute_deltas(systems_report: dict[str, Any], *, baseline: str) -> dict[str, dict[str, float]]:
    baseline_payload = systems_report.get(baseline)
    if not baseline_payload or baseline_payload.get("status") != "ok":
        return {}
    baseline_summary = baseline_payload.get("summary", {})
    deltas: dict[str, dict[str, float]] = {}
    for name, payload in systems_report.items():
        if name == baseline or payload.get("status") != "ok":
            continue
        summary = payload.get("summary", {})
        deltas[name] = {
            metric: float(summary.get(metric, 0.0)) - float(baseline_summary.get(metric, 0.0))
            for metric in baseline_summary
            if metric != "k"
        }
    return deltas


def build_analysis(systems_report: dict[str, Any], *, baseline: str) -> list[str]:
    ok_systems = {
        name: payload
        for name, payload in systems_report.items()
        if payload.get("status") == "ok"
    }
    if not ok_systems:
        return ["No se pudo evaluar ningun sistema porque faltan artefactos de recuperacion."]

    best_f1 = max(ok_systems.items(), key=lambda item: item[1].get("summary", {}).get("f1_at_k", 0.0))
    best_ndcg = max(ok_systems.items(), key=lambda item: item[1].get("summary", {}).get("ndcg_at_k", 0.0))
    best_mrr = max(ok_systems.items(), key=lambda item: item[1].get("summary", {}).get("mrr_at_k", 0.0))

    analysis = [
        f"El mejor compromiso Precision/Recall segun F1@k lo obtiene `{best_f1[0]}` con {best_f1[1]['summary']['f1_at_k']:.4f}.",
        f"El mejor ordenamiento global segun NDCG@k lo obtiene `{best_ndcg[0]}` con {best_ndcg[1]['summary']['ndcg_at_k']:.4f}.",
        f"La mejor ubicacion temprana del primer relevante segun MRR@k la obtiene `{best_mrr[0]}` con {best_mrr[1]['summary']['mrr_at_k']:.4f}.",
    ]

    baseline_payload = ok_systems.get(baseline)
    if baseline_payload:
        baseline_f1 = baseline_payload["summary"]["f1_at_k"]
        improved = [
            name
            for name, payload in ok_systems.items()
            if name != baseline and payload["summary"]["f1_at_k"] > baseline_f1
        ]
        if improved:
            analysis.append(
                "Los sistemas que superan al baseline en F1@k son: "
                + ", ".join(f"`{name}`" for name in improved)
                + "."
            )
        else:
            analysis.append("Ningun sistema evaluado supera al baseline en F1@k con este conjunto de consultas.")

    skipped = [name for name, payload in systems_report.items() if payload.get("status") == "skipped"]
    if skipped:
        analysis.append("Sistemas omitidos por artefactos no disponibles: " + ", ".join(f"`{name}`" for name in skipped) + ".")
    return analysis
