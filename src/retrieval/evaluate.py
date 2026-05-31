from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from src.retrieval.query_expansion import QueryExpander
from src.retrieval.search import (
    DEFAULT_LSI_META,
    DEFAULT_LSI_MODEL,
    DEFAULT_LSI_VECTORS,
    DEFAULT_TFIDF_MATRIX,
    DEFAULT_TFIDF_META,
    DEFAULT_TFIDF_VOCAB,
    SemanticSearcher,
)

DEFAULT_QRELS_PATH = Path("data/evaluation/rec01_qrels.json")
DEFAULT_REPORT_PATH = Path("data/evaluation/reports/eval_report.json")
DEFAULT_MARKDOWN_REPORT_PATH = Path("data/evaluation/reports/eval_report.md")
SUPPORTED_SYSTEMS = [
    "lsi_baseline",
    "lsi_refined",
    "lsi_expanded",
    "vectorial",
    "hybrid_search",
]
BASELINE_SYSTEM = "lsi_baseline"


@dataclass(frozen=True)
class QueryJudgment:
    query_id: str
    query: str
    relevant_ids: set[str]
    judgments: dict[str, float]


@dataclass
class RunnableSystem:
    name: str
    label: str
    search: Callable[[str, int], list[dict[str, Any]]]


def missing_lsi_artifacts() -> list[str]:
    matrix_path = Path(DEFAULT_TFIDF_MATRIX)
    matrix_exists = matrix_path.exists() or matrix_path.with_suffix(".npy").exists()
    required_files = [
        DEFAULT_TFIDF_VOCAB,
        DEFAULT_TFIDF_META,
        DEFAULT_LSI_MODEL,
        DEFAULT_LSI_VECTORS,
        DEFAULT_LSI_META,
    ]
    missing = [path for path in required_files if not Path(path).exists()]
    if not matrix_exists:
        missing.insert(0, DEFAULT_TFIDF_MATRIX)
    return missing


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = _hit_count_at_k(retrieved_ids, relevant_ids, k)
    return hits / float(k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0 or not relevant_ids:
        return 0.0
    hits = _hit_count_at_k(retrieved_ids, relevant_ids, k)
    return hits / float(len(relevant_ids))


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

        empty_ids = [doc_id for doc_id in relevant_doc_ids if not doc_id]
        if empty_ids:
            warnings.append(f"{query_id}: contiene doc_id vacios.")

        duplicates = sorted({doc_id for doc_id in relevant_doc_ids if relevant_doc_ids.count(doc_id) > 1})
        if duplicates:
            warnings.append(f"{query_id}: doc_ids repetidos: {', '.join(duplicates)}.")

    return warnings


def parse_systems(raw_systems: str | list[str] | None) -> list[str]:
    if raw_systems is None:
        return list(SUPPORTED_SYSTEMS)
    if isinstance(raw_systems, str):
        requested = [item.strip() for item in raw_systems.split(",") if item.strip()]
    else:
        requested = [str(item).strip() for item in raw_systems if str(item).strip()]

    if not requested or requested == ["all"] or "all" in requested:
        return list(SUPPORTED_SYSTEMS)

    invalid = [system for system in requested if system not in SUPPORTED_SYSTEMS]
    if invalid:
        raise ValueError(
            f"Sistemas no soportados: {', '.join(invalid)}. "
            f"Usa: {', '.join(SUPPORTED_SYSTEMS)} o all."
        )
    return requested


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
    runnable_systems, skipped = _build_systems(selected_systems, searcher=searcher)

    systems_report: dict[str, Any] = {}
    for name in selected_systems:
        if name in skipped:
            systems_report[name] = {
                "system": name,
                "label": _system_label(name),
                "status": "skipped",
                "reason": skipped[name],
                "summary": _empty_summary(top_k),
                "queries": [],
            }
            continue

        runnable = runnable_systems[name]
        rows: list[dict[str, Any]] = []
        for query_judgment in judgments:
            try:
                results = runnable.search(query_judgment.query, top_k)
            except Exception as exc:
                validation_warnings.append(
                    f"{name}/{query_judgment.query_id}: error recuperando resultados: {exc}"
                )
                results = []

            retrieved_ids = _extract_doc_ids(results)
            if not retrieved_ids:
                validation_warnings.append(f"{name}/{query_judgment.query_id}: sin resultados.")

            rows.append(
                _evaluate_query(
                    query_judgment=query_judgment,
                    retrieved_ids=retrieved_ids,
                    top_k=top_k,
                )
            )

        systems_report[name] = {
            "system": name,
            "label": runnable.label,
            "status": "ok",
            "summary": _summarize(rows, top_k),
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
        "delta_vs_baseline": _compute_deltas(systems_report, baseline=BASELINE_SYSTEM),
        "analysis": _build_analysis(systems_report, baseline=BASELINE_SYSTEM),
    }
    if include_markdown:
        report["markdown"] = render_markdown_report(report)
    return report


def write_reports(report: dict[str, Any], json_path: str | Path, markdown_path: str | Path | None = None) -> None:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {key: value for key, value in report.items() if key != "markdown"}
    json_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")

    if markdown_path is not None:
        markdown_path = Path(markdown_path)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            report.get("markdown") or render_markdown_report(report),
            encoding="utf-8",
        )


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Reporte de evaluacion IR",
        "",
        f"- Consultas evaluadas: {report.get('query_count', 0)}",
        f"- Top-k: {report.get('top_k', 0)}",
        f"- Baseline: `{report.get('baseline_system', BASELINE_SYSTEM)}`",
        "",
        "## Resumen por sistema",
        "",
        "| Sistema | Estado | P@k | Recall@k | F1@k | MAP | MRR@k | NDCG@k | R-Precision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name, payload in report.get("systems", {}).items():
        summary = payload.get("summary", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    payload.get("status", "unknown"),
                    _fmt(summary.get("precision_at_k")),
                    _fmt(summary.get("recall_at_k")),
                    _fmt(summary.get("f1_at_k")),
                    _fmt(summary.get("map")),
                    _fmt(summary.get("mrr_at_k")),
                    _fmt(summary.get("ndcg_at_k")),
                    _fmt(summary.get("r_precision")),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Delta contra baseline", ""])
    deltas = report.get("delta_vs_baseline", {})
    if deltas:
        lines.extend(
            [
                "| Sistema | Delta F1@k | Delta NDCG@k | Delta MRR@k |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, values in deltas.items():
            lines.append(
                f"| {name} | {_fmt_delta(values.get('f1_at_k'))} | "
                f"{_fmt_delta(values.get('ndcg_at_k'))} | {_fmt_delta(values.get('mrr_at_k'))} |"
            )
    else:
        lines.append("No hay deltas disponibles.")

    lines.extend(["", "## Analisis cuantitativo", ""])
    for item in report.get("analysis", []):
        lines.append(f"- {item}")

    warnings = report.get("validation", {}).get("warnings", [])
    if warnings:
        lines.extend(["", "## Advertencias de validacion", ""])
        for warning in warnings:
            lines.append(f"- {warning}")

    lines.extend(["", "## Mejores y peores consultas por sistema", ""])
    for name, payload in report.get("systems", {}).items():
        if payload.get("status") != "ok":
            continue
        rows = payload.get("queries", [])
        if not rows:
            continue
        best = max(rows, key=lambda row: row.get("f1_at_k", 0.0))
        worst = min(rows, key=lambda row: row.get("f1_at_k", 0.0))
        lines.append(
            f"- `{name}` mejor: {best['query_id']} ({best['query']}) F1@k={best['f1_at_k']:.4f}; "
            f"peor: {worst['query_id']} ({worst['query']}) F1@k={worst['f1_at_k']:.4f}."
        )

    return "\n".join(lines) + "\n"


def _build_systems(
    selected_systems: list[str],
    *,
    searcher: SemanticSearcher | None = None,
) -> tuple[dict[str, RunnableSystem], dict[str, str]]:
    runnable: dict[str, RunnableSystem] = {}
    skipped: dict[str, str] = {}

    lsi_systems = {"lsi_baseline", "lsi_refined", "lsi_expanded"} & set(selected_systems)
    if lsi_systems:
        missing = missing_lsi_artifacts()
        if missing:
            reason = "Faltan artefactos LSI/TF-IDF: " + ", ".join(missing)
            for name in lsi_systems:
                skipped[name] = reason
        else:
            searcher = searcher or SemanticSearcher()
            runnable["lsi_baseline"] = RunnableSystem(
                name="lsi_baseline",
                label="TF-IDF + LSI baseline",
                search=lambda query, top_k: searcher.search_baseline(query, top_k=top_k),
            )
            runnable["lsi_refined"] = RunnableSystem(
                name="lsi_refined",
                label="TF-IDF + LSI + reranking",
                search=lambda query, top_k: searcher.search(query, top_k=top_k),
            )
            expander = QueryExpander(searcher)
            runnable["lsi_expanded"] = RunnableSystem(
                name="lsi_expanded",
                label="TF-IDF + LSI + expansion + reranking",
                search=lambda query, top_k: _search_with_expansion(searcher, expander, query, top_k),
            )

    rag_modes = {"vectorial", "hybrid_search"} & set(selected_systems)
    if rag_modes:
        try:
            from src.RAG.rag_pipeline import RAGPipeline

            rag_searcher = searcher
            if "hybrid_search" in rag_modes and rag_searcher is None and not missing_lsi_artifacts():
                rag_searcher = SemanticSearcher()

            rag = RAGPipeline.from_preset(semantic_searcher=rag_searcher)
            runnable["vectorial"] = RunnableSystem(
                name="vectorial",
                label="Vectorial embeddings",
                search=lambda query, top_k: _rag_retrieve(rag, query, top_k, "vectorial"),
            )
            runnable["hybrid_search"] = RunnableSystem(
                name="hybrid_search",
                label="Hybrid search RRF",
                search=lambda query, top_k: _rag_retrieve(rag, query, top_k, "hybrid_search"),
            )
        except Exception as exc:
            for name in rag_modes:
                skipped[name] = f"No se pudo inicializar RAG/VectorDB: {exc}"

    return (
        {name: system for name, system in runnable.items() if name in selected_systems},
        skipped,
    )


def _search_with_expansion(
    searcher: SemanticSearcher,
    expander: QueryExpander,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    preview = searcher.search(query, top_k=max(3, top_k))
    expansion = expander.expand_query(query, top_documents=preview[:3])
    selected_query = expansion.expanded_query if expansion.applied else query
    return searcher.search(selected_query, top_k=top_k)


def _rag_retrieve(rag: Any, query: str, top_k: int, mode: str) -> list[dict[str, Any]]:
    documents = rag.retrieve(query, top_k=top_k, search_mode=mode, include_explanations=False)
    return [{"doc_id": doc.doc_id, "score": doc.score, "title": doc.title} for doc in documents]


def _evaluate_query(query_judgment: QueryJudgment, retrieved_ids: list[str], top_k: int) -> dict[str, Any]:
    relevant_ids = query_judgment.relevant_ids
    return {
        "query_id": query_judgment.query_id,
        "query": query_judgment.query,
        "relevant_count": len(relevant_ids),
        "retrieved_count": len(retrieved_ids),
        "retrieved_doc_ids": retrieved_ids,
        "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, top_k),
        "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, top_k),
        "f1_at_k": f1_at_k(retrieved_ids, relevant_ids, top_k),
        "ap": average_precision(retrieved_ids, relevant_ids),
        "mrr_at_k": reciprocal_rank_at_k(retrieved_ids, relevant_ids, top_k),
        "ndcg_at_k": ndcg_at_k(retrieved_ids, query_judgment.judgments, top_k),
        "r_precision": r_precision(retrieved_ids, relevant_ids),
    }


def _summarize(rows: list[dict[str, Any]], top_k: int) -> dict[str, float]:
    return {
        "precision_at_k": _mean_metric(rows, "precision_at_k"),
        "recall_at_k": _mean_metric(rows, "recall_at_k"),
        "f1_at_k": _mean_metric(rows, "f1_at_k"),
        "map": _mean_metric(rows, "ap"),
        "mrr_at_k": _mean_metric(rows, "mrr_at_k"),
        "ndcg_at_k": _mean_metric(rows, "ndcg_at_k"),
        "r_precision": _mean_metric(rows, "r_precision"),
        "k": float(top_k),
    }


def _empty_summary(top_k: int) -> dict[str, float]:
    return {
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "f1_at_k": 0.0,
        "map": 0.0,
        "mrr_at_k": 0.0,
        "ndcg_at_k": 0.0,
        "r_precision": 0.0,
        "k": float(top_k),
    }


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rows]
    return mean(values) if values else 0.0


def _compute_deltas(systems_report: dict[str, Any], *, baseline: str) -> dict[str, dict[str, float]]:
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


def _build_analysis(systems_report: dict[str, Any], *, baseline: str) -> list[str]:
    ok_systems = {
        name: payload
        for name, payload in systems_report.items()
        if payload.get("status") == "ok"
    }
    if not ok_systems:
        return ["No se pudo evaluar ningun sistema porque faltan artefactos de recuperacion."]

    best_f1 = max(
        ok_systems.items(),
        key=lambda item: item[1].get("summary", {}).get("f1_at_k", 0.0),
    )
    best_ndcg = max(
        ok_systems.items(),
        key=lambda item: item[1].get("summary", {}).get("ndcg_at_k", 0.0),
    )
    best_mrr = max(
        ok_systems.items(),
        key=lambda item: item[1].get("summary", {}).get("mrr_at_k", 0.0),
    )

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
                "Los sistemas que superan al baseline en F1@k son: " + ", ".join(f"`{name}`" for name in improved) + "."
            )
        else:
            analysis.append("Ningun sistema evaluado supera al baseline en F1@k con este conjunto de consultas.")

    skipped = [name for name, payload in systems_report.items() if payload.get("status") == "skipped"]
    if skipped:
        analysis.append("Sistemas omitidos por artefactos no disponibles: " + ", ".join(f"`{name}`" for name in skipped) + ".")
    return analysis


def _extract_doc_ids(results: list[Any]) -> list[str]:
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


def _system_label(name: str) -> str:
    labels = {
        "lsi_baseline": "TF-IDF + LSI baseline",
        "lsi_refined": "TF-IDF + LSI + reranking",
        "lsi_expanded": "TF-IDF + LSI + expansion + reranking",
        "vectorial": "Vectorial embeddings",
        "hybrid_search": "Hybrid search RRF",
    }
    return labels.get(name, name)


def _fmt(value: Any) -> str:
    return f"{float(value or 0.0):.4f}"


def _fmt_delta(value: Any) -> str:
    return f"{float(value or 0.0):+.4f}"


def _print_report(report: dict[str, Any]) -> None:
    print("Evaluacion IR")
    print(f"- Consultas: {report['query_count']}")
    print(f"- top_k: {report['top_k']}")
    print("")

    for name, payload in report["systems"].items():
        print(f"{name} [{payload['status']}]")
        if payload["status"] == "skipped":
            print(f"  razon: {payload['reason']}")
            continue
        summary = payload["summary"]
        print(f"  Precision@k: {summary['precision_at_k']:.4f}")
        print(f"  Recall@k: {summary['recall_at_k']:.4f}")
        print(f"  F1@k: {summary['f1_at_k']:.4f}")
        print(f"  MAP: {summary['map']:.4f}")
        print(f"  MRR@k: {summary['mrr_at_k']:.4f}")
        print(f"  NDCG@k: {summary['ndcg_at_k']:.4f}")
        print(f"  R-Precision: {summary['r_precision']:.4f}")

    if report.get("analysis"):
        print("")
        print("Analisis:")
        for item in report["analysis"]:
            print(f"- {item}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evalua sistemas IR con metricas objetivas.")
    parser.add_argument("--qrels", default=str(DEFAULT_QRELS_PATH), help="Ruta al archivo JSON de consultas relevantes.")
    parser.add_argument("--top-k", type=int, default=5, help="Cantidad maxima de resultados a evaluar por consulta.")
    parser.add_argument("--systems", default="all", help="Sistemas separados por coma o 'all'.")
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT_PATH), help="Ruta de salida para guardar el reporte JSON.")
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN_REPORT_PATH), help="Ruta de salida para guardar el reporte Markdown.")
    args = parser.parse_args(argv)

    qrels_path = Path(args.qrels)
    if not qrels_path.exists():
        raise FileNotFoundError(f"No se encontro archivo de evaluacion: {qrels_path}")

    qrels = load_qrels(qrels_path)
    report = evaluate_systems(qrels=qrels, top_k=args.top_k, systems=args.systems)
    write_reports(report, args.report_out, args.markdown_out)
    _print_report(report)
    print(f"\nReporte JSON: {args.report_out}")
    print(f"Reporte Markdown: {args.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
