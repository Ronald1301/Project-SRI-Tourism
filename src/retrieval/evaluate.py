from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

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
DEFAULT_REPORT_PATH = Path("data/evaluation/reports/rec01_eval_report.json")


def missing_lsi_artifacts() -> list[str]:
    required_files = [
        DEFAULT_TFIDF_MATRIX,
        DEFAULT_TFIDF_VOCAB,
        DEFAULT_TFIDF_META,
        DEFAULT_LSI_MODEL,
        DEFAULT_LSI_VECTORS,
        DEFAULT_LSI_META,
    ]
    return [path for path in required_files if not Path(path).exists()]


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return hits / float(k)


def average_precision(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0

    hits = 0
    cumulative_precision = 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id not in relevant_ids:
            continue
        hits += 1
        cumulative_precision += hits / float(rank)

    return cumulative_precision / float(len(relevant_ids))


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0 or not relevant_ids:
        return 0.0

    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        if rel > 0:
            dcg += rel / math.log2(rank + 1)

    ideal_hits = min(len(relevant_ids), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def _mean_metric(rows: Iterable[dict[str, float]], key: str) -> float:
    values = [row[key] for row in rows]
    return mean(values) if values else 0.0


def evaluate_searcher(searcher: SemanticSearcher, qrels: dict[str, Any], top_k: int) -> dict[str, Any]:
    baseline_rows: list[dict[str, Any]] = []
    refined_rows: list[dict[str, Any]] = []

    for entry in qrels.get("queries", []):
        query = str(entry.get("query") or "").strip()
        query_id = str(entry.get("query_id") or "")
        relevant_ids = set(entry.get("relevant_doc_ids") or [])
        if not query:
            continue

        baseline_results = searcher.search_baseline(query, top_k=top_k)
        refined_results = searcher.search(query, top_k=top_k)

        baseline_ids = [str(item.get("doc_id") or "") for item in baseline_results]
        refined_ids = [str(item.get("doc_id") or "") for item in refined_results]

        baseline_rows.append(
            {
                "query_id": query_id,
                "query": query,
                "retrieved_doc_ids": baseline_ids,
                "p_at_3": precision_at_k(baseline_ids, relevant_ids, 3),
                "p_at_5": precision_at_k(baseline_ids, relevant_ids, 5),
                "ap": average_precision(baseline_ids, relevant_ids),
                "ndcg_at_5": ndcg_at_k(baseline_ids, relevant_ids, 5),
            }
        )
        refined_rows.append(
            {
                "query_id": query_id,
                "query": query,
                "retrieved_doc_ids": refined_ids,
                "p_at_3": precision_at_k(refined_ids, relevant_ids, 3),
                "p_at_5": precision_at_k(refined_ids, relevant_ids, 5),
                "ap": average_precision(refined_ids, relevant_ids),
                "ndcg_at_5": ndcg_at_k(refined_ids, relevant_ids, 5),
            }
        )

    baseline_summary = {
        "p_at_3": _mean_metric(baseline_rows, "p_at_3"),
        "p_at_5": _mean_metric(baseline_rows, "p_at_5"),
        "map": _mean_metric(baseline_rows, "ap"),
        "ndcg_at_5": _mean_metric(baseline_rows, "ndcg_at_5"),
    }
    refined_summary = {
        "p_at_3": _mean_metric(refined_rows, "p_at_3"),
        "p_at_5": _mean_metric(refined_rows, "p_at_5"),
        "map": _mean_metric(refined_rows, "ap"),
        "ndcg_at_5": _mean_metric(refined_rows, "ndcg_at_5"),
    }

    deltas = {
        key: refined_summary[key] - baseline_summary[key]
        for key in baseline_summary
    }

    return {
        "query_count": len(refined_rows),
        "top_k": top_k,
        "baseline": {
            "system": "tfidf+lsi",
            "summary": baseline_summary,
            "queries": baseline_rows,
        },
        "refined": {
            "system": "tfidf+lsi+rerank+threshold",
            "summary": refined_summary,
            "queries": refined_rows,
        },
        "delta": deltas,
    }


def _print_report(report: dict[str, Any]) -> None:
    print("Evaluacion REC-01")
    print(f"- Consultas: {report['query_count']}")
    print(f"- top_k: {report['top_k']}")

    for section in ("baseline", "refined"):
        summary = report[section]["summary"]
        print(f"\n{report[section]['system']}")
        print(f"- P@3: {summary['p_at_3']:.4f}")
        print(f"- P@5: {summary['p_at_5']:.4f}")
        print(f"- MAP: {summary['map']:.4f}")
        print(f"- NDCG@5: {summary['ndcg_at_5']:.4f}")

    print("\nDelta refinado - baseline")
    for key, value in report["delta"].items():
        print(f"- {key}: {value:+.4f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evalua REC-01: baseline LSI vs recuperador refinado.")
    parser.add_argument("--qrels", default=str(DEFAULT_QRELS_PATH), help="Ruta al archivo JSON de consultas relevantes.")
    parser.add_argument("--top-k", type=int, default=5, help="Cantidad maxima de resultados a evaluar por consulta.")
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT_PATH), help="Ruta de salida para guardar el reporte JSON.")
    args = parser.parse_args(argv)

    qrels_path = Path(args.qrels)
    if not qrels_path.exists():
        raise FileNotFoundError(f"No se encontro archivo de evaluacion: {qrels_path}")

    missing = missing_lsi_artifacts()
    if missing:
        print("Faltan archivos de indice LSI/TF-IDF:")
        for path in missing:
            print(f"- {path}")
        print("Primero debes construir el TF-IDF y entrenar el LSI.")
        print("Ejecuta: python main.py lsi_train")
        return 1

    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    searcher = SemanticSearcher()
    report = evaluate_searcher(searcher, qrels, top_k=args.top_k)

    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_report(report)
    print(f"\nReporte JSON: {report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
