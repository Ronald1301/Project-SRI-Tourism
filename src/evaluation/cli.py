from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.evaluation.constants import (
    DEFAULT_MARKDOWN_REPORT_PATH,
    DEFAULT_QRELS_PATH,
    DEFAULT_REPORT_PATH,
)
from src.evaluation.experiment_runner import evaluate_systems
from src.evaluation.qrels import load_qrels
from src.evaluation.report_generator import write_reports


def print_report(report: dict[str, Any]) -> None:
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
        print(f"  Precision@3: {summary['precision_at_3']:.4f}")
        print(f"  Precision@5: {summary['precision_at_5']:.4f}")
        print(f"  Recall@5: {summary['recall_at_5']:.4f}")
        print(f"  Recall@10: {summary['recall_at_10']:.4f}")
        print(f"  MAP: {summary['map']:.4f}")
        print(f"  NDCG@5: {summary['ndcg_at_5']:.4f}")
        print(f"  MRR: {summary['mrr']:.4f}")
        print(f"  Precision@k: {summary['precision_at_k']:.4f}")
        print(f"  Recall@k: {summary['recall_at_k']:.4f}")
        print(f"  F1@k: {summary['f1_at_k']:.4f}")

    baseline_name = report.get("baseline_system")
    deltas = report.get("delta_vs_baseline", {})
    if baseline_name and deltas:
        print("")
        print("Comparacion con baseline:")
        print(f"  baseline: {baseline_name}")
        for name, values in deltas.items():
            print(
                f"  {name}: "
                f"Delta MAP={float(values.get('map', 0.0)):+.4f}, "
                f"Delta NDCG@5={float(values.get('ndcg_at_5', 0.0)):+.4f}, "
                f"Delta MRR={float(values.get('mrr', values.get('mrr_at_k', 0.0))):+.4f}"
            )

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
    print_report(report)
    print(f"\nReporte JSON: {args.report_out}")
    print(f"Reporte Markdown: {args.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
