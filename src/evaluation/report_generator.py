from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluation.constants import BASELINE_SYSTEM


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
                    fmt(summary.get("precision_at_k")),
                    fmt(summary.get("recall_at_k")),
                    fmt(summary.get("f1_at_k")),
                    fmt(summary.get("map")),
                    fmt(summary.get("mrr_at_k")),
                    fmt(summary.get("ndcg_at_k")),
                    fmt(summary.get("r_precision")),
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
                f"| {name} | {fmt_delta(values.get('f1_at_k'))} | "
                f"{fmt_delta(values.get('ndcg_at_k'))} | {fmt_delta(values.get('mrr_at_k'))} |"
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


def fmt(value: Any) -> str:
    return f"{float(value or 0.0):.4f}"


def fmt_delta(value: Any) -> str:
    return f"{float(value or 0.0):+.4f}"
