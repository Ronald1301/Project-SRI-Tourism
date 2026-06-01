from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Permite ejecutar el script directamente desde la raiz del repo.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.retrieval.query_expansion import FeedbackStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Migra feedback legacy JSON a SQLite.")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/feedback/query_feedback.db"),
        help="Ruta del archivo SQLite de salida.",
    )
    parser.add_argument(
        "--legacy",
        type=Path,
        default=Path("data/feedback/query_feedback.json"),
        help="Ruta del JSON legado a importar.",
    )
    args = parser.parse_args()

    legacy_path = args.legacy
    if not legacy_path.exists():
        raise FileNotFoundError(f"No existe el JSON legado: {legacy_path}")

    payload = json.loads(legacy_path.read_text(encoding="utf-8"))
    store = FeedbackStore(args.db)

    explicit_rows = payload.get("explicit") or []
    implicit_rows = payload.get("implicit") or []
    explicit_added = 0
    implicit_added = 0

    for item in explicit_rows:
        store.add_explicit(
            query=str(item.get("query") or ""),
            doc_id=str(item.get("doc_id") or ""),
            relevance=int(item.get("relevance") or 0),
            expanded_query=item.get("expanded_query"),
            search_mode=item.get("search_mode"),
        )
        explicit_added += 1

    for item in implicit_rows:
        _, inserted = store.add_implicit(
            query=str(item.get("query") or ""),
            doc_id=str(item.get("doc_id") or ""),
            event=str(item.get("event") or ""),
            search_mode=item.get("search_mode"),
        )
        if inserted:
            implicit_added += 1

    print(f"SQLite listo en: {args.db}")
    print(f"Legacy importado desde: {legacy_path}")
    print(f"Registros explicit importados: {explicit_added}")
    print(f"Registros implicit importados: {implicit_added}")


if __name__ == "__main__":
    main()
