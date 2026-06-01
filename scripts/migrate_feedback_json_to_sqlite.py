from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.expantion.config import ConfigLoader
from src.expantion.constants import DEFAULT_FEEDBACK_PATH
from src.expantion.feedback_database import FeedbackDatabase


def migrate(json_path: Path, sqlite_path: Path | None = None) -> dict[str, int]:
    config = ConfigLoader().load()
    db = FeedbackDatabase(sqlite_path or config.feedback_db_path, config)
    if not json_path.exists():
        return {"explicit": 0, "implicit": 0}

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    explicit_count = 0
    implicit_count = 0

    for item in payload.get("explicit") or []:
        db.add_explicit(
            query=str(item.get("query") or ""),
            doc_id=str(item.get("doc_id") or ""),
            relevance=int(item.get("relevance") or 0),
            expanded_query=item.get("expanded_query"),
            search_mode=item.get("search_mode"),
        )
        explicit_count += 1

    for item in payload.get("implicit") or []:
        _, counted = db.add_implicit(
            query=str(item.get("query") or ""),
            doc_id=str(item.get("doc_id") or ""),
            event=str(item.get("event") or item.get("event_group") or "view_result"),
            search_mode=item.get("search_mode"),
        )
        implicit_count += 1 if counted else 0

    return {"explicit": explicit_count, "implicit": implicit_count}


def main() -> int:
    parser = argparse.ArgumentParser(description="Migra feedback JSON del modulo de expansion a SQLite.")
    parser.add_argument("--json-path", default=str(DEFAULT_FEEDBACK_PATH))
    parser.add_argument("--sqlite-path", default=None)
    args = parser.parse_args()

    summary = migrate(
        Path(args.json_path),
        Path(args.sqlite_path) if args.sqlite_path else None,
    )
    print(f"Migracion completa: explicit={summary['explicit']} implicit={summary['implicit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
