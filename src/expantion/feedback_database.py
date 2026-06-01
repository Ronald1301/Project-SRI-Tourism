from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.expantion.config import ExpansionConfig
from src.expantion.constants import IMPLICIT_EVENT_GROUPS, IMPLICIT_WEIGHTS
from src.expantion.text import normalize_key


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def query_hash(query: str) -> str:
    return hashlib.sha256(normalize_key(query).encode("utf-8")).hexdigest()


class FeedbackDatabase:
    def __init__(self, path: Path | str, config: ExpansionConfig | None = None) -> None:
        self.path = Path(path)
        self.config = config
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with closing(self._connect()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS feedback_explicit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT NOT NULL,
                    query_key TEXT NOT NULL,
                    query TEXT NOT NULL,
                    expanded_query TEXT,
                    doc_id TEXT NOT NULL,
                    relevance INTEGER NOT NULL,
                    search_mode TEXT,
                    timestamp TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS feedback_implicit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT NOT NULL,
                    query_key TEXT NOT NULL,
                    query TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_group TEXT NOT NULL,
                    event_weight REAL NOT NULL,
                    search_mode TEXT,
                    timestamp TEXT NOT NULL,
                    UNIQUE(query_hash, doc_id, event_group)
                );

                CREATE INDEX IF NOT EXISTS idx_explicit_query_doc
                    ON feedback_explicit(query_hash, doc_id);
                CREATE INDEX IF NOT EXISTS idx_explicit_query_date
                    ON feedback_explicit(query_hash, timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_implicit_query_doc
                    ON feedback_implicit(query_hash, doc_id);
                CREATE INDEX IF NOT EXISTS idx_implicit_query_date
                    ON feedback_implicit(query_hash, timestamp DESC);
                """
            )
            conn.commit()

    def add_explicit(
        self,
        *,
        query: str,
        doc_id: str,
        relevance: int,
        expanded_query: str | None = None,
        search_mode: str | None = None,
    ) -> dict[str, Any]:
        key = normalize_key(query)
        event = {
            "query": query,
            "query_key": key,
            "query_hash": query_hash(query),
            "expanded_query": expanded_query,
            "doc_id": str(doc_id or "").strip(),
            "relevance": int(relevance),
            "search_mode": search_mode,
            "timestamp": now_iso(),
        }
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                """
                INSERT INTO feedback_explicit
                    (query_hash, query_key, query, expanded_query, doc_id, relevance, search_mode, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event["query_hash"],
                    event["query_key"],
                    event["query"],
                    event["expanded_query"],
                    event["doc_id"],
                    event["relevance"],
                    event["search_mode"],
                    event["timestamp"],
                ),
            )
            event["id"] = cursor.lastrowid
            conn.commit()
        return event

    def add_implicit(
        self,
        *,
        query: str,
        doc_id: str,
        event: str,
        search_mode: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        key = normalize_key(query)
        event_group = IMPLICIT_EVENT_GROUPS.get(normalize_key(event), normalize_key(event))
        weight = self._implicit_weight(event_group)
        payload = {
            "query": query,
            "query_key": key,
            "query_hash": query_hash(query),
            "doc_id": str(doc_id or "").strip(),
            "event": event,
            "event_group": event_group,
            "weight": weight,
            "search_mode": search_mode,
            "timestamp": now_iso(),
        }
        with closing(self._connect()) as conn:
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO feedback_implicit
                        (query_hash, query_key, query, doc_id, event_type, event_group,
                         event_weight, search_mode, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        payload["query_hash"],
                        payload["query_key"],
                        payload["query"],
                        payload["doc_id"],
                        payload["event"],
                        payload["event_group"],
                        payload["weight"],
                        payload["search_mode"],
                        payload["timestamp"],
                    ),
                )
                payload["id"] = cursor.lastrowid
                conn.commit()
                return payload, True
            except sqlite3.IntegrityError:
                row = conn.execute(
                    """
                    SELECT * FROM feedback_implicit
                    WHERE query_hash = ? AND doc_id = ? AND event_group = ?
                    """,
                    (payload["query_hash"], payload["doc_id"], payload["event_group"]),
                ).fetchone()
                return self._row_to_implicit(row), False

    def query_feedback(self, query: str) -> dict[str, list[dict[str, Any]]]:
        q_hash = query_hash(query)
        with closing(self._connect()) as conn:
            explicit = [
                self._row_to_explicit(row)
                for row in conn.execute(
                    "SELECT * FROM feedback_explicit WHERE query_hash = ? ORDER BY timestamp DESC",
                    (q_hash,),
                ).fetchall()
            ]
            implicit = [
                self._row_to_implicit(row)
                for row in conn.execute(
                    "SELECT * FROM feedback_implicit WHERE query_hash = ? ORDER BY timestamp DESC",
                    (q_hash,),
                ).fetchall()
            ]
        return {"explicit": explicit, "implicit": implicit}

    def export_json(self) -> dict[str, list[dict[str, Any]]]:
        with closing(self._connect()) as conn:
            explicit = [self._row_to_explicit(row) for row in conn.execute("SELECT * FROM feedback_explicit")]
            implicit = [self._row_to_implicit(row) for row in conn.execute("SELECT * FROM feedback_implicit")]
        return {"explicit": explicit, "implicit": implicit}

    def _implicit_weight(self, event_group: str) -> float:
        if self.config is not None:
            if event_group == "source_interaction":
                return float(self.config.get("techniques.feedback.implicit_source_weight", 0.65))
            if event_group == "dwell":
                return float(self.config.get("techniques.feedback.implicit_dwell_weight", 0.35))
            return float(self.config.get("techniques.feedback.implicit_default_weight", 0.2))
        return float(IMPLICIT_WEIGHTS.get(event_group, 0.2))

    def _row_to_explicit(self, row: sqlite3.Row | None) -> dict[str, Any]:
        if row is None:
            return {}
        return {
            "id": row["id"],
            "query": row["query"],
            "query_key": row["query_key"],
            "query_hash": row["query_hash"],
            "expanded_query": row["expanded_query"],
            "doc_id": row["doc_id"],
            "relevance": row["relevance"],
            "search_mode": row["search_mode"],
            "timestamp": row["timestamp"],
        }

    def _row_to_implicit(self, row: sqlite3.Row | None) -> dict[str, Any]:
        if row is None:
            return {}
        return {
            "id": row["id"],
            "query": row["query"],
            "query_key": row["query_key"],
            "query_hash": row["query_hash"],
            "doc_id": row["doc_id"],
            "event": row["event_type"],
            "event_group": row["event_group"],
            "weight": row["event_weight"],
            "search_mode": row["search_mode"],
            "timestamp": row["timestamp"],
        }

    def dump_to_json_file(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.export_json(), ensure_ascii=False, indent=2), encoding="utf-8")
