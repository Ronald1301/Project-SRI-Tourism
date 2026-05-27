from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

_PATH_LOCKS: dict[str, Lock] = {}
_PATH_LOCKS_GUARD = Lock()


def _lock_for_path(path: Path) -> Lock:
    key = str(path.resolve())
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = Lock()
            _PATH_LOCKS[key] = lock
        return lock

class CrawlStorage:
    def __init__(self, output_dir: Path) -> None:
        self.documents_path = Path(output_dir) / "documents.jsonl"
        self.documents_path.parent.mkdir(parents=True, exist_ok=True)
        self._documents_lock = _lock_for_path(self.documents_path)

    def append_document(self, document: dict[str, object]) -> None:
        with self._documents_lock:
            with self.documents_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(document, ensure_ascii=False))
                file.write("\n")
