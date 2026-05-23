from __future__ import annotations

import json
from pathlib import Path

class CrawlStorage:
    def __init__(self, output_dir: Path) -> None:
        self.documents_path = Path(output_dir) / "documents.jsonl"
        self.documents_path.parent.mkdir(parents=True, exist_ok=True)

    def append_document(self, document: dict[str, object]) -> None:
        with self.documents_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(document, ensure_ascii=False))
            file.write("\n")