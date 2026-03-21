from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

class CrawlStorage:
    def __init__(self,output_dir : Path, run_id : str | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.run_id = run_id or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        self.html_dir = self.output_dir / "html" / self.run_id
        self.structured_dir = self.output_dir / "structured" / self.run_id
        self.documents_path = self.structured_dir / "documents.jsonl"
        self.errors_path = self.structured_dir / "errors.jsonl"
        self.report_path = self.structured_dir / "crawl_report.json"

        self.html_dir.mkdir(parents=True,exist_ok=True)
        self.structured_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def safe_name(value : str) -> str:
        return "".join(c if c.isalnum() or c in {"_","-"} else "_" for c in value).strip("_")

    def save_html(self, url : str , html : str , doc_id : str) -> str:
        host = self.safe_name(urlparse(url).netloc.lower() or "unknown_host")
        filename = f"{host}_{doc_id}.html"
        file_path = self.html_dir / filename
        file_path.write_text(html,encoding="utf-8",errors="ignore")
        return str(file_path)
    
    def append_document(self , document : dict[str,object]) -> None:
        with self.documents_path.open("a",encoding="utf-8") as file:
            file.write(json.dumps(document,ensure_ascii=False))
            file.write("\n")
    
    def append_error(self, error : dict[str,object]) -> None:
        with self.errors_path.open("a",encoding="utf-8") as file:
            file.write(json.dumps(error,ensure_ascii=False))
            file.write("\n")
    
    def save_report(self , report : dict[str , object]) -> str:
        self.report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        return str(self.report_path)