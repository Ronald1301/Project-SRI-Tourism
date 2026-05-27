from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

from src.web_crawler import WebCrawler, build_default_config
from src.web_crawler.sites import SITE_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

CONSOLIDATED_DOCUMENTS_PATH = Path("data/raw/documents.jsonl")
SHARED_VISITED_URLS_PATH = Path(__file__).resolve().parent / "visited_urls.txt"

def _run_site_crawler(site_name: str) -> dict[str, object]:
    config = build_default_config(site_name=site_name)
    config.output_dir = Path("data/raw")
    config.visited_urls_path = SHARED_VISITED_URLS_PATH

    crawler = WebCrawler(config, site_name=site_name)
    report = crawler.crawl()
    return {
        "site_name": site_name,
        "report": report,
    }


def main() -> int:
    site_names = sorted(SITE_REGISTRY.keys())
    if not site_names:
        print("No hay sitios registrados para crawling.")
        return 1

    CONSOLIDATED_DOCUMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONSOLIDATED_DOCUMENTS_PATH.touch(exist_ok=True)
    SHARED_VISITED_URLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SHARED_VISITED_URLS_PATH.touch(exist_ok=True)

    results: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=len(site_names)) as executor:
        futures: dict[Future[dict[str, object]], str] = {
            executor.submit(_run_site_crawler, site_name): site_name
            for site_name in site_names
        }
        for future in as_completed(futures):
            site_name = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append((site_name, str(exc)))

    print("Crawling multi-sitio finalizado")
    print(f"- Sitios ejecutados: {len(site_names)}")
    print(f"- Archivo consolidado: {CONSOLIDATED_DOCUMENTS_PATH}")
    print(f"- Archivo compartido visited: {SHARED_VISITED_URLS_PATH}")

    for item in sorted(results, key=lambda value: str(value["site_name"])):
        site_name = str(item["site_name"])
        report = item["report"]
        if isinstance(report, dict) and "stats" in report:
            stats = report["stats"]
            run_id = str(report.get("run_id", "n/d"))
            documents_path = str(report.get("paths", {}).get("documents_jsonl", CONSOLIDATED_DOCUMENTS_PATH))
            report_path = str(report.get("paths", {}).get("report_json", "n/d"))
        else:
            stats = report if isinstance(report, dict) else {}
            run_id = "n/d"
            documents_path = str(CONSOLIDATED_DOCUMENTS_PATH)
            report_path = "n/d"

        print(f"[{site_name}]")
        print(f"  - Run ID: {run_id}")
        print(f"  - Documentos guardados: {stats.get('documents_saved', 0)}")
        print(f"  - Paginas HTML procesadas: {stats.get('pages_fetched', 0)}")
        print(f"  - URLs visitadas: {stats.get('urls_visited', 0)}")
        print(f"  - Errores: {stats.get('errors', 0)}")
        print(f"  - documents.jsonl: {documents_path}")
        print(f"  - Reporte: {report_path}")

    if failures:
        print("Sitios con fallo:")
        for site_name, detail in failures:
            print(f"- {site_name}: {detail}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
