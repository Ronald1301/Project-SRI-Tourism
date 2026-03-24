from __future__ import annotations

import argparse

from src.vector_db.preset import OUTPUT_DIR, build_vector_db_from_preset, resolve_documents_path
from src.web_crawler import WebCrawler, build_default_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sri-tourism",
        description="Sistema de Recuperacion de Informacion (dominio turismo).",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("crawl", help="Ejecuta crawling + scraping con el preset.")
    subparsers.add_parser("vectordb", help="Construye la base de datos vectorial inicial")
    return parser


def _run_crawl() -> int:
    config = build_default_config()
    crawler = WebCrawler(config)
    report = crawler.crawl()

    stats = report["stats"]
    print("Crawling finalizado")
    print(f"- Run ID: {report['run_id']}")
    print(f"- Documentos guardados: {stats['documents_saved']}")
    print(f"- Paginas HTML procesadas: {stats['pages_fetched']}")
    print(f"- URLs visitadas: {stats['urls_visited']}")
    print(f"- Errores: {stats['errors']}")
    print(f"- Reporte: {report['paths']['report_json']}")
    return 0

def run_vector_db() -> int:
    try:
        source_path = resolve_documents_path()
        db = build_vector_db_from_preset()
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return 1
    
    print("Vector DB creada")
    print(f"- Source: {source_path}")
    print(f"- Documentos indexados: {len(db.doc_ids)}")
    print(f"- Output dir: {OUTPUT_DIR}")
    return 0

def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "crawl":
        return _run_crawl()
    if args.command == "vectordb":
        return run_vector_db()

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
