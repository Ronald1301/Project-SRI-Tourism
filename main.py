from __future__ import annotations

import argparse

from web_crawler import WebCrawler, build_default_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sri-tourism",
        description="Sistema de Recuperacion de Informacion (dominio turismo).",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("crawl", help="Ejecuta crawling + scraping con el preset.")
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


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "crawl":
        return _run_crawl()

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
