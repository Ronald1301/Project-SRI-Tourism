from __future__ import annotations

import argparse
from pathlib import Path

from src.indexing.tfidf_index import TFIDFIndex
from src.preprocessing.pipeline import process_all_sources
from src.retrieval.lsi_model import LSIModel
from src.retrieval.search import (
    DEFAULT_LSI_META,
    DEFAULT_LSI_MODEL,
    DEFAULT_LSI_VECTORS,
    DEFAULT_TFIDF_MATRIX,
    DEFAULT_TFIDF_META,
    DEFAULT_TFIDF_VOCAB,
    SemanticSearcher,
)
from src.vector_db.preset import OUTPUT_DIR, build_vector_db_from_preset, resolve_documents_path
from src.vector_db.vector_store import VectorDatabase
from src.web_crawler import WebCrawler, build_default_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sri-tourism",
        description="Pipeline basico para probar el proyecto.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("crawl", help="Ejecuta crawling + scraping con el preset.")
    subparsers.add_parser("vectordb", help="Construye la base de datos vectorial inicial.")
    subparsers.add_parser("pipeline", help="Ejecuta crawling y luego vector DB.")
    query_parser = subparsers.add_parser("query", help="Consulta la base vectorial.")
    query_parser.add_argument("query", nargs="+", help="Texto de la consulta.")
    query_parser.add_argument("--top-k", type=int, default=5, help="Cantidad de resultados.")
    subparsers.add_parser("lsi_train", help="Entrena y guarda TF-IDF + LSI.")
    lsi_parser = subparsers.add_parser("lsi_query", help="Consulta el modelo LSI.")
    lsi_parser.add_argument("query", nargs="+", help="Texto de la consulta.")
    lsi_parser.add_argument("--top-k", type=int, default=5, help="Cantidad de resultados.")
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


def _run_vector_db() -> int:
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


def _run_pipeline() -> int:
    code = _run_crawl()
    if code != 0:
        return code
    return _run_vector_db()


def _run_vector_db_query(query_text: str, top_k: int) -> int:
    output_dir = Path(OUTPUT_DIR)
    try:
        db = VectorDatabase.load(output_dir)
    except FileNotFoundError:
        print(f"No se encontro base vectorial en {output_dir}. Ejecuta: python3 main.py vectordb")
        return 1

    results = db.search(query_text, top_k=top_k)
    print(f"Resultados para: {query_text}")
    if not results:
        print("- Sin resultados")
        return 0

    for idx, item in enumerate(results, start=1):
        title = item.get("title") or item.get("entity_name") or ""
        url = item.get("url") or ""
        score = item.get("score", 0.0)
        print(f"{idx}. score={score:.4f}  {title}")
        if url:
            print(f"   {url}")
    return 0

def _run_lsi_query(query_text: str, top_k: int) -> int:
    required_files = [
        DEFAULT_TFIDF_MATRIX,
        DEFAULT_TFIDF_VOCAB,
        DEFAULT_TFIDF_META,
        DEFAULT_LSI_MODEL,
        DEFAULT_LSI_VECTORS,
        DEFAULT_LSI_META,
    ]
    missing = [path for path in required_files if not Path(path).exists()]
    if missing:
        print("Faltan archivos de indice LSI/TF-IDF:")
        for path in missing:
            print(f"- {path}")
        print("Primero debes construir el TF-IDF y entrenar el LSI.")
        return 1

    searcher = SemanticSearcher()
    results = searcher.search(query_text, top_k=top_k)
    print(f"Resultados LSI para: {query_text}")
    if not results:
        print("- Sin resultados")
        return 0

    for idx, (doc_id, score) in enumerate(results, start=1):
        print(f"{idx}. score={score:.4f}  doc_id={doc_id}")
    return 0


def _run_lsi_train() -> int:
    raw_dir = resolve_documents_path()
    processed_dir = Path("data/processed/lsi_training")
    language = "spanish"
    n_components = 100

    documents = process_all_sources(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        language=language,
    )
    if not documents:
        print("No se encontraron documentos para entrenar LSI.")
        return 1

    tfidf = TFIDFIndex()
    tfidf.build(documents)
    tfidf.save(DEFAULT_TFIDF_MATRIX, DEFAULT_TFIDF_VOCAB, DEFAULT_TFIDF_META)

    lsi = LSIModel(n_components=n_components)
    lsi.train(tfidf.matrix)
    lsi.save(DEFAULT_LSI_MODEL, DEFAULT_LSI_VECTORS, DEFAULT_LSI_META)

    print("LSI entrenado y guardado")
    print(f"- Documentos: {len(tfidf.doc_ids)}")
    print(f"- Vocabulario: {len(tfidf.vocabulary)}")
    print(f"- Componentes LSI: {n_components}")
    print(f"- TF-IDF matrix: {DEFAULT_TFIDF_MATRIX}")
    print(f"- LSI model: {DEFAULT_LSI_MODEL}")
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "crawl":
        return _run_crawl()
    if args.command == "vectordb":
        return _run_vector_db()
    if args.command == "pipeline":
        return _run_pipeline()
    if args.command == "query":
        return _run_vector_db_query(" ".join(args.query), args.top_k)
    if args.command == "lsi_train":
        return _run_lsi_train()
    if args.command == "lsi_query":
        return _run_lsi_query(" ".join(args.query), args.top_k)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
