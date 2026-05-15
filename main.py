from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path
import subprocess
import sys


def _binary_supports_arm64(binary_path: Path) -> bool:
    try:
        output = subprocess.check_output(
            ["/usr/bin/lipo", "-archs", str(binary_path)],
            text=True,
        ).strip()
    except Exception:
        return False
    return "arm64" in output.split()


def _maybe_reexec_in_venv() -> None:
    project_dir = Path(__file__).resolve().parent
    venv_python = project_dir / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return

    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return

    current_machine = platform.machine().lower()
    if current_machine == "x86_64" and _binary_supports_arm64(venv_python):
        os.execv("/usr/bin/arch", ["/usr/bin/arch", "-arm64", str(venv_python), *sys.argv])

    os.execv(str(venv_python), [str(venv_python), *sys.argv])


_maybe_reexec_in_venv()

from src.indexing.tfidf_index import TFIDFIndex
from src.preprocessing.pipeline import process_all_sources
from src.retrieval.lsi_model import LSIModel
from src.retrieval.evaluate import DEFAULT_QRELS_PATH, DEFAULT_REPORT_PATH, evaluate_searcher
from src.retrieval.search import (
    DEFAULT_LSI_META,
    DEFAULT_LSI_MODEL,
    DEFAULT_LSI_VECTORS,
    DEFAULT_TFIDF_MATRIX,
    DEFAULT_TFIDF_META,
    DEFAULT_TFIDF_VOCAB,
    SemanticSearcher,
)
from src.web_crawler import WebCrawler, build_default_config


def _missing_lsi_artifacts() -> list[str]:
    required_files = [
        DEFAULT_TFIDF_MATRIX,
        DEFAULT_TFIDF_VOCAB,
        DEFAULT_TFIDF_META,
        DEFAULT_LSI_MODEL,
        DEFAULT_LSI_VECTORS,
        DEFAULT_LSI_META,
    ]
    return [path for path in required_files if not Path(path).exists()]


def _print_missing_lsi_artifacts() -> None:
    missing = _missing_lsi_artifacts()
    if not missing:
        return
    print("Faltan archivos de indice LSI/TF-IDF:")
    for path in missing:
        print(f"- {path}")
    print("Primero debes construir el TF-IDF y entrenar el LSI.")
    print("Ejecuta: python main.py lsi_train")


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
    rag_parser = subparsers.add_parser("rag_query", help="Consulta el RAG sobre la base vectorial.")
    rag_parser.add_argument("query", nargs="+", help="Texto de la consulta.")
    rag_parser.add_argument("--top-k", type=int, default=4, help="Cantidad de documentos recuperados.")
    rag_parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Muestra el prompt construido para la respuesta.",
    )
    lsi_rag_parser = subparsers.add_parser("lsi_rag", help="Consulta RAG usando recuperador LSI refinado.")
    lsi_rag_parser.add_argument("query", nargs="+", help="Texto de la consulta.")
    lsi_rag_parser.add_argument("--top-k", type=int, default=4, help="Cantidad de documentos recuperados.")
    lsi_rag_parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Muestra el prompt construido para la respuesta.",
    )
    subparsers.add_parser("lsi_train", help="Entrena y guarda TF-IDF + LSI.")
    lsi_parser = subparsers.add_parser("lsi_query", help="Consulta el modelo LSI.")
    lsi_parser.add_argument("query", nargs="+", help="Texto de la consulta.")
    lsi_parser.add_argument("--top-k", type=int, default=5, help="Cantidad de resultados.")
    web_search_parser = subparsers.add_parser("web_search", help="Prueba solo el modulo de busqueda web.")
    web_search_parser.add_argument("query", nargs="+", help="Texto de la consulta web.")
    web_search_parser.add_argument("--top-k", type=int, default=5, help="Cantidad de resultados web.")
    web_search_parser.add_argument(
        "--output",
        default="data/raw/web_search/documents.jsonl",
        help="Archivo JSONL donde guardar los documentos extraidos.",
    )
    eval_parser = subparsers.add_parser("evaluate_rec01", help="Compara baseline LSI vs recuperador refinado.")
    eval_parser.add_argument("--qrels", default=str(DEFAULT_QRELS_PATH), help="Archivo JSON con consultas y documentos relevantes.")
    eval_parser.add_argument("--top-k", type=int, default=5, help="Cantidad de resultados a evaluar por consulta.")
    eval_parser.add_argument("--report-out", default=str(DEFAULT_REPORT_PATH), help="Ruta donde guardar el reporte JSON.")
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
    from src.vector_db.preset import OUTPUT_DIR, build_vector_db_from_preset, resolve_documents_path

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
    from src.vector_db.preset import OUTPUT_DIR
    from src.vector_db.vector_store import VectorDatabase

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


def _run_rag_query(query_text: str, top_k: int, show_prompt: bool) -> int:
    from src.retrieval.rag_pipeline import RAGPipeline

    try:
        rag = RAGPipeline.from_preset()
    except FileNotFoundError as exc:
        print(str(exc))
        print("Ejecuta antes: python3 main.py vectordb")
        return 1
    except ValueError as exc:
        print(str(exc))
        return 1

    result = rag.answer_query(query_text, top_k=top_k)
    print(f"Respuesta RAG para: {query_text}")
    print(result.answer)
    print("")
    print("Fuentes:")
    for doc in result.documents:
        print(f"[{doc.citation_id}] score={doc.score:.4f}  {doc.title}")
        if doc.url:
            print(f"   {doc.url}")

    if show_prompt:
        print("")
        print("Prompt:")
        print(result.prompt)
    return 0


def _run_lsi_rag_query(query_text: str, top_k: int, show_prompt: bool) -> int:
    from src.retrieval.rag_pipeline import RAGPipeline

    missing = _missing_lsi_artifacts()
    if missing:
        _print_missing_lsi_artifacts()
        return 1

    searcher = SemanticSearcher()
    lsi_results = searcher.search(query_text, top_k=top_k)

    if not lsi_results:
        print(f"No se encontraron resultados LSI para: {query_text}")
        return 0

    try:
        rag = RAGPipeline.from_preset()
    except FileNotFoundError as exc:
        print(str(exc))
        print("Ejecuta antes: python3 main.py vectordb")
        return 1
    except ValueError as exc:
        print(str(exc))
        return 1

    result = rag.answer_with_lsi(query_text, lsi_results, top_k=top_k)
    print(f"Respuesta RAG+LSI para: {query_text}")
    print(result.answer)
    print("")
    print("Fuentes:")
    for doc in result.documents:
        print(f"[{doc.citation_id}] score={doc.score:.4f}  {doc.title}")
        if doc.url:
            print(f"   {doc.url}")

    if show_prompt:
        print("")
        print("Prompt:")
        print(result.prompt)
    return 0


def _run_lsi_query(query_text: str, top_k: int) -> int:
    missing = _missing_lsi_artifacts()
    if missing:
        _print_missing_lsi_artifacts()
        return 1

    searcher = SemanticSearcher()
    results = searcher.search(query_text, top_k=top_k)
    print(f"Resultados LSI para: {query_text}")
    if not results:
        print("- Sin resultados por encima del umbral de relevancia")
        return 0

    for item in results:
        rank = item.get("rank", 0)
        score = float(item.get("score", 0.0))
        lsi_score = float(item.get("lsi_score", 0.0))
        doc_id = item.get("doc_id", "")
        title = item.get("title") or "(sin titulo)"
        url = item.get("url") or ""
        snippet = item.get("snippet") or ""

        print(f"{rank}. score={score:.4f}  lsi={lsi_score:.4f}  {title}")
        print(f"   doc_id={doc_id}")
        if url:
            print(f"   {url}")
        if snippet:
            print(f"   snippet: {snippet}")
    return 0


def _run_web_search(query_text: str, top_k: int, output_path: str) -> int:
    from src.utils.file_manager import save_documents_to_jsonl
    from src.web_crawler import build_default_config
    from src.web_search import DuckDuckGoWebSearchClient

    crawler_config = build_default_config()
    client = DuckDuckGoWebSearchClient(visited_urls_path=crawler_config.visited_urls_path)
    documents = client.search(query_text, max_results=top_k)

    print(f"Resultados Web para: {query_text}")
    if not documents:
        print("- Sin documentos nuevos (posiblemente ya visitados o no accesibles)")
        return 0

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_documents_to_jsonl(documents, output_file)

    print(f"- Documentos extraidos: {len(documents)}")
    print(f"- Guardados en: {output_file}")
    return 0


def _run_lsi_train() -> int:
    from src.vector_db.preset import resolve_documents_path

    raw_dir = resolve_documents_path().parent
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


def _run_rec01_evaluation(qrels_path: str, top_k: int, report_out: str) -> int:
    path = Path(qrels_path)
    if not path.exists():
        print(f"No se encontro archivo de evaluacion: {path}")
        return 1

    missing = _missing_lsi_artifacts()
    if missing:
        _print_missing_lsi_artifacts()
        return 1

    import json

    qrels = json.loads(path.read_text(encoding="utf-8"))
    searcher = SemanticSearcher()
    report = evaluate_searcher(searcher, qrels, top_k=top_k)
    report_path = Path(report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Evaluacion REC-01")
    print(f"- Consultas: {report['query_count']}")
    print(f"- top_k: {report['top_k']}")

    for section in ("baseline", "refined"):
        summary = report[section]["summary"]
        print(f"{report[section]['system']}")
        print(f"  P@3: {summary['p_at_3']:.4f}")
        print(f"  P@5: {summary['p_at_5']:.4f}")
        print(f"  MAP: {summary['map']:.4f}")
        print(f"  NDCG@5: {summary['ndcg_at_5']:.4f}")

    print("Delta refinado - baseline")
    for key, value in report["delta"].items():
        print(f"  {key}: {value:+.4f}")
    print(f"Reporte JSON: {report_path}")
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
    if args.command == "rag_query":
        return _run_rag_query(" ".join(args.query), args.top_k, args.show_prompt)
    if args.command == "lsi_rag":
        return _run_lsi_rag_query(" ".join(args.query), args.top_k, args.show_prompt)
    if args.command == "lsi_train":
        return _run_lsi_train()
    if args.command == "lsi_query":
        return _run_lsi_query(" ".join(args.query), args.top_k)
    if args.command == "web_search":
        return _run_web_search(" ".join(args.query), args.top_k, args.output)
    if args.command == "evaluate_rec01":
        return _run_rec01_evaluation(args.qrels, args.top_k, args.report_out)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
