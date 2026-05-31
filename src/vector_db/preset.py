from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.vector_db.vector_store import VectorDatabase

CRAWL_DOCUMENTS_PATH = Path("data/raw/documents.jsonl")

OUTPUT_DIR = Path("data/processed/vector_db")
TEXT_FIELDS = ["title", "content_text"]
ID_FIELD = "doc_id"
STORE_FIELDS = ["url", "title", "summary", "content_type", "rating", "review_date", "location"]
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 32
NORMALIZE_EMBEDDINGS = True
SHOW_PROGRESS_BAR = True
CHUNK_SIZE = 120
CHUNK_OVERLAP = 30
FAISS_METRIC = "ip"
FAISS_INDEX_TYPE = "hnsw"
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64

def resolve_documents_path() -> Path:
    """Resuelve la ruta del corpus JSONL base para construir la base vectorial.

    Returns:
        Path: Ruta existente al archivo de documentos.

    Raises:
        FileNotFoundError: Si el corpus inicial no existe.
    """
    if CRAWL_DOCUMENTS_PATH.exists():
        return CRAWL_DOCUMENTS_PATH
    raise FileNotFoundError(
        f"No se encontro {CRAWL_DOCUMENTS_PATH}. Ejecuta el crawler primero."
    )

def build_vector_db_from_preset() -> "VectorDatabase":
    """Construye y persiste la base vectorial usando la configuracion por defecto.

    Returns:
        VectorDatabase: Instancia construida y guardada en `OUTPUT_DIR`.

    Raises:
        FileNotFoundError: Si no existe el corpus de entrada.
        ValueError: Si el corpus no contiene texto utilizable.
    """
    from src.vector_db.vector_store import VectorDatabase

    jsonl_path = resolve_documents_path()
    db = VectorDatabase.build_from_jsonl(
        jsonl_path = jsonl_path,
        text_fields = TEXT_FIELDS,
        id_field = ID_FIELD,
        store_fields = STORE_FIELDS,
        model_name = MODEL_NAME,
        batch_size = BATCH_SIZE,
        normalize_embeddings = NORMALIZE_EMBEDDINGS,
        show_progress_bar = SHOW_PROGRESS_BAR,
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        faiss_metric = FAISS_METRIC,
        faiss_index_type = FAISS_INDEX_TYPE,
        hnsw_m = HNSW_M,
        hnsw_ef_construction = HNSW_EF_CONSTRUCTION,
        hnsw_ef_search = HNSW_EF_SEARCH,
    )
    db.save(OUTPUT_DIR)
    return db
