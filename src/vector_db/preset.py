from __future__ import annotations

from pathlib import Path

from src.vector_db.vector_store import VectorDatabase

DOCUMENTS_JSONL_PATH: Path | None = None
CRAWL_STRUCTURED_DIR = Path("data/raw/crawl/structured")
AUTO_DISCOVER_LATEST = True

OUTPUT_DIR = Path("data/processed/vector_db")
TEXT_FIELDS = ["title", "content_text"]
ID_FIELD = "doc_id"
STORE_FIELDS = ["url", "title", "summary", "content_type", "rating", "review_date", "location"]
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32
NORMALIZE_EMBEDDINGS = True
SHOW_PROGRESS_BAR = True

def find_lastest_jsonl(base_dir : Path) -> Path | None:
    if not base_dir.exists():
        return None
    candidates = []
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        jsonl_path = child / "documents.jsonl"
        if jsonl_path.exists():
            candidates.append(jsonl_path)
    if not candidates:
        return None
    return sorted(candidates,key=lambda p : p.parent.name)[-1]

def resolve_documents_path() -> Path:
    if DOCUMENTS_JSONL_PATH:
        path = Path(DOCUMENTS_JSONL_PATH)
        if path.exists():
            return path
    if AUTO_DISCOVER_LATEST:
        latest = find_lastest_jsonl(CRAWL_STRUCTURED_DIR)
        if latest:
            return latest
    raise FileNotFoundError(
            "No documents.jsonl found. Set DOCUMENTS_JSONL_PATH in vector_db/preset.py"
        )

def build_vector_db_from_preset() -> VectorDatabase:
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
    )
    db.save(OUTPUT_DIR)
    return db
