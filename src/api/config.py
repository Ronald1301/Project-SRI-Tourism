from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_TFIDF_MATRIX = str(PROJECT_ROOT / "data/index/tfidf_matrix.npy")
DEFAULT_TFIDF_VOCAB = str(PROJECT_ROOT / "data/index/vocabulary.json")
DEFAULT_TFIDF_META = str(PROJECT_ROOT / "data/index/tfidf_meta.json")
DEFAULT_LSI_MODEL = str(PROJECT_ROOT / "data/index/lsi_model.pkl")
DEFAULT_LSI_VECTORS = str(PROJECT_ROOT / "data/index/doc_vectors.npy")
DEFAULT_LSI_META = str(PROJECT_ROOT / "data/index/lsi_metadata.json")
DEFAULT_DOCUMENTS = str(PROJECT_ROOT / "data/processed/documents.jsonl")
VECTOR_DB_DIR = str(PROJECT_ROOT / "data/processed/vector_db")

print(PROJECT_ROOT)