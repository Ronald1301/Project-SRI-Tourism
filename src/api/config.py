from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "data" / "config"
DEFAULT_TFIDF_MATRIX = str(PROJECT_ROOT / "data/index/tfidf_matrix.npz")
DEFAULT_TFIDF_VOCAB = str(PROJECT_ROOT / "data/index/vocabulary.json")
DEFAULT_TFIDF_META = str(PROJECT_ROOT / "data/index/tfidf_meta.json")
DEFAULT_LSI_MODEL = str(PROJECT_ROOT / "data/index/lsi_model.pkl")
DEFAULT_LSI_VECTORS = str(PROJECT_ROOT / "data/index/doc_vectors.npy")
DEFAULT_LSI_META = str(PROJECT_ROOT / "data/index/lsi_metadata.json")
DEFAULT_DOCUMENTS = str(PROJECT_ROOT / "data/processed/documents.jsonl")
VECTOR_DB_DIR = str(PROJECT_ROOT / "data/processed/vector_db")
DEFAULT_DOMAIN_LLM_MODEL = "phi3"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 300.0
DEFAULT_QUERY_EXPANSION_CONFIG = str(DEFAULT_CONFIG_DIR / "query_expansion.json")
DEFAULT_DOMAIN_DETECTION_CONFIG = str(DEFAULT_CONFIG_DIR / "domain_detection.json")
DEFAULT_EVALUATION_CONFIG = str(DEFAULT_CONFIG_DIR / "evaluation.json")
DEFAULT_EVALUATION_RESULTS_DIR = str(PROJECT_ROOT / "data/evaluation/results")
DEFAULT_DOMAIN_SYNONYMS = str(DEFAULT_CONFIG_DIR / "tourism_synonyms.json")
