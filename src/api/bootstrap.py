from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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
)
from src.vector_db.preset import OUTPUT_DIR, build_vector_db_from_preset, resolve_documents_path
from src.vector_db.vector_store import VectorDatabase

logger = logging.getLogger("src.api.bootstrap")

DEFAULT_LSI_LANGUAGE = "spanish"
DEFAULT_LSI_PROCESSED_DIR = Path("data/processed/lsi_training")
DEFAULT_LSI_REQUESTED_COMPONENTS = 100


def _missing_lsi_artifacts() -> list[str]:
    """Detecta que artefactos LSI/TF-IDF faltan para poder cargar el recuperador.

    Returns:
        list[str]: Lista de rutas faltantes. Si esta vacia, los artefactos estan disponibles.
    """
    tfidf_matrix_exists = Path(DEFAULT_TFIDF_MATRIX).exists() or Path(DEFAULT_TFIDF_MATRIX).with_suffix(".npy").exists()
    required_files = [
        DEFAULT_TFIDF_VOCAB,
        DEFAULT_TFIDF_META,
        DEFAULT_LSI_MODEL,
        DEFAULT_LSI_VECTORS,
        DEFAULT_LSI_META,
    ]
    missing = [path for path in required_files if not Path(path).exists()]
    if not tfidf_matrix_exists:
        missing.insert(0, DEFAULT_TFIDF_MATRIX)
    return missing


def ensure_vector_database(output_dir: Path = OUTPUT_DIR) -> tuple[VectorDatabase, bool]:
    """Garantiza que la base vectorial exista y pueda cargarse.

    Args:
        output_dir: Directorio donde se persiste la base vectorial.

    Returns:
        tuple[VectorDatabase, bool]: La base vectorial cargada o construida y un indicador de si fue reconstruida.

    Raises:
        FileNotFoundError: Si el corpus fuente no existe y la reconstruccion no es posible.
        ValueError: Si el corpus fuente no produce texto utilizable.
    """
    try:
        db = VectorDatabase.load(Path(output_dir))
        logger.info("Base vectorial encontrada en %s", output_dir)
        return db, False
    except (FileNotFoundError, ValueError) as exc:
        logger.info("Base vectorial ausente o invalida (%s). Reconstruyendo...", exc)

    db = build_vector_db_from_preset()
    if Path(output_dir) != OUTPUT_DIR:
        db.save(Path(output_dir))
    logger.info("Base vectorial reconstruida en %s", output_dir)
    return db, True


def build_lsi_artifacts(
    *,
    language: str = DEFAULT_LSI_LANGUAGE,
    processed_dir: Path = DEFAULT_LSI_PROCESSED_DIR,
    requested_components: int = DEFAULT_LSI_REQUESTED_COMPONENTS,
) -> dict[str, Any]:
    """Construye los artefactos TF-IDF + LSI usados por el recuperador semantico.

    Args:
        language: Idioma usado por el preprocesamiento.
        processed_dir: Directorio donde se guardan los artefactos intermedios.
        requested_components: Numero deseado de componentes latentes.

    Returns:
        dict[str, Any]: Resumen tecnico del entrenamiento realizado.

    Raises:
        FileNotFoundError: Si no existe el corpus base requerido para construir el indice.
        ValueError: Si no se obtienen documentos procesables.
    """
    raw_dir = resolve_documents_path().parent
    documents = process_all_sources(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        language=language,
    )
    if not documents:
        raise ValueError("No se encontraron documentos para entrenar LSI.")

    tfidf = TFIDFIndex()
    tfidf.build(documents)
    tfidf.save(DEFAULT_TFIDF_MATRIX, DEFAULT_TFIDF_VOCAB, DEFAULT_TFIDF_META)

    max_components = max(1, min(tfidf.matrix.shape[0], tfidf.matrix.shape[1]) - 1)
    n_components = min(int(requested_components), max_components)
    lsi = LSIModel(n_components=n_components)
    lsi.train(tfidf.matrix)
    lsi.save(DEFAULT_LSI_MODEL, DEFAULT_LSI_VECTORS, DEFAULT_LSI_META)

    return {
        "documents": len(tfidf.doc_ids),
        "vocabulary_size": len(tfidf.vocabulary),
        "tfidf_shape": tuple(int(value) for value in tfidf.matrix.shape),
        "min_df_effective": tfidf.effective_min_df,
        "max_df_effective": tfidf.effective_max_df,
        "lsi_components": n_components,
        "lsi_model_path": DEFAULT_LSI_MODEL,
        "tfidf_matrix_path": DEFAULT_TFIDF_MATRIX,
    }


def ensure_lsi_artifacts(
    *,
    language: str = DEFAULT_LSI_LANGUAGE,
    processed_dir: Path = DEFAULT_LSI_PROCESSED_DIR,
    requested_components: int = DEFAULT_LSI_REQUESTED_COMPONENTS,
) -> dict[str, Any]:
    """Garantiza que TF-IDF y LSI existan antes de iniciar la API.

    Args:
        language: Idioma usado por el preprocesamiento.
        processed_dir: Directorio donde se guardan los artefactos intermedios.
        requested_components: Numero deseado de componentes latentes.

    Returns:
        dict[str, Any]: Estado de la operacion y resumen del entrenamiento si se realizo.
    """
    missing = _missing_lsi_artifacts()
    if not missing:
        logger.info("Artefactos LSI/TF-IDF ya disponibles.")
        return {"built": False, "missing": []}

    logger.info("Faltan artefactos LSI/TF-IDF: %s", ", ".join(missing))
    summary = build_lsi_artifacts(
        language=language,
        processed_dir=processed_dir,
        requested_components=requested_components,
    )
    summary["built"] = True
    summary["missing"] = missing
    logger.info(
        "LSI entrenado y guardado | documentos=%d | vocabulario=%d | componentes=%d",
        summary["documents"],
        summary["vocabulary_size"],
        summary["lsi_components"],
    )
    return summary


def ensure_api_artifacts(
    *,
    output_dir: Path = OUTPUT_DIR,
    language: str = DEFAULT_LSI_LANGUAGE,
    processed_dir: Path = DEFAULT_LSI_PROCESSED_DIR,
    requested_components: int = DEFAULT_LSI_REQUESTED_COMPONENTS,
) -> dict[str, Any]:
    """Garantiza los artefactos de recuperacion requeridos por la API.

    Args:
        output_dir: Directorio de la base vectorial.
        language: Idioma del preprocesamiento para TF-IDF/LSI.
        processed_dir: Directorio intermedio para el entrenamiento LSI.
        requested_components: Numero objetivo de componentes latentes.

    Returns:
        dict[str, Any]: Resumen de la inicializacion realizada.
    """
    vector_db, vector_db_built = ensure_vector_database(output_dir=output_dir)
    lsi_summary = ensure_lsi_artifacts(
        language=language,
        processed_dir=processed_dir,
        requested_components=requested_components,
    )
    return {
        "vector_db_built": vector_db_built,
        "vector_db_doc_count": len(vector_db.doc_ids),
        "lsi": lsi_summary,
    }
