from __future__ import annotations

from typing import List, Tuple

import numpy as np

from indexing.tfidf_index import TFIDFIndex
from preprocessing.pipeline import PreprocessingPipeline
from retrieval.lsi_model import LSIModel

DEFAULT_TFIDF_MATRIX = "data/index/tfidf_matrix.npy"
DEFAULT_TFIDF_VOCAB = "data/index/vocabulary.json"
DEFAULT_TFIDF_META = "data/index/tfidf_meta.json"

DEFAULT_LSI_MODEL = "data/index/lsi_model.pkl"
DEFAULT_LSI_VECTORS = "data/index/doc_vectors.npy"
DEFAULT_LSI_META = "data/index/lsi_metadata.json"


def cosine_similarity(query_vector: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    if doc_matrix is None or doc_matrix.size == 0:
        return np.array([], dtype=float)

    query_vector = np.asarray(query_vector, dtype=float).reshape(-1)
    doc_matrix = np.asarray(doc_matrix, dtype=float)

    doc_norms = np.linalg.norm(doc_matrix, axis=1)
    query_norm = np.linalg.norm(query_vector)
    denom = doc_norms * query_norm

    sims = np.zeros(doc_matrix.shape[0], dtype=float)
    valid = denom > 0
    if np.any(valid):
        sims[valid] = (doc_matrix[valid] @ query_vector) / denom[valid]
    return sims


class SemanticSearcher:
    def __init__(
        self,
        *,
        language: str = "spanish",
        tfidf_matrix_path: str = DEFAULT_TFIDF_MATRIX,
        tfidf_vocab_path: str = DEFAULT_TFIDF_VOCAB,
        tfidf_meta_path: str = DEFAULT_TFIDF_META,
        lsi_model_path: str = DEFAULT_LSI_MODEL,
        lsi_vectors_path: str = DEFAULT_LSI_VECTORS,
        lsi_meta_path: str = DEFAULT_LSI_META,
    ) -> None:
        self.pipeline = PreprocessingPipeline(language=language)

        self.tfidf_index = TFIDFIndex.load(
            tfidf_matrix_path,
            tfidf_vocab_path,
            tfidf_meta_path,
        )

        self.lsi_model = LSIModel.load(
            model_path=lsi_model_path,
            vectors_path=lsi_vectors_path,
            metadata_path=lsi_meta_path,
        )

        if self.lsi_model.doc_vectors is None:
            raise ValueError("LSI document vectors are missing. Train and save LSI first.")

        if self.tfidf_index.doc_ids:
            self.doc_ids = list(self.tfidf_index.doc_ids)
        else:
            self.doc_ids = [str(i) for i in range(self.lsi_model.doc_vectors.shape[0])]

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if not query or not query.strip():
            return []

        tokens = self.pipeline.process_text(query)
        query_vector = self.tfidf_index.vectorize_query(tokens)
        query_lsi = self.lsi_model.transform_query(query_vector)

        similarities = cosine_similarity(query_lsi, self.lsi_model.doc_vectors)
        if similarities.size == 0:
            return []

        top_k = max(int(top_k), 0)
        if top_k == 0:
            return []

        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self.doc_ids[i], float(similarities[i])) for i in top_indices]
