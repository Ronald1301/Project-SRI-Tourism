import math
import os
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.utils.file_manager import load_json, load_numpy, save_json


class TFIDFIndex:
    def __init__(
        self,
        alpha=0.5,
        log_base=None,
        *,
        min_df: int = 2,
        max_df: float = 0.8,
        max_features: int | None = 50000,
        dtype=np.float32,
    ):
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if log_base is not None and (log_base <= 0.0 or log_base == 1.0):
            raise ValueError("log_base must be positive and != 1")

        self.alpha = float(alpha)
        self.log_base = log_base
        self.min_df = max(int(min_df), 1)
        self.max_df = max_df
        self.max_features = max_features
        self.dtype = dtype
        self.vocabulary = {}
        self.doc_ids = []
        self.doc_id_to_index = {}
        self.matrix = None
        self.idf = None
        self.effective_min_df = self.min_df
        self.effective_max_df = self.max_df

    def build(self, documents):
        """
        Build a sparse TF-IDF matrix with sklearn's TfidfVectorizer.

        documents: {doc_id: [tokens]}
        """
        self.vocabulary = {}
        self.doc_ids = []
        self.doc_id_to_index = {}
        self.matrix = None
        self.idf = None

        if not documents:
            self.matrix = sparse.csr_matrix((0, 0), dtype=self.dtype)
            self.idf = np.zeros(0, dtype=self.dtype)
            return self

        self.doc_ids = list(documents.keys())
        self.doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
        num_docs = len(self.doc_ids)
        corpus = [
            " ".join(str(token) for token in documents.get(doc_id, []) if token)
            for doc_id in self.doc_ids
        ]

        self.effective_min_df = min(self.min_df, max(num_docs, 1))
        self.effective_max_df = self.max_df
        vectorizer = self._build_vectorizer(self.effective_min_df, self.effective_max_df)
        try:
            matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            # Very small corpora can be fully pruned by min_df/max_df. Keep the
            # production defaults, but fall back instead of crashing training.
            self.effective_min_df = 1
            self.effective_max_df = 1.0
            vectorizer = self._build_vectorizer(self.effective_min_df, self.effective_max_df)
            matrix = vectorizer.fit_transform(corpus)

        self.matrix = matrix.astype(self.dtype).tocsr()
        self.vocabulary = {str(term): int(index) for term, index in vectorizer.vocabulary_.items()}
        self.idf = vectorizer.idf_.astype(self.dtype, copy=False)

        return self

    def _build_vectorizer(self, min_df, max_df):
        return TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            lowercase=False,
            min_df=min_df,
            max_df=max_df,
            max_features=self.max_features,
            dtype=self.dtype,
            norm="l2",
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
        )

    def _tf(self, freq, max_freq):
        if max_freq <= 0:
            return 0.0
        return self.alpha + (1.0 - self.alpha) * (float(freq) / float(max_freq))

    def _idf(self, num_docs, df):
        if df == 0:
            return 0.0
        ratio = num_docs / df
        if self.log_base is None:
            return math.log(ratio)
        return math.log(ratio, self.log_base)

    def vectorize_query(self, tokens):
        """
        Vectorize a query using the same TF-IDF scheme as the index.

        tokens: list[str]
        returns: numpy array shape (vocab_size,)
        """
        if not self.vocabulary:
            raise ValueError("Vocabulary is empty. Build or load the index first.")
        if self.idf is None or len(self.idf) == 0:
            raise ValueError("IDF is missing. Load index metadata or rebuild the index.")

        if not tokens:
            return sparse.csr_matrix((1, len(self.vocabulary)), dtype=self.dtype)

        term_freq = {}
        for token in tokens:
            if token is None:
                continue
            if token not in self.vocabulary:
                continue
            term_freq[token] = term_freq.get(token, 0) + 1

        if not term_freq:
            return sparse.csr_matrix((1, len(self.vocabulary)), dtype=self.dtype)

        rows = []
        cols = []
        data = []
        for term, freq in term_freq.items():
            col_idx = self.vocabulary[term]
            rows.append(0)
            cols.append(col_idx)
            data.append(float(freq) * float(self.idf[col_idx]))

        vector = sparse.csr_matrix(
            (np.asarray(data, dtype=self.dtype), (rows, cols)),
            shape=(1, len(self.vocabulary)),
            dtype=self.dtype,
        )
        return normalize(vector, norm="l2", copy=False)

    def save(self, matrix_path, vocab_path, meta_path=None):
        """
        Save artifacts.
        - matrix_path: .npz sparse TF-IDF matrix
        - vocab_path: .json file for vocabulary (term -> index)
        - meta_path: optional .json for doc_ids and idf
        """
        Path(matrix_path).parent.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(matrix_path, self.matrix.tocsr() if sparse.issparse(self.matrix) else sparse.csr_matrix(self.matrix))
        save_json(self.vocabulary, vocab_path)
        if meta_path:
            meta = {
                "doc_ids": self.doc_ids,
                "idf": self.idf.tolist() if self.idf is not None else [],
                "alpha": self.alpha,
                "log_base": self.log_base,
                "matrix_format": "csr",
                "dtype": str(np.dtype(self.dtype)),
                "min_df": self.min_df,
                "max_df": self.max_df,
                "max_features": self.max_features,
                "effective_min_df": self.effective_min_df,
                "effective_max_df": self.effective_max_df,
                "vectorizer": "sklearn.feature_extraction.text.TfidfVectorizer",
            }
            save_json(meta, meta_path)

    @classmethod
    def load(cls, matrix_path, vocab_path, meta_path=None):
        obj = cls()
        matrix_file = Path(matrix_path)
        if not matrix_file.exists() and matrix_file.suffix == ".npz":
            legacy_file = matrix_file.with_suffix(".npy")
            if legacy_file.exists():
                matrix_file = legacy_file

        if matrix_file.suffix == ".npz":
            obj.matrix = sparse.load_npz(matrix_file).astype(obj.dtype).tocsr()
        else:
            loaded = load_numpy(matrix_file)
            obj.matrix = sparse.csr_matrix(loaded, dtype=obj.dtype)
        obj.vocabulary = load_json(vocab_path)
        obj.doc_ids = []
        obj.doc_id_to_index = {}
        obj.idf = None

        if meta_path and os.path.exists(meta_path):
            meta = load_json(meta_path)
            obj.doc_ids = meta.get("doc_ids", [])
            obj.doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(obj.doc_ids)}
            idf_list = meta.get("idf", [])
            obj.idf = np.array(idf_list, dtype=obj.dtype) if idf_list else None
            obj.alpha = float(meta.get("alpha", obj.alpha))
            obj.log_base = meta.get("log_base", obj.log_base)
            obj.min_df = int(meta.get("min_df", obj.min_df))
            obj.max_df = meta.get("max_df", obj.max_df)
            obj.max_features = meta.get("max_features", obj.max_features)
            obj.effective_min_df = int(meta.get("effective_min_df", obj.min_df))
            obj.effective_max_df = meta.get("effective_max_df", obj.max_df)

        return obj
