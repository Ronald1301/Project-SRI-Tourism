"""
TF-IDF index implementation.

Builds a document-term matrix using:
TF = alpha + (1 - alpha) * (freq / max_freq)
IDF = log(N / df)
"""

import math
import os

import numpy as np

from utils.file_manager import load_json, load_numpy, save_json, save_numpy


class TFIDFIndex:
    def __init__(self, alpha=0.5, log_base=None):
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if log_base is not None and (log_base <= 0.0 or log_base == 1.0):
            raise ValueError("log_base must be positive and != 1")

        self.alpha = float(alpha)
        self.log_base = log_base
        self.vocabulary = {}
        self.doc_ids = []
        self.doc_id_to_index = {}
        self.matrix = None
        self.idf = None

    def build(self, documents):
        """
        Build TF-IDF matrix and vocabulary.

        documents: {doc_id: [tokens]}
        """
        self.vocabulary = {}
        self.doc_ids = []
        self.doc_id_to_index = {}
        self.matrix = None
        self.idf = None

        if not documents:
            self.matrix = np.zeros((0, 0), dtype=float)
            self.idf = np.zeros(0, dtype=float)
            return self

        self.doc_ids = list(documents.keys())
        self.doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
        num_docs = len(self.doc_ids)

        term_df = {}
        doc_term_freqs = []
        doc_max_freqs = []

        for doc_id in self.doc_ids:
            tokens = documents.get(doc_id, [])
            term_freq = {}
            if tokens:
                for token in tokens:
                    if token is None:
                        continue
                    term_freq[token] = term_freq.get(token, 0) + 1
            doc_term_freqs.append(term_freq)
            doc_max_freqs.append(max(term_freq.values()) if term_freq else 0)
            for term in term_freq.keys():
                term_df[term] = term_df.get(term, 0) + 1

        terms = sorted(term_df.keys())
        self.vocabulary = {term: idx for idx, term in enumerate(terms)}
        vocab_size = len(terms)

        self.idf = np.zeros(vocab_size, dtype=float)
        if num_docs > 0:
            for term, df in term_df.items():
                self.idf[self.vocabulary[term]] = self._idf(num_docs, df)

        self.matrix = np.zeros((num_docs, vocab_size), dtype=float)
        for row_idx, term_freq in enumerate(doc_term_freqs):
            if not term_freq:
                continue
            max_freq = doc_max_freqs[row_idx]
            for term, freq in term_freq.items():
                col_idx = self.vocabulary[term]
                self.matrix[row_idx, col_idx] = self._tf(freq, max_freq) * self.idf[col_idx]

        return self

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
            return np.zeros(len(self.vocabulary), dtype=float)

        term_freq = {}
        for token in tokens:
            if token is None:
                continue
            if token not in self.vocabulary:
                continue
            term_freq[token] = term_freq.get(token, 0) + 1

        if not term_freq:
            return np.zeros(len(self.vocabulary), dtype=float)

        max_freq = max(term_freq.values()) if term_freq else 0
        vector = np.zeros(len(self.vocabulary), dtype=float)
        for term, freq in term_freq.items():
            col_idx = self.vocabulary[term]
            vector[col_idx] = self._tf(freq, max_freq) * self.idf[col_idx]

        return vector

    def save(self, matrix_path, vocab_path, meta_path=None):
        """
        Save artifacts.
        - matrix_path: .npy file for TF-IDF matrix
        - vocab_path: .json file for vocabulary (term -> index)
        - meta_path: optional .json for doc_ids and idf
        """
        save_numpy(self.matrix, matrix_path)
        save_json(self.vocabulary, vocab_path)
        if meta_path:
            meta = {
                "doc_ids": self.doc_ids,
                "idf": self.idf.tolist() if self.idf is not None else [],
                "alpha": self.alpha,
                "log_base": self.log_base,
            }
            save_json(meta, meta_path)

    @classmethod
    def load(cls, matrix_path, vocab_path, meta_path=None):
        obj = cls()
        obj.matrix = load_numpy(matrix_path)
        obj.vocabulary = load_json(vocab_path)
        obj.doc_ids = []
        obj.doc_id_to_index = {}
        obj.idf = None

        if meta_path and os.path.exists(meta_path):
            meta = load_json(meta_path)
            obj.doc_ids = meta.get("doc_ids", [])
            obj.doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(obj.doc_ids)}
            idf_list = meta.get("idf", [])
            obj.idf = np.array(idf_list, dtype=float) if idf_list else None
            obj.alpha = float(meta.get("alpha", obj.alpha))
            obj.log_base = meta.get("log_base", obj.log_base)

        return obj
