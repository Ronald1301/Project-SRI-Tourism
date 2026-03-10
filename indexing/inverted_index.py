"""
Inverted index implementation.

Structure: term -> {doc_id: frequency}
"""

from utils.file_manager import load_pickle, save_pickle


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.doc_ids = []
        self.doc_count = 0

    def build(self, documents):
        """
        Build the inverted index from tokenized documents.

        documents: {doc_id: [tokens]}
        """
        self.index = {}
        self.doc_ids = []
        self.doc_count = 0

        if not documents:
            return self

        for doc_id, tokens in documents.items():
            self.doc_ids.append(doc_id)
            self.doc_count += 1
            if not tokens:
                continue
            for token in tokens:
                if token is None:
                    continue
                postings = self.index.get(token)
                if postings is None:
                    postings = {}
                    self.index[token] = postings
                postings[doc_id] = postings.get(doc_id, 0) + 1

        return self

    def get_postings(self, term):
        """
        Return postings list as a list of (doc_id, frequency).
        """
        postings = self.index.get(term)
        if not postings:
            return []

        if self.doc_ids:
            return [(doc_id, postings[doc_id]) for doc_id in self.doc_ids if doc_id in postings]

        try:
            return sorted(postings.items(), key=lambda x: x[0])
        except TypeError:
            return list(postings.items())

    def save(self, path):
        data = {
            "index": self.index,
            "doc_ids": self.doc_ids,
            "doc_count": self.doc_count,
        }
        save_pickle(data, path)

    @classmethod
    def load(cls, path):
        data = load_pickle(path)
        obj = cls()
        obj.index = data.get("index", {})
        obj.doc_ids = data.get("doc_ids", [])
        obj.doc_count = data.get("doc_count", len(obj.doc_ids))
        return obj
