from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.file_manager import load_json, save_json

try:
    import faiss
except ImportError:  # pragma: no cover - dependencia opcional en tiempo de import
    faiss = None


def _require_faiss() -> None:
    if faiss is None:
        raise ImportError(
            "Faiss no esta instalado. Instala la dependencia con: pip install faiss-cpu"
        )


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if not text:
        return []
    words = text.split()
    if not words:
        return []
    if chunk_size <= 0:
        return [" ".join(words)]
    if chunk_overlap < 0:
        chunk_overlap = 0

    step = max(chunk_size - chunk_overlap, 1)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        end = start + chunk_size
        part = words[start:end]
        if not part:
            continue
        chunks.append(" ".join(part))
        if end >= len(words):
            break
    return chunks


@dataclass
class VectorDatabase:
    model_name: str
    embeddings: np.ndarray
    index_to_doc_id: list[str]
    doc_id_to_meta: dict[str, dict]
    text_fields: list[str]
    id_field: str
    normalize_embeddings: bool
    chunk_size: int
    chunk_overlap: int
    faiss_metric: str = "ip"
    faiss_index_type: str = "hnsw"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64
    output_dir: Path | None = None
    model: SentenceTransformer | None = field(default=None, repr=False)
    index: object | None = field(default=None, repr=False)

    EMBEDDINGS_FILENAME = "embeddings.npy"
    META_FILENAME = "meta.json"
    INDEX_FILENAME = "faiss.index"
    DOC_META_FILENAME = "doc_id_to_meta.json"
    INDEX_TO_DOC_FILENAME = "index_to_doc_id.json"
    FORMAT_VERSION = 3

    @property
    def doc_ids(self) -> list[str]:
        return list(self.doc_id_to_meta.keys())

    @classmethod
    def build_from_jsonl(
        cls,
        jsonl_path: Path,
        text_fields: Iterable[str],
        id_field: str = "doc_id",
        store_fields: Iterable[str] | None = None,
        min_text_length: int = 1,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = True,
        chunk_size: int = 120,
        chunk_overlap: int = 30,
        faiss_metric: str = "ip",
        faiss_index_type: str = "hnsw",
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 64,
    ) -> "VectorDatabase":
        _require_faiss()
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

        text_fields = [name for name in text_fields if name]
        if not text_fields:
            raise ValueError("text_fields must include at least one field")

        store_fields = list(store_fields or [])
        index_to_doc_id: list[str] = []
        chunk_texts: list[str] = []
        doc_id_to_meta: dict[str, dict] = {}

        for idx, doc in enumerate(iter_jsonl(jsonl_path)):
            doc_id = str(doc.get(id_field) or f"doc_{idx}")
            parts: list[str] = []
            for field_name in text_fields:
                value = doc.get(field_name)
                if value:
                    parts.append(str(value))
            text = " ".join(parts).strip()
            if len(text) < min_text_length:
                continue

            meta = {key: doc.get(key) for key in store_fields}
            meta[id_field] = doc_id
            doc_id_to_meta[doc_id] = meta

            chunks = _chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for chunk in chunks:
                if len(chunk) < min_text_length:
                    continue
                chunk_texts.append(chunk)
                index_to_doc_id.append(doc_id)

        if not chunk_texts:
            raise ValueError("No valid text chunks found to vectorize")

        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            chunk_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=False,
        )
        embeddings = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
        prepared = cls._prepare_vectors_for_index(embeddings, normalize_embeddings)
        index = cls._create_faiss_index(
            prepared.shape[1],
            faiss_metric,
            faiss_index_type=faiss_index_type,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_ef_search=hnsw_ef_search,
        )
        index.add(prepared)

        db = cls(
            model_name=model_name,
            embeddings=embeddings,
            index_to_doc_id=index_to_doc_id,
            doc_id_to_meta=doc_id_to_meta,
            text_fields=list(text_fields),
            id_field=id_field,
            normalize_embeddings=normalize_embeddings,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            faiss_metric=faiss_metric,
            faiss_index_type=faiss_index_type,
            hnsw_m=int(hnsw_m),
            hnsw_ef_construction=int(hnsw_ef_construction),
            hnsw_ef_search=int(hnsw_ef_search),
            index=index,
        )
        db._validate_state()
        return db

    def get_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def ensure_index(self) -> object:
        _require_faiss()
        if self.index is None:
            if self.embeddings.ndim != 2:
                raise ValueError("Embeddings must be 2D to rebuild FAISS index")
            index = self._create_faiss_index(
                self.embeddings.shape[1],
                self.faiss_metric,
                faiss_index_type=self.faiss_index_type,
                hnsw_m=self.hnsw_m,
                hnsw_ef_construction=self.hnsw_ef_construction,
                hnsw_ef_search=self.hnsw_ef_search,
            )
            vectors = self._prepare_vectors_for_index(self.embeddings, self.normalize_embeddings)
            if vectors.size > 0:
                index.add(vectors)
            self.index = index
        return self.index

    def save(self, output_dir: Path) -> None:
        _require_faiss()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._validate_state()

        embeddings_path = output_dir / self.EMBEDDINGS_FILENAME
        index_path = output_dir / self.INDEX_FILENAME
        meta_path = output_dir / self.META_FILENAME
        doc_meta_path = output_dir / self.DOC_META_FILENAME
        idx_to_doc_path = output_dir / self.INDEX_TO_DOC_FILENAME

        embeddings = np.ascontiguousarray(np.asarray(self.embeddings, dtype=np.float32))
        np.save(embeddings_path, embeddings, allow_pickle=False)
        save_json(self.doc_id_to_meta, str(doc_meta_path))
        save_json(self.index_to_doc_id, str(idx_to_doc_path))

        faiss.write_index(self.ensure_index(), str(index_path))

        meta = {
            "format_version": self.FORMAT_VERSION,
            "model_name": self.model_name,
            "id_field": self.id_field,
            "text_fields": self.text_fields,
            "normalize_embeddings": self.normalize_embeddings,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "faiss_metric": self.faiss_metric,
            "faiss_index_type": self.faiss_index_type,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_construction": self.hnsw_ef_construction,
            "hnsw_ef_search": self.hnsw_ef_search,
            "vector_count": int(embeddings.shape[0]),
            "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            "artifacts": {
                "embeddings": self.EMBEDDINGS_FILENAME,
                "faiss_index": self.INDEX_FILENAME,
                "doc_id_to_meta": self.DOC_META_FILENAME,
                "index_to_doc_id": self.INDEX_TO_DOC_FILENAME,
            },
            "created_at": datetime.now(UTC).isoformat(),
        }
        save_json(meta, str(meta_path))
        self.output_dir = output_dir

    @classmethod
    def load(cls, output_dir: Path) -> "VectorDatabase":
        _require_faiss()
        meta_path = output_dir / cls.META_FILENAME
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {meta_path}")
        meta = load_json(str(meta_path))

        artifacts = meta.get("artifacts") or {}
        embeddings_path = output_dir / str(artifacts.get("embeddings") or cls.EMBEDDINGS_FILENAME)
        index_path = output_dir / str(artifacts.get("faiss_index") or cls.INDEX_FILENAME)
        doc_meta_path = output_dir / str(artifacts.get("doc_id_to_meta") or cls.DOC_META_FILENAME)
        idx_to_doc_path = output_dir / str(artifacts.get("index_to_doc_id") or cls.INDEX_TO_DOC_FILENAME)

        if not embeddings_path.exists():
            raise FileNotFoundError(f"Missing embeddings file: {embeddings_path}")

        embeddings = np.load(embeddings_path, mmap_mode="r")
        normalize_embeddings = bool(meta.get("normalize_embeddings", True))
        faiss_metric = str(meta.get("faiss_metric", "ip"))
        faiss_index_type = str(meta.get("faiss_index_type", "hnsw"))
        hnsw_m = int(meta.get("hnsw_m", 32))
        hnsw_ef_construction = int(meta.get("hnsw_ef_construction", 200))
        hnsw_ef_search = int(meta.get("hnsw_ef_search", 64))
        id_field = str(meta.get("id_field", "doc_id"))

        if index_path.exists():
            index = faiss.read_index(str(index_path))
        else:
            index = cls._create_faiss_index(
                embeddings.shape[1],
                faiss_metric,
                faiss_index_type=faiss_index_type,
                hnsw_m=hnsw_m,
                hnsw_ef_construction=hnsw_ef_construction,
                hnsw_ef_search=hnsw_ef_search,
            )
            vectors = cls._prepare_vectors_for_index(embeddings, normalize_embeddings)
            if vectors.size > 0:
                index.add(vectors)

        doc_id_to_meta: dict[str, dict] = {}
        index_to_doc_id: list[str] = []

        if doc_meta_path.exists() and idx_to_doc_path.exists():
            doc_id_to_meta_raw = load_json(str(doc_meta_path))
            index_to_doc_raw = load_json(str(idx_to_doc_path))
            if isinstance(doc_id_to_meta_raw, dict):
                for key, value in doc_id_to_meta_raw.items():
                    if isinstance(value, dict):
                        doc_id_to_meta[str(key)] = value
                    else:
                        doc_id_to_meta[str(key)] = {}
            index_to_doc_id = [str(value) for value in index_to_doc_raw]
        else:
            # Backward compatibility for old format:
            # meta.json containing ["doc_ids"] and ["metadata"] aligned by index.
            old_doc_ids = [str(value) for value in meta.get("doc_ids", [])]
            old_metadata = list(meta.get("metadata", []))
            if old_doc_ids:
                index_to_doc_id = old_doc_ids
                for pos, doc_id in enumerate(old_doc_ids):
                    base_meta: dict = {}
                    if pos < len(old_metadata) and isinstance(old_metadata[pos], dict):
                        base_meta.update(old_metadata[pos])
                    base_meta.setdefault(id_field, doc_id)
                    doc_id_to_meta[doc_id] = base_meta
            elif embeddings.ndim == 2 and embeddings.shape[0] > 0:
                index_to_doc_id = [f"doc_{idx}" for idx in range(embeddings.shape[0])]
                for doc_id in index_to_doc_id:
                    doc_id_to_meta[doc_id] = {id_field: doc_id}

        db = cls(
            model_name=str(meta.get("model_name", "all-MiniLM-L6-v2")),
            embeddings=embeddings,
            index_to_doc_id=index_to_doc_id,
            doc_id_to_meta=doc_id_to_meta,
            text_fields=list(meta.get("text_fields", [])),
            id_field=id_field,
            normalize_embeddings=normalize_embeddings,
            chunk_size=int(meta.get("chunk_size", 120)),
            chunk_overlap=int(meta.get("chunk_overlap", 30)),
            faiss_metric=faiss_metric,
            faiss_index_type=faiss_index_type,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_ef_search=hnsw_ef_search,
            output_dir=output_dir,
            index=index,
        )
        db._validate_state()
        return db

    def add_documents(
        self,
        documents: Iterable[dict],
        *,
        store_fields: Iterable[str] | None = None,
        min_text_length: int = 1,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        persist: bool = True,
        output_dir: Path | None = None,
    ) -> int:
        _require_faiss()
        store_fields = list(store_fields or [])
        chunk_texts: list[str] = []
        chunk_doc_ids: list[str] = []
        new_doc_meta: dict[str, dict] = {}

        for idx, doc in enumerate(documents):
            doc_id = str(doc.get(self.id_field) or f"new_doc_{idx}")
            parts: list[str] = []
            for field_name in self.text_fields:
                value = doc.get(field_name)
                if value:
                    parts.append(str(value))
            text = " ".join(parts).strip()
            if len(text) < min_text_length:
                continue

            meta = {key: doc.get(key) for key in store_fields}
            meta[self.id_field] = doc_id
            new_doc_meta[doc_id] = meta

            chunks = _chunk_text(text, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            for chunk in chunks:
                if len(chunk) < min_text_length:
                    continue
                chunk_texts.append(chunk)
                chunk_doc_ids.append(doc_id)

        if not chunk_texts:
            return 0

        model = self.get_model()
        new_embeddings = model.encode(
            chunk_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=False,
        )
        new_embeddings = np.ascontiguousarray(np.asarray(new_embeddings, dtype=np.float32))
        vectors_for_index = self._prepare_vectors_for_index(new_embeddings, self.normalize_embeddings)

        if self.embeddings.size == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.ascontiguousarray(
                np.vstack([np.asarray(self.embeddings, dtype=np.float32), new_embeddings]),
                dtype=np.float32,
            )

        self.index_to_doc_id.extend(chunk_doc_ids)
        self.doc_id_to_meta.update(new_doc_meta)

        index = self.ensure_index()
        index.add(vectors_for_index)
        self._validate_state()

        if persist:
            target = output_dir or self.output_dir
            if target is None:
                raise ValueError(
                    "No output_dir configured for incremental persistence. "
                    "Pass output_dir=... or load/build with a persisted location."
                )
            self.save(Path(target))

        return len(chunk_doc_ids)

    def search(
        self,
        query: str,
        top_k: int = 10,
        chunk_pool_factor: int = 8,
    ) -> list[dict]:
        if not query:
            return []
        top_k = max(int(top_k), 0)
        if top_k == 0:
            return []

        if len(self.index_to_doc_id) == 0:
            return []
        index = self.ensure_index()

        model = self.get_model()
        query_embedding = model.encode(
            [query],
            normalize_embeddings=False,
        )
        query_embedding = np.ascontiguousarray(np.asarray(query_embedding, dtype=np.float32))
        query_for_search = self._prepare_vectors_for_index(query_embedding, self.normalize_embeddings)

        chunk_k = min(
            max(top_k * max(int(chunk_pool_factor), 1), top_k),
            len(self.index_to_doc_id),
        )
        distances, labels = index.search(query_for_search, chunk_k)
        raw_scores = distances[0]
        raw_indices = labels[0]

        doc_best_score: dict[str, float] = {}
        doc_chunk_hits: dict[str, int] = {}

        for distance, idx in zip(raw_scores, raw_indices):
            if idx < 0:
                continue
            if idx >= len(self.index_to_doc_id):
                continue
            doc_id = self.index_to_doc_id[idx]
            score = self._distance_to_score(float(distance), self.faiss_metric)
            prev = doc_best_score.get(doc_id)
            if prev is None or score > prev:
                doc_best_score[doc_id] = score
            doc_chunk_hits[doc_id] = doc_chunk_hits.get(doc_id, 0) + 1

        ranked = sorted(doc_best_score.items(), key=lambda item: item[1], reverse=True)[:top_k]

        results: list[dict] = []
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            payload = {
                self.id_field: doc_id,
                "doc_id": doc_id,
                "score": float(score),
                "rank": rank,
                "chunk_hits": int(doc_chunk_hits.get(doc_id, 0)),
            }
            meta = self.doc_id_to_meta.get(doc_id, {})
            payload.update(meta)
            results.append(payload)
        return results

    @staticmethod
    def _distance_to_score(distance: float, metric: str) -> float:
        # IP: mayor es mejor. L2: menor es mejor, se convierte a score creciente.
        if metric.lower() == "l2":
            return -distance
        return distance

    @staticmethod
    def _create_faiss_index(
        dim: int,
        metric: str,
        *,
        faiss_index_type: str,
        hnsw_m: int,
        hnsw_ef_construction: int,
        hnsw_ef_search: int,
    ) -> object:
        _require_faiss()
        index_type = faiss_index_type.lower()
        metric = metric.lower()
        if metric == "ip":
            metric_type = faiss.METRIC_INNER_PRODUCT
        elif metric == "l2":
            metric_type = faiss.METRIC_L2
        else:
            raise ValueError("Unsupported FAISS metric. Use 'ip' or 'l2'.")

        if index_type == "flat":
            if metric == "ip":
                return faiss.IndexFlatIP(dim)
            return faiss.IndexFlatL2(dim)

        if index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, max(int(hnsw_m), 2), metric_type)
            index.hnsw.efConstruction = max(int(hnsw_ef_construction), 8)
            index.hnsw.efSearch = max(int(hnsw_ef_search), 8)
            return index

        raise ValueError("Unsupported FAISS index type. Use 'hnsw' or 'flat'.")

    @staticmethod
    def _prepare_vectors_for_index(vectors: np.ndarray, normalize: bool) -> np.ndarray:
        _require_faiss()
        prepared = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        if prepared.ndim != 2:
            raise ValueError(f"Vectors must be 2D. Got shape={prepared.shape}")
        if prepared.size == 0:
            return prepared
        if normalize:
            faiss.normalize_L2(prepared)
        return prepared

    def _validate_state(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D. Got shape={self.embeddings.shape}")
        if self.embeddings.shape[0] != len(self.index_to_doc_id):
            raise ValueError(
                "Inconsistent state: embeddings rows and index_to_doc_id length differ "
                f"({self.embeddings.shape[0]} != {len(self.index_to_doc_id)})."
            )
        for doc_id in self.index_to_doc_id:
            if doc_id not in self.doc_id_to_meta:
                self.doc_id_to_meta[doc_id] = {self.id_field: doc_id}
