from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.file_manager import load_json, save_json

def iter_jsonl(path : Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

@dataclass
class VectorDatabase:
    model_name : str
    embeddings : np.ndarray
    doc_ids : list[str]
    metadata : list[dict]
    text_fields : list[str]
    id_field : str
    normalize_embeddings : bool
    model : SentenceTransformer | None = field(default=None, repr=False)

    @classmethod
    def build_from_jsonl(
        cls,
        json_path : Path,
        text_fields : Iterable[str],
        id_field : str = "doc_id",
        store_fields : Iterable[str] | None = None,
        min_text_length : int = 1,
        model_name : str = "all-MiniLM-L6-v2",
        batch_size : int = 32,
        normalize_embeddings : bool = True,
        show_progress_bar : bool = True,
    ) -> "VectorDatabase":
        if not json_path.exists():
            raise FileNotFoundError(f"JSONL not found: {json_path}")
        
        text_fields = [field for field in text_fields if field]
        if not text_fields:
            raise ValueError("text_fields must include at least one field")
        
        doc_ids : list[str] = []
        texts : list[str] = []
        metadata : list[dict] = []
        store_fields = list(store_fields or [])

        for idx, doc in enumerate(iter_jsonl(json_path)):
            doc_id = doc.get(id_field) or f"doc_{idx}"
            parts = []
            for field_name in text_fields:
                value = doc.get(field_name)
                if value:
                    parts.append(str(value))
            text = " ".join(parts).strip()
            if len(text) < min_text_length:
                continue

            doc_ids.append(str(doc_id))
            texts.append(text)

            meta = {key: doc.get(key) for key in store_fields}
            meta[id_field] = doc_id
            metadata.append(meta)

        if not texts:
            raise ValueError("No valid documents found to vectorize")

        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            texts,
            batch_size = batch_size,
            show_progress_bar = show_progress_bar,
            normalize_embeddings = normalize_embeddings
        )        
        embeddings = np.asarray(embeddings, dtype = np.float32)

        return cls(
            model_name = model_name,
            embeddings = embeddings,
            doc_ids = doc_ids,
            metadata = metadata,
            text_fields = list(text_fields),
            id_field = id_field,
            normalize_embeddings = normalize_embeddings,
        )
    
    def get_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self.model_name
    
    def save(self, output_dir : Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        embeddings_path = output_dir / "embeddings.npy"
        meta_path = output_dir / "meta.json"

        np.save(embeddings_path, self.embeddings)
        meta = {
            "model_name" : self.model_name,
            "doc_ids" : self.doc_ids,
            "metadata" : self.metadata,
            "text_fields" : self.text_fields,
            "id_field" : self.id_field,
            "normalize_embeddings" : self.normalize_embeddings,
            "created_at" : datetime.now(UTC).isoformat(),
        }
        save_json(meta, str(meta_path))

    @classmethod
    def load(cls, output_dir : Path) -> "VectorDatabase":
        embeddings_path = output_dir / "embeddings.npy"
        meta_path = output_dir / "meta.json"

        embeddings = np.load(embeddings_path)
        meta = load_json(str(meta_path))
        return cls(
            model_name = meta.get("model_name", "all-MiniLM-L6_v2"),
            embeddings = embeddings,
            doc_ids = meta.get("doc_ids", []),
            metadata = meta.get("metadata", []),
            text_fields = meta.get("text_fields", []),
            id_field = meta.get("id_field", []),
            normalize_embeddings = bool(meta.get("normalize_embeddings", True)),
        )
    
    def search(self, query : str, top_k : int = 10) -> list[dict]:
        if not query:
            return []
        model = self.get_model()
        query_embedding = model.encode(
            [query],
            normalize_embeddings = self.normalize_embeddings,
        )[0]
        query_embedding = np.asarray(query_embedding, dtype=np.float32)

        if self.embeddings.size == 0:
            return []
        
        if self.normalize_embeddings:
            scores = self.embeddings @ query_embedding
        else:
            doc_norms = np.linalg.norm(self.embeddings, axis=1)
            query_norm = np.linalg.norm(query_embedding)
            denom = (doc_norms * query_norm) + 1e-12
            scores = (self.embeddings @ query_embedding) / denom
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        results : list[dict] = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            payload = {
                "doc_id" : doc_id,
                "score" : float(scores[idx]),
            }
            if idx < len(self.metadata):
                payload.update(self.metadata[idx])
            results.append(payload)
        return results
