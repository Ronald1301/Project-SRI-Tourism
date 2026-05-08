from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np # type: ignore

from src.indexing.tfidf_index import TFIDFIndex
from src.preprocessing.pipeline import PreprocessingPipeline
from src.vector_db.preset import OUTPUT_DIR, resolve_documents_path
from src.vector_db.vector_store import VectorDatabase


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


@dataclass(frozen=True)
class RetrievedDocument:
    citation_id: int
    doc_id: str
    title: str
    url: str
    score: float
    summary: str
    content_text: str
    metadata: dict


@dataclass(frozen=True)
class CandidatePassage:
    citation_id: int
    title: str
    text: str
    score: float


@dataclass(frozen=True)
class RAGResult:
    query: str
    prompt: str
    answer: str
    documents: list[RetrievedDocument]


class DocumentRepository:
    def __init__(self, documents: dict[str, dict]) -> None:
        self.documents = documents

    @classmethod
    def from_jsonl(cls, path: Path) -> "DocumentRepository":
        documents: dict[str, dict] = {}
        for doc in iter_jsonl(path):
            doc_id = str(doc.get("doc_id") or "").strip()
            if not doc_id:
                continue
            documents[doc_id] = doc
        return cls(documents)

    def get(self, doc_id: str) -> dict | None:
        return self.documents.get(doc_id)


class RAGPipeline:
    def __init__(
        self,
        vector_db: VectorDatabase,
        repository: DocumentRepository,
        *,
        language: str = "spanish",
    ) -> None:
        self.vector_db = vector_db
        self.repository = repository
        self.preprocessing = PreprocessingPipeline(language=language)
        self._document_token_cache: dict[str, set[str]] = {}
        self._title_token_cache: dict[str, set[str]] = {}
        self._fallback_index: TFIDFIndex | None = None
        self._vector_search_available = self._has_local_vector_model()

    @classmethod
    def from_preset(
        cls,
        *,
        output_dir: Path = OUTPUT_DIR,
        language: str = "spanish",
    ) -> "RAGPipeline":
        documents_path = resolve_documents_path()
        vector_db = VectorDatabase.load(Path(output_dir))
        repository = DocumentRepository.from_jsonl(documents_path)
        return cls(vector_db, repository, language=language)

    def answer_query(self, query: str, top_k: int = 4) -> RAGResult:
        documents = self.retrieve(query, top_k=top_k)
        prompt = self.build_prompt(query, documents)
        answer = self.generate_answer(query, documents)
        return RAGResult(
            query=query,
            prompt=prompt,
            answer=answer,
            documents=documents,
        )

    def answer_with_lsi(
        self,
        query: str,
        lsi_results: list[dict],
        top_k: int = 4,
    ) -> RAGResult:
        documents = self._convert_lsi_results(lsi_results[:top_k])
        prompt = self.build_prompt(query, documents)
        answer = self.generate_answer(query, documents)
        return RAGResult(
            query=query,
            prompt=prompt,
            answer=answer,
            documents=documents,
        )

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedDocument]:
        if self._vector_search_available:
            try:
                raw_results = self.vector_db.search(query, top_k=top_k)
            except Exception:
                self._vector_search_available = False
                raw_results = self._tfidf_search(query, top_k=top_k)
        else:
            raw_results = self._tfidf_search(query, top_k=top_k)
        documents: list[RetrievedDocument] = []

        for citation_id, item in enumerate(raw_results, start=1):
            doc_id = str(item.get("doc_id") or "").strip()
            source_doc = self.repository.get(doc_id) or {}

            merged = dict(source_doc)
            for key, value in item.items():
                merged.setdefault(key, value)

            title = str(
                merged.get("title")
                or merged.get("entity_name")
                or merged.get("doc_id")
                or f"Documento {citation_id}"
            ).strip()
            summary = str(merged.get("summary") or "").strip()
            content_text = str(merged.get("content_text") or "").strip()
            url = str(merged.get("url") or "").strip()
            score = float(item.get("score", 0.0))

            documents.append(
                RetrievedDocument(
                    citation_id=citation_id,
                    doc_id=doc_id,
                    title=title,
                    url=url,
                    score=score,
                    summary=summary,
                    content_text=content_text,
                    metadata=merged,
                )
            )

        return documents

    def _convert_lsi_results(self, lsi_results: list[dict]) -> list[RetrievedDocument]:
        documents: list[RetrievedDocument] = []
        for citation_id, item in enumerate(lsi_results, start=1):
            doc_id = str(item.get("doc_id") or "").strip()
            source_doc = self.repository.get(doc_id) or {}

            merged = dict(source_doc)
            for key, value in item.items():
                merged.setdefault(key, value)

            title = str(
                merged.get("title")
                or merged.get("entity_name")
                or merged.get("doc_id")
                or f"Documento {citation_id}"
            ).strip()
            summary = str(merged.get("summary") or "").strip()
            content_text = str(merged.get("content") or merged.get("content_text") or "").strip()
            url = str(merged.get("url") or "").strip()
            score = float(item.get("score", 0.0))

            documents.append(
                RetrievedDocument(
                    citation_id=citation_id,
                    doc_id=doc_id,
                    title=title,
                    url=url,
                    score=score,
                    summary=summary,
                    content_text=content_text,
                    metadata=merged,
                )
            )
        return documents

    def _tfidf_search(self, query: str, top_k: int = 4) -> list[dict]:
        index = self._get_fallback_index()
        query_tokens = self.preprocessing.process_text(query)
        if not query_tokens or index.matrix is None or index.matrix.size == 0:
            return []
        focus_tokens = self._focus_query_tokens(set(query_tokens))

        query_vector = index.vectorize_query(query_tokens)
        doc_matrix = np.asarray(index.matrix, dtype=float)
        doc_norms = np.linalg.norm(doc_matrix, axis=1)
        query_norm = np.linalg.norm(query_vector)
        denom = doc_norms * query_norm

        similarities = np.zeros(doc_matrix.shape[0], dtype=float)
        valid = denom > 0
        if np.any(valid):
            similarities[valid] = (doc_matrix[valid] @ query_vector) / denom[valid]

        candidate_count = max(int(top_k) * 6, 12)
        top_indices = np.argsort(similarities)[::-1][:candidate_count]
        reranked_results: list[dict] = []
        for index_pos in top_indices:
            base_score = float(similarities[index_pos])
            if base_score <= 0:
                continue

            doc_id = index.doc_ids[index_pos]
            source_doc = self.repository.get(doc_id) or {}
            title = str(source_doc.get("title") or source_doc.get("entity_name") or "").strip()
            url = str(source_doc.get("url") or "").strip()
            summary = str(source_doc.get("summary") or "").strip()
            content_text = str(source_doc.get("content_text") or "").strip()
            title_tokens = self._get_title_tokens(doc_id, title)
            doc_tokens = self._get_document_tokens(doc_id, title, summary, content_text)
            focus_overlap = len(focus_tokens & doc_tokens)
            title_focus_overlap = len(focus_tokens & title_tokens)
            score = base_score + (title_focus_overlap * 1.1) + (focus_overlap * 0.18)

            reranked_results.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "url": url,
                    "score": score,
                }
            )

        reranked_results.sort(key=lambda item: item["score"], reverse=True)
        return reranked_results[: max(int(top_k), 0)]

    def build_prompt(self, query: str, documents: list[RetrievedDocument]) -> str:
        if not documents:
            context_block = "No se recuperaron documentos relevantes."
        else:
            context_parts: list[str] = []
            for doc in documents:
                snippet = self._best_passages_for_document(doc, query, limit=1)
                excerpt = snippet[0].text if snippet else self._fallback_excerpt(doc)
                context_parts.append(
                    "\n".join(
                        [
                            f"[{doc.citation_id}] Titulo: {doc.title}",
                            f"URL: {doc.url or 'N/D'}",
                            f"Score: {doc.score:.4f}",
                            f"Contexto: {excerpt}",
                        ]
                    )
                )
            context_block = "\n\n".join(context_parts)

        return "\n".join(
            [
                "Eres un asistente RAG especializado en turismo.",
                "Responde unicamente con la evidencia del contexto recuperado.",
                "Reglas:",
                "1. No inventes hechos, precios, fechas o ubicaciones que no aparezcan en el contexto.",
                "2. Si la evidencia es insuficiente o parcial, dilo explicitamente.",
                "3. Integra detalles concretos del contexto y ancla cada idea con citas [1], [2], etc.",
                "4. Prioriza claridad, sintesis y fidelidad al contexto recuperado.",
                "5. Responde en espanol.",
                "",
                "Consulta del usuario:",
                query.strip(),
                "",
                "Contexto recuperado:",
                context_block,
                "",
                "Formato esperado:",
                "- Un parrafo breve que responda la consulta.",
                "- Uno o dos detalles complementarios si aportan valor.",
                "- No extrapoles mas alla de la evidencia.",
            ]
        )

    def generate_answer(self, query: str, documents: list[RetrievedDocument]) -> str:
        if not documents:
            return (
                "No encontre documentos suficientemente relevantes para responder con evidencia "
                f"a la consulta: {query}."
            )

        passages = self._select_passages(query, documents, max_passages=3)
        if not passages:
            titles = ", ".join(f"[{doc.citation_id}] {doc.title}" for doc in documents[:3])
            return (
                f"Recupere documentos relacionados con la consulta, entre ellos {titles}, "
                "pero el contenido disponible no alcanza para construir una respuesta mas detallada."
            )

        answer_parts: list[str] = []
        prefixes = [
            "Segun la informacion recuperada, ",
            "Ademas, ",
            "Tambien, ",
        ]

        for index, passage in enumerate(passages):
            prefix = prefixes[index] if index < len(prefixes) else ""
            answer_parts.append(
                self._with_citation(prefix, passage.text, passage.citation_id)
            )

        if len(documents) > 1:
            sources = ", ".join(
                f"[{doc.citation_id}] {doc.title}" for doc in documents[: min(3, len(documents))]
            )
            answer_parts.append(
                f"Las fuentes principales consultadas fueron {sources}."
            )

        return " ".join(answer_parts)

    def _select_passages(
        self,
        query: str,
        documents: list[RetrievedDocument],
        *,
        max_passages: int,
    ) -> list[CandidatePassage]:
        ranked: list[CandidatePassage] = []
        for doc in documents:
            ranked.extend(self._best_passages_for_document(doc, query, limit=1))

        ranked.sort(key=lambda item: item.score, reverse=True)
        if ranked:
            return ranked[:max_passages]

        fallbacks: list[CandidatePassage] = []
        for doc in documents[:max_passages]:
            excerpt = self._fallback_excerpt(doc)
            if not excerpt:
                continue
            fallbacks.append(
                CandidatePassage(
                    citation_id=doc.citation_id,
                    title=doc.title,
                    text=excerpt,
                    score=doc.score,
                )
            )
        return fallbacks

    def _best_passages_for_document(
        self,
        doc: RetrievedDocument,
        query: str,
        *,
        limit: int,
    ) -> list[CandidatePassage]:
        query_tokens = set(self.preprocessing.process_text(query))
        focus_tokens = self._focus_query_tokens(query_tokens)
        active_query_tokens = focus_tokens or query_tokens
        title_tokens = self._get_title_tokens(doc.doc_id, doc.title)
        title_focus_overlap = len(active_query_tokens & title_tokens)
        candidates: list[CandidatePassage] = []
        seen: set[str] = set()

        for raw_text in self._candidate_segments(doc):
            normalized_text = self._normalize_whitespace(raw_text)
            if not self._is_useful_segment(normalized_text, doc.title):
                continue
            normalized_key = normalized_text.casefold()
            if normalized_key in seen:
                continue
            seen.add(normalized_key)

            segment_tokens = set(self.preprocessing.process_text(normalized_text))
            if (
                active_query_tokens
                and title_focus_overlap == 0
                and not (active_query_tokens & segment_tokens)
            ):
                continue

            score = self._score_segment(
                active_query_tokens,
                segment_tokens,
                doc.score,
                len(normalized_text),
                title_focus_overlap,
            )
            candidates.append(
                CandidatePassage(
                    citation_id=doc.citation_id,
                    title=doc.title,
                    text=normalized_text,
                    score=score,
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[:limit]

    def _get_fallback_index(self) -> TFIDFIndex:
        if self._fallback_index is None:
            documents: dict[str, list[str]] = {}
            for doc_id, source_doc in self.repository.documents.items():
                title = str(source_doc.get("title") or source_doc.get("entity_name") or "").strip()
                summary = str(source_doc.get("summary") or "").strip()
                content_text = str(source_doc.get("content_text") or "").strip()

                title_tokens = self.preprocessing.process_text(title)
                summary_tokens = self.preprocessing.process_text(summary)
                content_tokens = self.preprocessing.process_text(content_text)

                weighted_tokens = (title_tokens * 3) + (summary_tokens * 2) + content_tokens
                if weighted_tokens:
                    documents[doc_id] = weighted_tokens

            index = TFIDFIndex()
            index.build(documents)
            self._fallback_index = index
        return self._fallback_index

    def _candidate_segments(self, doc: RetrievedDocument) -> Iterable[str]:
        if doc.summary:
            yield doc.summary

        text = doc.content_text
        if not text:
            return

        paragraphs = [part.strip() for part in re.split(r"\n+", text) if part.strip()]
        for paragraph in paragraphs:
            if len(paragraph) <= 360:
                yield paragraph
                continue

            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) >= 50:
                    yield sentence

    def _get_document_tokens(
        self,
        doc_id: str,
        title: str,
        summary: str,
        content_text: str,
    ) -> set[str]:
        if doc_id not in self._document_token_cache:
            combined = "\n".join(part for part in [title, summary, content_text] if part)
            self._document_token_cache[doc_id] = set(self.preprocessing.process_text(combined))
        return self._document_token_cache[doc_id]

    def _get_title_tokens(self, doc_id: str, title: str) -> set[str]:
        if doc_id not in self._title_token_cache:
            self._title_token_cache[doc_id] = set(self.preprocessing.process_text(title))
        return self._title_token_cache[doc_id]

    def _score_segment(
        self,
        query_tokens: set[str],
        segment_tokens: set[str],
        doc_score: float,
        segment_length: int,
        title_focus_overlap: int,
    ) -> float:
        overlap = len(query_tokens & segment_tokens)
        coverage = overlap / max(len(query_tokens), 1)
        density = overlap / max(len(segment_tokens), 1)
        length_bonus = min(segment_length, 220) / 220.0 * 0.2
        short_penalty = 0.2 if segment_length < 70 else 0.0
        return (
            float(doc_score)
            + (coverage * 1.2)
            + (density * 0.35)
            + min(overlap, 4) * 0.12
            + (title_focus_overlap * 0.15)
            + length_bonus
            - short_penalty
        )

    def _fallback_excerpt(self, doc: RetrievedDocument) -> str:
        for raw_text in self._candidate_segments(doc):
            normalized = self._normalize_whitespace(raw_text)
            if self._is_useful_segment(normalized, doc.title):
                text = normalized
                break
        else:
            text = doc.summary or doc.content_text or doc.title

        cleaned = self._normalize_whitespace(text)
        if len(cleaned) <= 280:
            return cleaned
        return cleaned[:277].rstrip() + "..."

    def _with_citation(self, prefix: str, text: str, citation_id: int) -> str:
        cleaned = self._normalize_whitespace(text).rstrip(".!?")
        return f"{prefix}{cleaned} [{citation_id}]."

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        return text.strip("-• ")

    def _is_useful_segment(self, text: str, title: str) -> bool:
        normalized = self._normalize_whitespace(text)
        lowered = normalized.casefold()
        title_lower = self._normalize_whitespace(title).casefold()

        if len(normalized) < 40:
            return False
        if lowered in {"visitar cuba", "organizacion de agencias cubanas", title_lower}:
            return False
        if any(
            phrase in lowered
            for phrase in (
                "visitar cuba es una organizacion de agencias cubanas",
                "si eres una agencia o tour operador",
                "datos personales y de contacto",
                "estoy de acuerdo con la politica de privacidad",
                "mapa interactivo",
            )
        ):
            return False
        if lowered.startswith("ubicacion"):
            return False
        if lowered.startswith("calle ") and "cuba" in lowered and normalized.count(",") >= 2:
            return False
        if normalized.count(":") >= 4:
            return False
        return True

    def _focus_query_tokens(self, query_tokens: set[str]) -> set[str]:
        generic_tokens = {
            "cub",
            "turism",
            "viaj",
            "destin",
            "vacacion",
            "hotel",
            "ciud",
            "lug",
        }
        focused = {token for token in query_tokens if token not in generic_tokens}
        return focused or query_tokens

    def _has_local_vector_model(self) -> bool:
        model_name = str(self.vector_db.model_name or "").strip()
        if not model_name:
            return False

        model_path = Path(model_name)
        if model_path.exists():
            return True

        cache_candidates = [
            Path.home() / ".cache" / "torch" / "sentence_transformers" / model_name,
            Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_name.replace('/', '--')}",
        ]
        return any(candidate.exists() for candidate in cache_candidates)
