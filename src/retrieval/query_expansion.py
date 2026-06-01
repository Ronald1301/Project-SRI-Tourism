from __future__ import annotations

import json
import hashlib
import logging
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


DEFAULT_FEEDBACK_DB_PATH = Path("data/feedback/query_feedback.db")
DEFAULT_EXPANSION_CONFIG_PATH = Path("data/config/query_expansion.json")
DEFAULT_DOMAIN_SYNONYMS_PATH = Path("data/config/tourism_synonyms.json")
WORD_RE = re.compile(r"[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ]{3,}")
logger = logging.getLogger("src.retrieval.query_expansion")

STOPWORDS = {
    "aqui",
    "al",
    "algo",
    "ante",
    "con",
    "contra",
    "de",
    "del",
    "como",
    "el",
    "cual",
    "en",
    "cuando",
    "entre",
    "donde",
    "desde",
    "e",
    "este",
    "esta",
    "estos",
    "estas",
    "la",
    "las",
    "lo",
    "los",
    "para",
    "pero",
    "por",
    "porque",
    "sin",
    "que",
    "sobre",
    "su",
    "sus",
    "todo",
    "una",
    "uno",
    "unos",
    "unas",
    "y",
    "o",
    "u",
    "ver",
    "visitar",
    "viaje",
    "viajar",
    "turismo",
    "turistico",
    "turistica",
    "cuba",
    "cubano",
    "cubana",
    "lugar",
    "lugares",
    "mejor",
    "mejores",
}

DOMAIN_SYNONYMS = {
    "playa": ["cayo", "balneario", "costa", "mar"],
    "playas": ["cayo", "balneario", "costa", "mar"],
    "hotel": ["alojamiento", "hostal", "reserva"],
    "hoteles": ["alojamiento", "hostal", "reserva"],
    "restaurante": ["gastronomia", "comida", "cocina"],
    "restaurantes": ["gastronomia", "comida", "cocina"],
    "habana": ["malecon", "vedado", "capital"],
    "vieja": ["casco historico", "centro historico"],
    "museo": ["galeria", "patrimonio", "cultura"],
    "museos": ["galeria", "patrimonio", "cultura"],
    "excursion": ["tour", "ruta", "recorrido"],
    "excursiones": ["tour", "ruta", "recorrido"],
    "naturaleza": ["parque", "sendero", "reserva"],
    "familia": ["ninos", "familiar", "actividades"],
}

IMPLICIT_EVENT_GROUPS = {
    "copy_url": "source_interaction",
    "copiar_url": "source_interaction",
    "open_source": "source_interaction",
    "abrir_fuente": "source_interaction",
    "source_interaction": "source_interaction",
}

IMPLICIT_WEIGHTS = {
    "source_interaction": 0.65,
    "view_result": 0.2,
    "dwell": 0.35,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def _normalize_key(text: str) -> str:
    return re.sub(r"\s+", " ", _strip_accents(str(text or "").casefold())).strip()


def _tokenize_raw(text: str) -> list[str]:
    tokens: list[str] = []
    for match in WORD_RE.finditer(str(text or "")):
        token = _normalize_key(match.group(0))
        if token and token not in STOPWORDS:
            tokens.append(token)
    return tokens


def _term_parts(term: str) -> list[str]:
    normalized = _normalize_key(term).replace(" ", "_")
    return [part for part in re.split(r"[_\s]+", normalized) if part]


def _contains_stopword_fragment(term: str) -> bool:
    parts = _term_parts(term)
    if not parts:
        return True
    return any(part in STOPWORDS or len(part) < 3 for part in parts)


def _is_valid_expansion_term(term: str) -> bool:
    normalized = _normalize_key(term)
    return bool(normalized) and not _contains_stopword_fragment(normalized)


@dataclass
class ExpansionResult:
    original_query: str
    expanded_query: str
    terms: list[str]
    term_scores: dict[str, float]
    method: str = "hybrid"
    applied: bool = False
    selected_query: str | None = None
    selected_strategy: str = "raw"
    cached: bool = False
    trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "original_query": self.original_query,
            "expanded_query": self.expanded_query,
            "selected_query": self.selected_query or self.expanded_query,
            "selected_strategy": self.selected_strategy,
            "terms": self.terms,
            "term_scores": self.term_scores,
            "method": self.method,
            "applied": self.applied,
            "cached": self.cached,
            "trace": self.trace,
        }


@dataclass(frozen=True)
class ExpansionSettings:
    max_terms: int = 28
    top_documents_for_context: int = 3
    acceptance_threshold: float = 0.75
    cache_max_size: int = 128
    cache_ttl_seconds: int = 300
    score_floor: float = 0.1
    synonym_limit_per_term: int = 2
    synonym_weight: float = 0.85
    pseudo_relevance_title_weight: float = 0.7
    pseudo_relevance_body_weight: float = 0.35
    ngram_unigram_weight: float = 1.0
    ngram_bigram_weight: float = 1.2
    ngram_trigram_weight: float = 1.3
    enable_ngrams: bool = True
    cooccurrence_weight: float = 0.28
    cooccurrence_window: int = 9
    feedback_positive_weight: float = 0.95
    feedback_negative_weight: float = -0.75
    feedback_implicit_weights: dict[str, float] = field(
        default_factory=lambda: {
            "source_interaction": 0.65,
            "view_result": 0.2,
            "dwell": 0.35,
        }
    )
    rocchio_alpha: float = 1.0
    rocchio_beta_with_feedback: float = 0.75
    rocchio_beta_without_feedback: float = 0.25
    rocchio_gamma: float = 0.15
    rocchio_terms_limit: int = 18
    logging_level: str = "INFO"

    @classmethod
    def from_file(cls, path: Path | str | None) -> "ExpansionSettings":
        defaults = cls()
        payload = _load_json_payload(path)
        weights = dict(payload.get("weights") or {})
        pseudo = dict(weights.get("pseudo_relevance") or {})
        feedback = dict(weights.get("feedback") or {})
        rocchio = dict(weights.get("rocchio") or {})
        implicit_weights = dict(defaults.feedback_implicit_weights)
        for key, value in dict(feedback.get("implicit_event") or {}).items():
            normalized_key = _normalize_key(str(key))
            try:
                implicit_weights[normalized_key] = float(value)
            except (TypeError, ValueError):
                continue

        return cls(
            max_terms=max(_coerce_int(payload.get("max_terms"), defaults.max_terms), 1),
            top_documents_for_context=max(
                _coerce_int(payload.get("top_documents_for_context"), defaults.top_documents_for_context),
                1,
            ),
            acceptance_threshold=_clamp_float(
                _coerce_float(payload.get("acceptance_threshold"), defaults.acceptance_threshold),
                0.0,
                1.0,
            ),
            cache_max_size=max(_coerce_int(payload.get("cache_max_size"), defaults.cache_max_size), 1),
            cache_ttl_seconds=max(_coerce_int(payload.get("cache_ttl_seconds"), defaults.cache_ttl_seconds), 1),
            score_floor=_clamp_float(
                _coerce_float(payload.get("score_floor"), defaults.score_floor),
                0.0,
                1.0,
            ),
            synonym_limit_per_term=max(
                _coerce_int(payload.get("synonym_limit_per_term"), defaults.synonym_limit_per_term),
                1,
            ),
            synonym_weight=_coerce_float(weights.get("synonym"), defaults.synonym_weight),
            pseudo_relevance_title_weight=_coerce_float(
                pseudo.get("title"),
                defaults.pseudo_relevance_title_weight,
            ),
            pseudo_relevance_body_weight=_coerce_float(
                pseudo.get("body"),
                defaults.pseudo_relevance_body_weight,
            ),
            ngram_unigram_weight=_coerce_float(
                payload.get("ngram_unigram_weight"),
                defaults.ngram_unigram_weight,
            ),
            ngram_bigram_weight=_coerce_float(
                payload.get("ngram_bigram_weight"),
                defaults.ngram_bigram_weight,
            ),
            ngram_trigram_weight=_coerce_float(
                payload.get("ngram_trigram_weight"),
                defaults.ngram_trigram_weight,
            ),
            enable_ngrams=bool(payload.get("enable_ngrams", defaults.enable_ngrams)),
            cooccurrence_weight=_coerce_float(weights.get("cooccurrence"), defaults.cooccurrence_weight),
            cooccurrence_window=max(
                _coerce_int(payload.get("cooccurrence_window"), defaults.cooccurrence_window),
                1,
            ),
            feedback_positive_weight=_coerce_float(
                feedback.get("positive_explicit"),
                defaults.feedback_positive_weight,
            ),
            feedback_negative_weight=_coerce_float(
                feedback.get("negative_explicit"),
                defaults.feedback_negative_weight,
            ),
            feedback_implicit_weights=implicit_weights,
            rocchio_alpha=_coerce_float(rocchio.get("alpha"), defaults.rocchio_alpha),
            rocchio_beta_with_feedback=_coerce_float(
                rocchio.get("beta"),
                defaults.rocchio_beta_with_feedback,
            ),
            rocchio_beta_without_feedback=_coerce_float(
                rocchio.get("beta_no_feedback"),
                defaults.rocchio_beta_without_feedback,
            ),
            rocchio_gamma=_coerce_float(rocchio.get("gamma"), defaults.rocchio_gamma),
            rocchio_terms_limit=max(
                _coerce_int(rocchio.get("top_terms_extracted"), defaults.rocchio_terms_limit),
                1,
            ),
            logging_level=str(payload.get("logging_level") or defaults.logging_level).upper(),
        )


def _load_json_payload(path: Path | str | None) -> dict[str, Any]:
    if path is None:
        return {}
    json_path = Path(path)
    if not json_path.exists():
        return {}
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


@dataclass
class CacheEntry:
    value: ExpansionResult
    expires_at: float


class CacheManager:
    def __init__(self, *, max_size: int = 128, ttl_seconds: int = 300) -> None:
        self.max_size = max(int(max_size), 1)
        self.ttl_seconds = max(int(ttl_seconds), 1)
        self._entries: "OrderedDict[str, CacheEntry]" = OrderedDict()

    def get(self, key: str) -> ExpansionResult | None:
        now = datetime.now(timezone.utc).timestamp()
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expires_at <= now:
            self._entries.pop(key, None)
            return None
        self._entries.move_to_end(key)
        return entry.value

    def set(self, key: str, value: ExpansionResult) -> None:
        now = datetime.now(timezone.utc).timestamp()
        if key in self._entries:
            self._entries.move_to_end(key)
        self._entries[key] = CacheEntry(value=value, expires_at=now + self.ttl_seconds)
        while len(self._entries) > self.max_size:
            self._entries.popitem(last=False)


class NGramExtractor:
    def __init__(self, *, max_n: int = 3) -> None:
        self.max_n = max(int(max_n), 1)

    def extract(self, tokens: list[str]) -> list[str]:
        normalized = [token for token in tokens if token and token not in STOPWORDS]
        ngrams: list[str] = []
        seen: set[str] = set()
        for n in range(1, self.max_n + 1):
            if len(normalized) < n:
                break
            for index in range(len(normalized) - n + 1):
                gram = "_".join(normalized[index : index + n])
                if gram and gram not in seen:
                    seen.add(gram)
                    ngrams.append(gram)
        return ngrams


@dataclass
class FeedbackStore:
    db_path: Path = DEFAULT_FEEDBACK_DB_PATH

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        conn = self._connect()
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_explicit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_key TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    relevance INTEGER NOT NULL,
                    expanded_query TEXT,
                    search_mode TEXT,
                    timestamp TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_implicit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_key TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    event TEXT NOT NULL,
                    event_group TEXT NOT NULL,
                    weight REAL NOT NULL,
                    search_mode TEXT,
                    timestamp TEXT NOT NULL,
                    UNIQUE(query_key, doc_id, event_group)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_explicit_query ON feedback_explicit(query_key)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_explicit_doc ON feedback_explicit(doc_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_implicit_query ON feedback_implicit(query_key)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_implicit_doc ON feedback_implicit(doc_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_implicit_query_doc ON feedback_implicit(query_key, doc_id)")
            conn.commit()
        finally:
            conn.close()

    def add_explicit(
        self,
        *,
        query: str,
        doc_id: str,
        relevance: int,
        expanded_query: str | None = None,
        search_mode: str | None = None,
    ) -> dict[str, Any]:
        event = {
            "query": query,
            "query_key": _normalize_key(query),
            "expanded_query": expanded_query,
            "doc_id": str(doc_id or "").strip(),
            "relevance": int(relevance),
            "search_mode": search_mode,
            "timestamp": _now_iso(),
        }
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO feedback_explicit(query_key, doc_id, relevance, expanded_query, search_mode, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event["query_key"],
                    event["doc_id"],
                    event["relevance"],
                    event["expanded_query"],
                    event["search_mode"],
                    event["timestamp"],
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return event

    def add_implicit(
        self,
        *,
        query: str,
        doc_id: str,
        event: str,
        search_mode: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        event_group = IMPLICIT_EVENT_GROUPS.get(_normalize_key(event), _normalize_key(event))
        query_key = _normalize_key(query)
        doc_key = str(doc_id or "").strip()
        payload = {
            "query": query,
            "query_key": query_key,
            "doc_id": doc_key,
            "event": event,
            "event_group": event_group,
            "weight": float(IMPLICIT_WEIGHTS.get(event_group, 0.2)),
            "search_mode": search_mode,
            "timestamp": _now_iso(),
        }
        conn = self._connect()
        try:
            existing = conn.execute(
                """
                SELECT id, timestamp FROM feedback_implicit
                WHERE query_key = ? AND doc_id = ? AND event_group = ?
                """,
                (query_key, doc_key, event_group),
            ).fetchone()
            if existing is not None:
                payload["timestamp"] = str(existing["timestamp"])
                return payload, False

            conn.execute(
                """
                INSERT INTO feedback_implicit(query_key, doc_id, event, event_group, weight, search_mode, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query_key,
                    doc_key,
                    event,
                    event_group,
                    payload["weight"],
                    search_mode,
                    payload["timestamp"],
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return payload, True

    def query_feedback(self, query: str) -> dict[str, list[dict[str, Any]]]:
        query_key = _normalize_key(query)
        conn = self._connect()
        try:
            explicit_rows = conn.execute(
                """
                SELECT query_key, doc_id, relevance, expanded_query, search_mode, timestamp
                FROM feedback_explicit
                WHERE query_key = ?
                ORDER BY timestamp ASC, id ASC
                """,
                (query_key,),
            ).fetchall()
            implicit_rows = conn.execute(
                """
                SELECT query_key, doc_id, event, event_group, weight, search_mode, timestamp
                FROM feedback_implicit
                WHERE query_key = ?
                ORDER BY timestamp ASC, id ASC
                """,
                (query_key,),
            ).fetchall()
        finally:
            conn.close()

        return {
            "explicit": [dict(row) for row in explicit_rows],
            "implicit": [dict(row) for row in implicit_rows],
        }

    def signature(self, query: str) -> str:
        feedback = self.query_feedback(query)
        payload = json.dumps(feedback, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()


class QueryExpander:
    def __init__(
        self,
        searcher: Any,
        *,
        feedback_db_path: Path | str = DEFAULT_FEEDBACK_DB_PATH,
        config_path: Path | str | None = DEFAULT_EXPANSION_CONFIG_PATH,
        synonyms_path: Path | str | None = DEFAULT_DOMAIN_SYNONYMS_PATH,
        max_terms: int | None = None,
    ) -> None:
        self.searcher = searcher
        self.feedback_store = FeedbackStore(Path(feedback_db_path))
        self.settings = ExpansionSettings.from_file(config_path)
        self.domain_terms = self._build_domain_terms()
        self.domain_synonyms = self._load_synonym_map(synonyms_path)
        self.ngram_extractor = NGramExtractor(max_n=3)
        self.cache = CacheManager(
            max_size=self.settings.cache_max_size,
            ttl_seconds=self.settings.cache_ttl_seconds,
        )
        self.logger = logger
        try:
            self.logger.setLevel(getattr(logging, self.settings.logging_level, logging.INFO))
        except Exception:
            self.logger.setLevel(logging.INFO)
        configured_max_terms = self.settings.max_terms if max_terms is None else int(max_terms)
        self.max_terms = max(configured_max_terms, 1)

    def expand_query(
        self,
        query: str,
        *,
        method: str = "hybrid",
        top_documents: list[Any] | None = None,
        max_terms: int | None = None,
    ) -> ExpansionResult:
        query = str(query or "").strip()
        if not query:
            return ExpansionResult(query, query, [], {}, method=method, applied=False)

        top_docs = [
            self._document_payload(doc)
            for doc in (top_documents or [])[: self.settings.top_documents_for_context]
        ]
        cache_key = self._build_cache_key(query, method, max_terms, top_docs)
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.logger.debug("expand_query cache hit | query=%s", query)
            return ExpansionResult(
                original_query=cached.original_query,
                expanded_query=cached.expanded_query,
                terms=list(cached.terms),
                term_scores=dict(cached.term_scores),
                method=cached.method,
                applied=cached.applied,
                selected_query=cached.selected_query,
                selected_strategy=cached.selected_strategy,
                cached=True,
                trace=list(cached.trace),
            )

        self.logger.info("expand_query | query=%s | method=%s", query, method)
        trace: list[dict[str, Any]] = []
        candidates: dict[str, float] = {}
        self._candidate_origins: dict[str, str] = {}
        original_terms = set(_tokenize_raw(query))
        feedback = self.feedback_store.query_feedback(query)

        self._add_pseudo_relevance_terms(top_docs, candidates, trace)
        self._add_rocchio_terms(query, top_docs, feedback, candidates, trace)
        self._add_synonym_terms(query, candidates, trace)
        self._add_cooccurrence_terms(query, top_docs, candidates, trace)
        self._add_feedback_terms(feedback, candidates, trace)
        self._add_ngram_terms(query, top_docs, candidates, trace)

        effective_max_terms = self.max_terms if max_terms is None else max(int(max_terms), 1)
        ranked_terms = self._rank_terms(candidates, original_terms, effective_max_terms, self._candidate_origins)
        expanded_query = " ".join([query] + ranked_terms).strip() if ranked_terms else query
        result = ExpansionResult(
            original_query=query,
            expanded_query=expanded_query,
            terms=ranked_terms,
            term_scores={term: round(float(candidates.get(term, 0.0)), 4) for term in ranked_terms},
            method=method,
            applied=bool(ranked_terms),
            selected_query=expanded_query,
            selected_strategy="expanded" if ranked_terms else "raw",
            cached=False,
            trace=trace,
        )
        self.cache.set(cache_key, result)
        self._candidate_origins = {}
        self.logger.info(
            "expand_query done | query=%s | applied=%s | terms=%d | expanded=%s",
            query,
            result.applied,
            len(result.terms),
            result.expanded_query,
        )
        return result

    def record_explicit_feedback(
        self,
        *,
        query: str,
        doc_id: str,
        relevance: int,
        expanded_query: str | None = None,
        search_mode: str | None = None,
    ) -> dict[str, Any]:
        return self.feedback_store.add_explicit(
            query=query,
            doc_id=doc_id,
            relevance=relevance,
            expanded_query=expanded_query,
            search_mode=search_mode,
        )

    def record_implicit_feedback(
        self,
        *,
        query: str,
        doc_id: str,
        event: str,
        search_mode: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        return self.feedback_store.add_implicit(
            query=query,
            doc_id=doc_id,
            event=event,
            search_mode=search_mode,
        )

    def _add_candidate(
        self,
        candidates: dict[str, float],
        term: str,
        score: float,
        *,
        origin: str | None = None,
    ) -> None:
        term = _normalize_key(term)
        if not _is_valid_expansion_term(term) or float(score) < self.settings.score_floor:
            return
        candidates[term] = candidates.get(term, 0.0) + float(score)
        if origin:
            origins = getattr(self, "_candidate_origins", None)
            if isinstance(origins, dict) and term not in origins:
                origins[term] = origin

    def _add_synonym_terms(self, query: str, candidates: dict[str, float], trace: list[dict[str, Any]]) -> None:
        added = 0
        for token in self._build_query_tokens(query):
            for synonym in self.domain_synonyms.get(token, [])[: self.settings.synonym_limit_per_term]:
                self._add_candidate(candidates, synonym, self.settings.synonym_weight, origin="synonyms")
                added += 1
        trace.append({"technique": "synonyms", "added": added})
        self.logger.debug("synonyms | added=%d", added)

    def _add_pseudo_relevance_terms(
        self,
        docs: list[dict[str, Any]],
        candidates: dict[str, float],
        trace: list[dict[str, Any]],
    ) -> None:
        added = 0
        for rank, doc in enumerate(docs, start=1):
            rank_weight = 1.0 / rank
            title_tokens = _tokenize_raw(str(doc.get("title") or ""))
            body_tokens = _tokenize_raw(
                " ".join(
                    [
                        str(doc.get("summary") or ""),
                        str(doc.get("content_text") or doc.get("content") or ""),
                    ]
                )
            )
            for token in self._top_frequency_terms(title_tokens, limit=24):
                self._add_candidate(
                    candidates,
                    token,
                    self.settings.pseudo_relevance_title_weight * rank_weight,
                    origin="pseudo_relevance",
                )
                added += 1
            for token in self._top_frequency_terms(body_tokens, limit=12):
                self._add_candidate(
                    candidates,
                    token,
                    self.settings.pseudo_relevance_body_weight * rank_weight,
                    origin="pseudo_relevance",
                )
                added += 1
            if self.settings.enable_ngrams:
                self._add_ngram_candidates(
                    title_tokens,
                    candidates,
                    self.settings.ngram_unigram_weight * rank_weight,
                    origin="ngrams",
                )
                self._add_ngram_candidates(
                    body_tokens,
                    candidates,
                    self.settings.ngram_bigram_weight * rank_weight,
                    origin="ngrams",
                )
        trace.append({"technique": "pseudo_relevance", "added": added, "documents": len(docs)})
        self.logger.debug("pseudo_relevance | docs=%d | added=%d", len(docs), added)

    def _add_cooccurrence_terms(
        self,
        query: str,
        docs: list[dict[str, Any]],
        candidates: dict[str, float],
        trace: list[dict[str, Any]],
    ) -> None:
        query_terms = set(self._build_query_tokens(query))
        if not query_terms:
            trace.append({"technique": "cooccurrence", "added": 0})
            return
        added = 0
        for rank, doc in enumerate(docs, start=1):
            tokens = _tokenize_raw(
                " ".join(
                    [
                        str(doc.get("title") or ""),
                        str(doc.get("summary") or ""),
                        str(doc.get("content_text") or doc.get("content") or ""),
                    ]
                )
            )
            rank_weight = 1.0 / rank
            window = max(self.settings.cooccurrence_window, 1)
            half_window = max((window - 1) // 2, 0)
            for idx, token in enumerate(tokens):
                if token not in query_terms:
                    continue
                start = max(idx - half_window, 0)
                end = min(idx + half_window + 1, len(tokens))
                for neighbor in tokens[start:end]:
                    if neighbor != token:
                        self._add_candidate(
                            candidates,
                            neighbor,
                            self.settings.cooccurrence_weight * rank_weight,
                            origin="cooccurrence",
                        )
                        added += 1
        trace.append({"technique": "cooccurrence", "added": added, "window": self.settings.cooccurrence_window})
        self.logger.debug("cooccurrence | added=%d | window=%d", added, self.settings.cooccurrence_window)

    def _add_feedback_terms(
        self,
        feedback: dict[str, list[dict[str, Any]]],
        candidates: dict[str, float],
        trace: list[dict[str, Any]],
    ) -> None:
        added = 0
        for item in feedback.get("explicit", []):
            doc_id = str(item.get("doc_id") or "")
            relevance = int(item.get("relevance") or 0)
            doc = self._document_by_id(doc_id)
            if not doc:
                continue
            score = self.settings.feedback_positive_weight if relevance > 0 else self.settings.feedback_negative_weight
            for token in self._top_terms_for_doc(doc, limit=10):
                self._add_candidate(candidates, token, score, origin="feedback")
                added += 1

        for item in feedback.get("implicit", []):
            doc_id = str(item.get("doc_id") or "")
            doc = self._document_by_id(doc_id)
            if not doc:
                continue
            event_group = _normalize_key(str(item.get("event_group") or item.get("event") or ""))
            score = float(
                self.settings.feedback_implicit_weights.get(
                    event_group,
                    float(item.get("weight") or 0.2),
                )
            )
            for token in self._top_terms_for_doc(doc, limit=6):
                self._add_candidate(candidates, token, score, origin="feedback")
                added += 1
        trace.append(
            {
                "technique": "feedback",
                "added": added,
                "explicit": len(feedback.get("explicit", [])),
                "implicit": len(feedback.get("implicit", [])),
            }
        )
        self.logger.debug(
            "feedback | explicit=%d | implicit=%d | added=%d",
            len(feedback.get("explicit", [])),
            len(feedback.get("implicit", [])),
            added,
        )

    def _add_rocchio_terms(
        self,
        query: str,
        top_docs: list[dict[str, Any]],
        feedback: dict[str, list[dict[str, Any]]],
        candidates: dict[str, float],
        trace: list[dict[str, Any]],
    ) -> None:
        matrix = getattr(getattr(self.searcher, "tfidf_index", None), "matrix", None)
        vocabulary = getattr(getattr(self.searcher, "tfidf_index", None), "vocabulary", None)
        if matrix is None or not vocabulary:
            trace.append({"technique": "rocchio", "status": "skipped", "reason": "missing_tfidf"})
            self.logger.warning("rocchio skipped | missing tfidf artifacts")
            return

        try:
            query_tokens = self.searcher.pipeline.process_text(query)
            raw_query_vector = self.searcher.tfidf_index.vectorize_query(query_tokens)
            query_vector = (
                raw_query_vector.toarray().ravel().astype(np.float32, copy=False)
                if sparse.issparse(raw_query_vector)
                else np.asarray(raw_query_vector, dtype=np.float32).reshape(-1)
            )
        except Exception:
            trace.append({"technique": "rocchio", "status": "failed", "reason": "vectorization_error"})
            self.logger.exception("rocchio failed during vectorization")
            return

        positive_ids = [
            str(item.get("doc_id"))
            for item in feedback.get("explicit", [])
            if int(item.get("relevance") or 0) > 0
        ]
        negative_ids = [
            str(item.get("doc_id"))
            for item in feedback.get("explicit", [])
            if int(item.get("relevance") or 0) <= 0
        ]
        if not positive_ids:
            positive_ids = [str(doc.get("doc_id")) for doc in top_docs if doc.get("doc_id")]

        pos_vectors = self._vectors_for_doc_ids(positive_ids)
        neg_vectors = self._vectors_for_doc_ids(negative_ids)
        if not pos_vectors:
            trace.append({"technique": "rocchio", "status": "skipped", "reason": "no_positive_vectors"})
            self.logger.debug("rocchio skipped | no positive vectors")
            return

        alpha = self.settings.rocchio_alpha
        beta = (
            self.settings.rocchio_beta_with_feedback
            if feedback.get("explicit")
            else self.settings.rocchio_beta_without_feedback
        )
        gamma = self.settings.rocchio_gamma
        rocchio_vector = alpha * query_vector
        rocchio_vector = rocchio_vector + beta * np.mean(pos_vectors, axis=0)
        if neg_vectors:
            rocchio_vector = rocchio_vector - gamma * np.mean(neg_vectors, axis=0)

        inverse_vocab = {idx: term for term, idx in vocabulary.items()}
        stem_to_raw = self._stem_to_raw_map(top_docs)
        added = 0
        for idx in np.argsort(rocchio_vector)[::-1][: self.settings.rocchio_terms_limit]:
            value = float(rocchio_vector[idx])
            if value <= 0:
                continue
            stem = inverse_vocab.get(int(idx))
            if not stem:
                continue
            raw_term = stem_to_raw.get(stem, stem)
            self._add_candidate(candidates, raw_term, min(value, 1.0) * 0.8, origin="rocchio")
            added += 1
        trace.append(
            {
                "technique": "rocchio",
                "status": "ok",
                "positive_docs": len(pos_vectors),
                "negative_docs": len(neg_vectors),
                "added": added,
            }
        )
        self.logger.debug(
            "rocchio | positive_docs=%d | negative_docs=%d | added=%d",
            len(pos_vectors),
            len(neg_vectors),
            added,
        )

    def _rank_terms(
        self,
        candidates: dict[str, float],
        original_terms: set[str],
        limit: int,
        origins: dict[str, str] | None = None,
    ) -> list[str]:
        ranked = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))
        selected: list[str] = []
        seen: set[str] = set()
        stage_caps = {
            "pseudo_relevance": 2,
            "rocchio": 1,
            "synonyms": 1,
            "cooccurrence": 1,
            "feedback": 1,
            "ngrams": 1,
        }
        stage_counts: dict[str, int] = {stage: 0 for stage in stage_caps}
        for term, score in ranked:
            key = _normalize_key(term)
            if score < self.settings.score_floor or key in seen or key in original_terms or not _is_valid_expansion_term(key):
                continue
            if any(key in other or other in key for other in seen):
                continue
            stage = (origins or {}).get(key, "")
            if stage in stage_caps and stage_counts[stage] >= stage_caps[stage]:
                continue
            selected.append(term)
            seen.add(key)
            if stage in stage_counts:
                stage_counts[stage] += 1
            if len(selected) >= limit:
                break
        return selected

    def _top_frequency_terms(self, tokens: list[str], *, limit: int) -> list[str]:
        counts: dict[str, int] = {}
        for token in tokens:
            if token in STOPWORDS:
                continue
            counts[token] = counts.get(token, 0) + 1
        return [
            term
            for term, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
        ]

    def _top_terms_for_doc(self, doc: dict[str, Any], *, limit: int) -> list[str]:
        tokens = _tokenize_raw(
            " ".join(
                [
                    str(doc.get("title") or ""),
                    str(doc.get("summary") or ""),
                    str(doc.get("content_text") or doc.get("content") or ""),
                ]
            )
        )
        return self._top_frequency_terms(tokens, limit=limit)

    def _add_ngram_terms(
        self,
        query: str,
        docs: list[dict[str, Any]],
        candidates: dict[str, float],
        trace: list[dict[str, Any]],
    ) -> None:
        if not self.settings.enable_ngrams:
            trace.append({"technique": "ngrams", "status": "disabled"})
            return

        added = 0
        query_tokens = self._build_query_tokens(query)
        for gram in self.ngram_extractor.extract(query_tokens):
            gram_size = gram.count("_") + 1
            if gram_size == 1:
                weight = self.settings.ngram_unigram_weight
            elif gram_size == 2:
                weight = self.settings.ngram_bigram_weight
            else:
                weight = self.settings.ngram_trigram_weight
            self._add_candidate(candidates, gram, weight, origin="ngrams")
            added += 1

        for rank, doc in enumerate(docs, start=1):
            rank_weight = 1.0 / rank
            doc_tokens = _tokenize_raw(
                " ".join(
                    [
                        str(doc.get("title") or ""),
                        str(doc.get("summary") or ""),
                        str(doc.get("content_text") or doc.get("content") or ""),
                    ]
                )
            )
            for gram in self.ngram_extractor.extract(doc_tokens):
                gram_size = gram.count("_") + 1
                if gram_size == 1:
                    weight = self.settings.ngram_unigram_weight
                elif gram_size == 2:
                    weight = self.settings.ngram_bigram_weight
                else:
                    weight = self.settings.ngram_trigram_weight
                weight *= rank_weight
                self._add_candidate(candidates, gram, weight, origin="ngrams")
                added += 1

        trace.append({"technique": "ngrams", "added": added, "enabled": True})
        self.logger.debug("ngrams | added=%d", added)

    def _vectors_for_doc_ids(self, doc_ids: list[str]) -> list[np.ndarray]:
        index = getattr(self.searcher, "tfidf_index", None)
        matrix = getattr(index, "matrix", None)
        doc_id_to_index = getattr(index, "doc_id_to_index", {}) or {}
        vectors: list[np.ndarray] = []
        if matrix is None:
            return vectors
        for doc_id in doc_ids:
            row_idx = doc_id_to_index.get(doc_id)
            if row_idx is None:
                continue
            row = matrix[row_idx]
            vector = row.toarray().ravel() if sparse.issparse(row) else np.asarray(row, dtype=float).reshape(-1)
            vectors.append(vector.astype(np.float32, copy=False))
        return vectors

    def _stem_to_raw_map(self, docs: list[dict[str, Any]]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for doc in docs:
            raw_text = " ".join(
                [
                    str(doc.get("title") or ""),
                    str(doc.get("summary") or ""),
                    str(doc.get("content_text") or doc.get("content") or ""),
                ]
            )
            for raw_term in _tokenize_raw(raw_text):
                try:
                    stems = self.searcher.pipeline.process_text(raw_term)
                except Exception:
                    stems = []
                for stem in stems:
                    mapping.setdefault(stem, raw_term)
        return mapping

    def _document_by_id(self, doc_id: str) -> dict[str, Any]:
        source = getattr(self.searcher, "documents_by_id", {}) or {}
        return dict(source.get(doc_id) or {})

    def _document_payload(self, doc: Any) -> dict[str, Any]:
        if isinstance(doc, dict):
            payload = dict(doc)
        else:
            payload = {
                "doc_id": getattr(doc, "doc_id", ""),
                "title": getattr(doc, "title", ""),
                "summary": getattr(doc, "summary", ""),
                "content_text": getattr(doc, "content_text", ""),
                "url": getattr(doc, "url", ""),
                "score": getattr(doc, "score", 0.0),
            }
            metadata = getattr(doc, "metadata", None)
            if isinstance(metadata, dict):
                payload.update({key: value for key, value in metadata.items() if key not in payload})

        doc_id = str(payload.get("doc_id") or "").strip()
        if doc_id:
            stored = self._document_by_id(doc_id)
            for key, value in stored.items():
                payload.setdefault(key, value)
        return payload

    def _load_synonym_map(self, path: Path | str | None) -> dict[str, list[str]]:
        payload = _load_json_payload(path)
        if not payload:
            payload = dict(DOMAIN_SYNONYMS)

        synonyms: dict[str, list[str]] = {}
        groups: dict[str, set[str]] = {}
        for raw_term, raw_synonyms in payload.items():
            term = _normalize_key(str(raw_term)).replace(" ", "_")
            if not term or not isinstance(raw_synonyms, list) or not _is_valid_expansion_term(term):
                continue
            group = groups.setdefault(term, set())
            group.add(term)
            for synonym in raw_synonyms:
                normalized = _normalize_key(str(synonym)).replace(" ", "_")
                if normalized and _is_valid_expansion_term(normalized):
                    group.add(normalized)

        if not groups:
            return dict(DOMAIN_SYNONYMS)

        for group in groups.values():
            for term in group:
                synonyms.setdefault(term, [])
                for synonym in sorted(
                    synonym
                    for synonym in (group - {term})
                    if self._is_domain_compatible(synonym) and _is_valid_expansion_term(synonym)
                )[: self.settings.synonym_limit_per_term]:
                    if synonym not in synonyms[term]:
                        synonyms[term].append(synonym)

        return {
            term: list(dict.fromkeys(values))[: self.settings.synonym_limit_per_term]
            for term, values in synonyms.items()
        }

    def _build_domain_terms(self) -> set[str]:
        terms: set[str] = set()
        documents = getattr(self.searcher, "documents_by_id", {}) or {}
        for document in documents.values():
            for field in ("title", "summary", "content_text", "entity_name", "location"):
                value = str(document.get(field) or "").strip()
                if not value:
                    continue
                terms.update(_tokenize_raw(value))
        return terms

    def _is_domain_compatible(self, term: str) -> bool:
        if not self.domain_terms:
            return True
        normalized = _normalize_key(term).replace(" ", "_")
        if normalized in self.domain_terms:
            return True
        parts = [part for part in normalized.split("_") if part]
        return any(part in self.domain_terms for part in parts)

    def _build_query_tokens(self, query: str) -> list[str]:
        raw_tokens = _tokenize_raw(query)
        if not self.settings.enable_ngrams:
            return raw_tokens
        ngrams = self.ngram_extractor.extract(raw_tokens)
        return list(dict.fromkeys(raw_tokens + ngrams))

    def _add_ngram_candidates(
        self,
        tokens: list[str],
        candidates: dict[str, float],
        weight: float,
        *,
        origin: str | None = None,
    ) -> None:
        for gram in self.ngram_extractor.extract(tokens):
            self._add_candidate(candidates, gram, weight, origin=origin)

    def _build_cache_key(
        self,
        query: str,
        method: str,
        max_terms: int | None,
        top_docs: list[dict[str, Any]],
    ) -> str:
        feedback_signature = self.feedback_store.signature(query)
        doc_signature = [
            {
                "doc_id": str(doc.get("doc_id") or ""),
                "score": round(float(doc.get("score") or 0.0), 4),
            }
            for doc in top_docs
        ]
        payload = {
            "query": _normalize_key(query),
            "method": method,
            "max_terms": int(max_terms) if max_terms is not None else self.max_terms,
            "docs": doc_signature,
            "feedback": feedback_signature,
            "settings": {
                "enable_ngrams": self.settings.enable_ngrams,
                "cache_ttl_seconds": self.settings.cache_ttl_seconds,
            },
        }
        return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
