from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

DEFAULT_RECENCY_HALF_LIFE_DAYS = 3650.0
DEFAULT_SPECIFICITY_PIVOT_WORDS = 220.0
DEFAULT_CONTENT_TYPE_VALUES = {
    "official": 1.0,
    "blog": 0.55,
    "forum": 0.25,
    "unknown": 0.0,
}

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
_ISO_Z_SUFFIX_RE = re.compile(r"Z$", flags=re.IGNORECASE)


@dataclass(frozen=True)
class QuerySignalContext:
    query: str
    normalized_query: str
    query_tokens: tuple[str, ...]
    query_token_set: frozenset[str]
    surface_tokens: tuple[str, ...]
    surface_ngrams: tuple[str, ...]


@dataclass(frozen=True)
class SignalValues:
    phrase_match: float
    recency: float
    content_type_boost: float
    authority: float
    specificity: float
    lexical_overlap: float
    title_match: float
    length_signal: float


def clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def build_query_signal_context(query: str, query_tokens: list[str]) -> QuerySignalContext:
    normalized_query = normalize_text(query)
    surface_tokens = tuple(_TOKEN_RE.findall(normalized_query))
    surface_ngrams = _build_ngrams(surface_tokens, min_n=2, max_n=3)
    clean_tokens = tuple(token for token in query_tokens if token)
    return QuerySignalContext(
        query=query,
        normalized_query=normalized_query,
        query_tokens=clean_tokens,
        query_token_set=frozenset(clean_tokens),
        surface_tokens=surface_tokens,
        surface_ngrams=surface_ngrams,
    )


def phrase_match_signal(query_context: QuerySignalContext, document: Mapping[str, Any]) -> float:
    text = str(document.get("normalized_full_text") or "")
    if not text:
        text = normalize_text(
            " ".join(
                part
                for part in [
                    str(document.get("title") or ""),
                    str(document.get("content") or ""),
                    str(document.get("summary") or ""),
                ]
                if part
            )
        )

    if not text:
        return 0.0

    if query_context.normalized_query and len(query_context.surface_tokens) >= 2:
        if query_context.normalized_query in text:
            return 1.0

    if query_context.surface_ngrams:
        matches = sum(1 for ngram in query_context.surface_ngrams if ngram in text)
        if matches > 0:
            return clamp01(matches / float(len(query_context.surface_ngrams)))

    if query_context.surface_tokens:
        padded = f" {text} "
        token_hits = sum(1 for token in query_context.surface_tokens if f" {token} " in padded)
        return clamp01(token_hits / float(len(query_context.surface_tokens)))
    return 0.0


def recency_signal(
    document: Mapping[str, Any],
    *,
    now: datetime | None = None,
    half_life_days: float = DEFAULT_RECENCY_HALF_LIFE_DAYS,
) -> float:
    dt = _extract_document_datetime(document)
    if dt is None:
        return 0.0

    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    delta_seconds = max((now - dt).total_seconds(), 0.0)
    age_days = delta_seconds / 86400.0
    half_life = max(float(half_life_days), 1.0)
    decay = math.exp(-math.log(2.0) * (age_days / half_life))
    return clamp01(decay)


def content_type_label(document: Mapping[str, Any]) -> str:
    raw_type = normalize_text(
        str(document.get("content_type") or _metadata_value(document, "content_type") or "")
    )
    url = normalize_text(str(document.get("url") or _metadata_value(document, "url") or ""))

    if raw_type:
        if _contains_any(
            raw_type,
            (
                "organization",
                "official",
                "faqpage",
                "government",
                "localbusiness",
                "web_page",
                "webpage",
            ),
        ):
            return "official"
        if _contains_any(raw_type, ("blog", "news", "article", "post")):
            return "blog"
        if _contains_any(raw_type, ("forum", "qapage", "discussion", "community", "question")):
            return "forum"

    if ".gov" in url or ".gob." in url:
        return "official"
    if "blog." in url or "/blog/" in url or "/news/" in url:
        return "blog"
    if "forum" in url or "/foro/" in url:
        return "forum"
    return "unknown"


def content_type_boost_signal(
    document: Mapping[str, Any],
    *,
    values: Mapping[str, float] | None = None,
) -> float:
    table = values or DEFAULT_CONTENT_TYPE_VALUES
    label = content_type_label(document)
    return clamp01(float(table.get(label, table.get("unknown", 0.0))))


def authority_signal(document: Mapping[str, Any]) -> float:
    candidates: list[float] = []
    fields: list[Mapping[str, Any]] = [document]
    metadata = document.get("metadata")
    if isinstance(metadata, Mapping):
        fields.append(metadata)

    for data in fields:
        rating = _to_float(data.get("rating"))
        if rating is not None and rating >= 0:
            candidates.append(clamp01(rating / 5.0 if rating > 1.0 else rating))

        for key in ("authority", "popularity", "authority_score", "popularity_score", "pagerank"):
            value = _to_float(data.get(key))
            if value is None or value < 0:
                continue
            candidates.append(_normalize_generic_metric(value))

        for key in ("reviews_count", "review_count", "helpful", "likes", "views"):
            value = _to_float(data.get(key))
            if value is None or value < 0:
                continue
            candidates.append(_normalize_count_metric(value))

    if not candidates:
        return 0.0
    return clamp01(max(candidates))


def lexical_overlap_signal(query_context: QuerySignalContext, document: Mapping[str, Any]) -> float:
    query_token_set = query_context.query_token_set
    if not query_token_set:
        return 0.0
    content_tokens = _to_token_set(document.get("content_tokens"))
    return clamp01(len(query_token_set & content_tokens) / float(len(query_token_set)))


def title_match_signal(query_context: QuerySignalContext, document: Mapping[str, Any]) -> float:
    query_token_set = query_context.query_token_set
    if not query_token_set:
        return 0.0
    title_tokens = _to_token_set(document.get("title_tokens"))
    return clamp01(len(query_token_set & title_tokens) / float(len(query_token_set)))


def length_signal(document: Mapping[str, Any]) -> float:
    word_count = int(document.get("word_count") or 0)
    if word_count <= 0:
        return 0.0
    return clamp01(word_count / 120.0)


def specificity_signal(
    query_context: QuerySignalContext,
    document: Mapping[str, Any],
    *,
    pivot_words: float = DEFAULT_SPECIFICITY_PIVOT_WORDS,
) -> float:
    query_token_set = query_context.query_token_set
    if not query_token_set:
        return 0.0

    content_tokens = _to_token_set(document.get("content_tokens"))
    title_tokens = _to_token_set(document.get("title_tokens"))
    overlap = len(query_token_set & content_tokens)
    coverage = overlap / float(len(query_token_set))
    title_coverage = len(query_token_set & title_tokens) / float(len(query_token_set))

    word_count = max(int(document.get("word_count") or 0), len(content_tokens), 1)
    density = overlap / float(word_count)
    density_signal = min(density * 120.0, 1.0)

    safe_pivot = max(float(pivot_words), 1.0)
    length_factor = 1.0 / (1.0 + (word_count / safe_pivot))
    score = (0.62 * coverage) + (0.20 * density_signal) + (0.18 * title_coverage)
    score *= 0.70 + (0.30 * length_factor)
    return clamp01(score)


def compute_signal_values(
    query_context: QuerySignalContext,
    document: Mapping[str, Any],
    *,
    recency_half_life_days: float = DEFAULT_RECENCY_HALF_LIFE_DAYS,
    content_type_values: Mapping[str, float] | None = None,
) -> SignalValues:
    return SignalValues(
        phrase_match=phrase_match_signal(query_context, document),
        recency=recency_signal(document, half_life_days=recency_half_life_days),
        content_type_boost=content_type_boost_signal(document, values=content_type_values),
        authority=authority_signal(document),
        specificity=specificity_signal(query_context, document),
        lexical_overlap=lexical_overlap_signal(query_context, document),
        title_match=title_match_signal(query_context, document),
        length_signal=length_signal(document),
    )


def _build_ngrams(tokens: tuple[str, ...], *, min_n: int, max_n: int) -> tuple[str, ...]:
    ngrams: list[str] = []
    token_count = len(tokens)
    for n in range(min_n, max_n + 1):
        if token_count < n:
            continue
        for i in range(0, token_count - n + 1):
            ngram = " ".join(tokens[i : i + n])
            if ngram:
                ngrams.append(ngram)
    return tuple(ngrams)


def _to_token_set(value: Any) -> set[str]:
    if isinstance(value, set):
        return value
    if isinstance(value, (list, tuple)):
        return {str(token) for token in value if token}
    return set()


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_generic_metric(value: float) -> float:
    if value <= 1.0:
        return clamp01(value)
    if value <= 5.0:
        return clamp01(value / 5.0)
    if value <= 100.0:
        return clamp01(value / 100.0)
    return clamp01(math.log1p(value) / math.log(1001.0))


def _normalize_count_metric(value: float) -> float:
    return clamp01(math.log1p(value) / math.log(1001.0))


def _metadata_value(document: Mapping[str, Any], key: str) -> Any:
    metadata = document.get("metadata")
    if isinstance(metadata, Mapping):
        return metadata.get(key)
    return None


def _extract_document_datetime(document: Mapping[str, Any]) -> datetime | None:
    date_keys = (
        "review_date",
        "timestamp",
        "scraped_at",
        "crawled_at",
        "published_at",
        "updated_at",
    )
    for key in date_keys:
        parsed = _parse_datetime(document.get(key))
        if parsed is not None:
            return parsed
        parsed = _parse_datetime(_metadata_value(document, key))
        if parsed is not None:
            return parsed

    metadata = document.get("metadata")
    if isinstance(metadata, Mapping):
        json_ld = metadata.get("jsonld")
        if isinstance(json_ld, Mapping):
            parsed = _parse_datetime(json_ld.get("datePublished"))
            if parsed is not None:
                return parsed
    return None


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp <= 0:
            return None
        if timestamp > 10_000_000_000:
            timestamp /= 1000.0
        try:
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (ValueError, OSError):
            return None

    text = str(value).strip()
    if not text:
        return None
    iso_text = _ISO_Z_SUFFIX_RE.sub("+00:00", text)

    parsed: datetime | None = None
    try:
        parsed = datetime.fromisoformat(iso_text)
    except ValueError:
        parsed = None

    if parsed is None:
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%d %b %Y", "%d %B %Y"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue

    if parsed is None:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _contains_any(text: str, fragments: tuple[str, ...]) -> bool:
    return any(fragment in text for fragment in fragments)
