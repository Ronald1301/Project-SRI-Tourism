from __future__ import annotations

import json
from pathlib import Path

from src.expantion.config import ExpansionConfig
from src.expantion.constants import get_domain_synonyms
from src.expantion.text import normalize_key


class SynonymLoader:
    def __init__(self, config: ExpansionConfig) -> None:
        self.config = config

    def load(self, synonyms_path: Path | str | None = None) -> dict[str, list[str]]:
        path = Path(synonyms_path or self.config.synonyms_path)
        raw = self._load_json(path) if path.exists() else self._fallback_synonyms()
        synonyms = self._flatten(raw)
        if bool(self.config.get("synonyms.bidirectional", True)):
            synonyms = self._make_bidirectional(synonyms)
        return synonyms

    def _load_json(self, path: Path) -> dict:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"No se pudo cargar el diccionario de sinonimos: {path}") from exc

    def _fallback_synonyms(self) -> dict[str, list[str]]:
        return get_domain_synonyms(str(self.config.get("global.language", "auto")))

    def _flatten(self, payload: dict) -> dict[str, list[str]]:
        flattened: dict[str, list[str]] = {}
        for value in payload.values():
            if not isinstance(value, dict):
                continue
            for term, synonyms in value.items():
                key = normalize_key(term)
                if not key:
                    continue
                flattened.setdefault(key, [])
                for synonym in synonyms or []:
                    normalized = normalize_key(str(synonym))
                    if normalized and normalized != key and normalized not in flattened[key]:
                        flattened[key].append(normalized)
        if not flattened:
            for term, synonyms in self._fallback_synonyms().items():
                flattened[normalize_key(term)] = [normalize_key(item) for item in synonyms]
        return flattened

    def _make_bidirectional(self, synonyms: dict[str, list[str]]) -> dict[str, list[str]]:
        expanded = {term: list(values) for term, values in synonyms.items()}
        for term, values in synonyms.items():
            for synonym in values:
                reverse = expanded.setdefault(synonym, [])
                if term not in reverse:
                    reverse.append(term)
        return expanded
