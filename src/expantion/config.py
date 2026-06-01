from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_EXPANSION_CONFIG_PATH = Path("src/config/expansion_config.json")


@dataclass(frozen=True)
class ExpansionConfig:
    data: dict[str, Any]
    path: Path = DEFAULT_EXPANSION_CONFIG_PATH

    def get(self, dotted_key: str, default: Any = None) -> Any:
        current: Any = self.data
        for part in dotted_key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    @property
    def max_terms(self) -> int:
        return int(self.get("global.max_terms_to_add", 5))

    @property
    def acceptance_threshold(self) -> float:
        return float(self.get("global.acceptance_threshold", 0.75))

    @property
    def language(self) -> str:
        return str(self.get("global.language", "auto"))

    @property
    def feedback_db_path(self) -> Path:
        return Path(str(self.get("feedback_database.path", "data/feedback/query_feedback.db")))

    @property
    def synonyms_path(self) -> Path:
        return Path(str(self.get("synonyms.path", "src/expantion/synonyms_tourism.json")))


class ConfigLoader:
    def load(self, config_path: Path | str | None = None) -> ExpansionConfig:
        path = Path(config_path or DEFAULT_EXPANSION_CONFIG_PATH)
        if not path.exists():
            return ExpansionConfig(data=self._default_data(), path=path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"No se pudo cargar la configuracion de expansion: {path}") from exc
        data = self._deep_merge(self._default_data(), payload)
        return ExpansionConfig(data=data, path=path)

    def _default_data(self) -> dict[str, Any]:
        return {
            "global": {
                "max_terms_to_add": 5,
                "acceptance_threshold": 0.75,
                "top_documents_for_prf": 3,
                "cache_enabled": True,
                "cache_max_size": 256,
                "cache_ttl_seconds": 600,
                "language": "auto",
            },
            "techniques": {
                "synonyms": {"enabled": True, "weight": 0.85},
                "pseudo_relevance": {
                    "enabled": True,
                    "title_weight": 0.7,
                    "body_weight": 0.35,
                    "max_title_terms": 24,
                    "max_body_terms": 12,
                },
                "cooccurrence": {"enabled": True, "weight": 0.28, "window_size": 4},
                "feedback": {
                    "enabled": True,
                    "explicit_positive_weight": 0.95,
                    "explicit_negative_weight": -0.75,
                    "implicit_default_weight": 0.2,
                },
                "rocchio": {
                    "enabled": True,
                    "alpha": 1.0,
                    "beta_with_explicit": 0.75,
                    "beta_without_explicit": 0.25,
                    "gamma": 0.15,
                    "candidate_weight": 0.8,
                    "max_vector_terms": 18,
                    "debug_vectors": False,
                },
            },
            "ngrams": {
                "enabled": True,
                "min_n": 1,
                "max_n": 3,
                "multipliers": {"1": 1.0, "2": 1.2, "3": 1.3},
            },
            "logging": {
                "enabled": True,
                "level": "INFO",
                "trace_candidates": True,
                "trace_rocchio_errors": True,
                "trace_cache": True,
            },
            "feedback_database": {"path": "data/feedback/query_feedback.db"},
            "synonyms": {
                "path": "src/expantion/synonyms_tourism.json",
                "bidirectional": True,
            },
        }

    def _deep_merge(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        result = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
