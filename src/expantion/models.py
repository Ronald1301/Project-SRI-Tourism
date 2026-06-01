from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    trace: dict[str, Any] | None = None

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
            "trace": self.trace or {},
        }
