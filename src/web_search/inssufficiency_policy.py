from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class InsufficiencyPolicy:
    min_local_results: int = 3
    min_top_score: float = 0.28
    min_avg_score_top3: float = 0.22

    def is_insufficient(self, local_results: list[dict[str, Any]]) -> bool:
        if len(local_results) < self.min_local_results:
            return True
        top_score = float(local_results[0].get("score", 0.0))
        if top_score < self.min_top_score:
            return True
        top3 = local_results[:3]
        avg_top3 = sum(float(item.get("score", 0.0)) for item in top3) / max(len(top3), 1)
        return avg_top3 < self.min_avg_score_top3