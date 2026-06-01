from __future__ import annotations


class NGramExtractor:
    def __init__(
        self,
        *,
        enabled: bool = True,
        min_n: int = 1,
        max_n: int = 3,
        multipliers: dict[str, float] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.min_n = max(int(min_n), 1)
        self.max_n = max(int(max_n), self.min_n)
        self.multipliers = multipliers or {"1": 1.0, "2": 1.2, "3": 1.3}

    def extract(self, tokens: list[str], *, max_items: int | None = None) -> list[tuple[str, float]]:
        if not self.enabled:
            return [(token, 1.0) for token in tokens[: max_items or len(tokens)]]
        grams: list[tuple[str, float]] = []
        for n in range(self.min_n, self.max_n + 1):
            if n <= 0 or len(tokens) < n:
                continue
            multiplier = float(self.multipliers.get(str(n), 1.0))
            for index in range(0, len(tokens) - n + 1):
                grams.append((" ".join(tokens[index : index + n]), multiplier))
        return grams[:max_items] if max_items is not None else grams
