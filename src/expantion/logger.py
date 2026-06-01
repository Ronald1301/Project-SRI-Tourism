from __future__ import annotations

import logging
from typing import Any

from src.expantion.config import ExpansionConfig


class ExpansionLogger:
    def __init__(self, config: ExpansionConfig) -> None:
        self.config = config
        self.enabled = bool(config.get("logging.enabled", True))
        self.logger = logging.getLogger("src.expantion")
        level_name = str(config.get("logging.level", "INFO")).upper()
        self.logger.setLevel(getattr(logging, level_name, logging.INFO))

    def info(self, message: str, *args: Any) -> None:
        if self.enabled:
            self.logger.info(message, *args)

    def debug(self, message: str, *args: Any) -> None:
        if self.enabled:
            self.logger.debug(message, *args)

    def warning(self, message: str, *args: Any) -> None:
        if self.enabled:
            self.logger.warning(message, *args)

    def trace_candidates(self, technique: str, candidates: dict[str, float]) -> None:
        if self.enabled and bool(self.config.get("logging.trace_candidates", True)):
            top = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))[:8]
            self.logger.debug("Expansion %s candidates=%s", technique, top)

    def rocchio_error(self, query: str, exc: Exception) -> None:
        if self.enabled and bool(self.config.get("logging.trace_rocchio_errors", True)):
            self.logger.warning("Rocchio expansion failed | query=%r | error=%s", query, exc)

    def cache(self, event: str, key: str) -> None:
        if self.enabled and bool(self.config.get("logging.trace_cache", True)):
            self.logger.debug("Expansion cache %s | key=%s", event, key)
