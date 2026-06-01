from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any


class CacheManager:
    def __init__(self, *, max_size: int = 256, ttl_seconds: int = 600, enabled: bool = True) -> None:
        self.max_size = max(int(max_size), 1)
        self.ttl_seconds = max(int(ttl_seconds), 1)
        self.enabled = bool(enabled)
        self._items: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        item = self._items.get(key)
        if item is None:
            return None
        created_at, value = item
        if time.time() - created_at > self.ttl_seconds:
            self._items.pop(key, None)
            return None
        self._items.move_to_end(key)
        return value

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        self._items[key] = (time.time(), value)
        self._items.move_to_end(key)
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()
