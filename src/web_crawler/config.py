from __future__ import annotations

from dataclasses import dataclass,field
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

DEFAULT_EXCLUDE_URL_PATTERNS = [
    r"\.pdf$",
    r"\.jpg$",
    r"\.jpeg$",
    r"\.png$",
    r"\.gif$",
    r"\.svg$",
    r"\.webp$",
    r"\.zip$",
    r"\.rar$",
    r"\.mp4$",
    r"\.mp3$",
    r"/wiki/File:",
    r"/wiki/Category:",
    r"/wiki/Categor[ií]a:",
    r"/wiki/Special:",
    r"/wiki/Template:",
]

@dataclass(slots=True)
class CrawlerConfig:
    seed_urls : list[str]
    allowed_domains : set[str] = field(default_factory=set)
    max_depth : int = 2
    max_pages : int = 100
    request_delay : float = 1.0
    timeout : float = 10.0
    user_agent : str = "SRI-Tourism-Crawler/1.0 (+academic-project)"
    output_dir : Path = Path("data/raw/crawl")
    obey_robots : bool = True
    save_html : bool = True
    include_url_patterns : list[str] = field(default_factory=list)
    exclude_url_patterns : list[str] = field(default_factory=lambda: DEFAULT_EXCLUDE_URL_PATTERNS.copy())
    allowed_schemes : tuple[str, ...] = ("http", "https")

    @classmethod
    def from_iterables(
        cls,
        seed_urls: Iterable[str],
        allowed_domains: Iterable[str] | None = None,
        **kwargs: object,
    ) -> "CrawlerConfig":
        clean_seeds = [url.strip() for url in seed_urls if url and url.strip()]
        if not clean_seeds:
            msg = "At least one seed URL is required"
            raise ValueError(msg)
        
        clean_domains : set[str] = set()
        for domain in allowed_domains or []:
            if not domain:
                continue
            value = domain.strip().lower()
            if not value:
                continue
            if "://" in value:
                value = urlparse(value).hostname or value
            value = value.split("/")[0]
            if value:
                clean_domains.add(value)
        return cls(seed_urls=clean_seeds, allowed_domains=clean_domains, **kwargs)
