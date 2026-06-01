"""Configuracion base del crawler web.

Este modulo define el contrato de entrada para cada ejecucion de crawling.
"""

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
    # r"/wiki/File:",
    # r"/wiki/Category:",
    # r"/wiki/Categor[ií]a:",
    # r"/wiki/Special:",
    # r"/wiki/Template:",
]

DEFAULT_VISITED_URLS_PATH = Path(__file__).resolve().parent / "visited_urls.txt"

@dataclass(slots=True)
class CrawlerConfig:
    """Contenedor de parametros del crawler.

    Cada atributo configura un aspecto del rastreo: semillas, dominios,
    limites de profundidad, persistencia y politicas de red.
    """

    seed_urls : list[str]
    allowed_domains : set[str] = field(default_factory=set)
    max_depth : int = 2
    max_pages : int = 100
    request_delay : float = 1.0
    timeout : float = 10.0
    user_agent : str = "SRI-Tourism-Crawler/1.0 (+academic-project)"
    output_dir : Path = Path("data/rawl")
    obey_robots : bool = True
    save_html : bool = True
    include_url_patterns : list[str] = field(default_factory=list)
    exclude_url_patterns : list[str] = field(default_factory=lambda: DEFAULT_EXCLUDE_URL_PATTERNS.copy())
    allowed_schemes : tuple[str, ...] = ("http", "https")
    progress_every_pages : int = 10
    persist_visited : bool = True
    visited_urls_path : Path = DEFAULT_VISITED_URLS_PATH
    max_redirects : int = 10

    @classmethod
    def from_iterables(
        cls,
        seed_urls: Iterable[str],
        allowed_domains: Iterable[str] | None = None,
        **kwargs: object,
    ) -> "CrawlerConfig":
        """Construye un `CrawlerConfig` a partir de iterables.

        Args:
            seed_urls: Secuencia de URLs semilla.
            allowed_domains: Dominios permitidos. Puede ser `None`.
            **kwargs: Parametros adicionales del dataclass.

        Returns:
            CrawlerConfig: Configuracion normalizada y valida.

        Raises:
            ValueError: Si no se proporciona al menos una URL semilla valida.
        """
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
