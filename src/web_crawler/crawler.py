"""Motor principal de crawling y scraping del proyecto."""

from __future__ import annotations
import logging
import time
from collections import deque
from datetime import UTC, datetime
from threading import Lock
from typing import Any
from pathlib import Path
from urllib.parse import urlparse
import requests
from .config import CrawlerConfig
from .policies import CrawlPolicies
from .scraper import extract_document, extract_links
from .storage import CrawlStorage

logger = logging.getLogger("src.web_crawler.crawler")


class WebCrawler:
    """Ejecuta crawling BFS con politicas, persistencia y filtros de calidad."""

    _VISITED_PATH_LOCKS: dict[str, Lock] = {}
    _VISITED_PATH_LOCKS_GUARD = Lock()
    _ALLOWED_LANGUAGE_PREFIXES = ("es", "en")
    _MIN_WORD_COUNT = 50

    def __init__(self, config: CrawlerConfig, site_name: str = "default"):
        """Inicializa el crawler para un sitio concreto.

        Args:
            config: Configuracion completa de crawling.
            site_name: Nombre logico del sitio para logging.

        Returns:
            None
        """
        self.config = config
        self.site_name = site_name
        self.logger = logging.getLogger(f"src.web_crawler.crawler.{site_name}")
        self.policies = CrawlPolicies(config)
        self.storage = CrawlStorage(Path("data/raw"))
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": config.user_agent})
        self.session.max_redirects = config.max_redirects

        self.last_request_at: dict[str, float] = {}
        self.queued: set[str] = set()
        self.visited_urls: set[str] = set()
        if self.config.persist_visited:
            self.visited_urls = self.load_visited_urls(self.config.visited_urls_path)

        self.stats: dict[str, int] = {
            "seed_urls": len(config.seed_urls),
            "urls_visited": 0,
            "pages_fetched": 0,
            "documents_saved": 0,
            "links_discovered": 0,
            "links_enqueued": 0,
            "errors": 0,
            "skipped_policy": 0,
            "skipped_robots": 0,
            "skipped_non_html": 0,
            "skipped_duplicate": 0,
            "skipped_persisted": 0,
            "skipped_language": 0,
            "skipped_short_text": 0,
        }

    @classmethod
    def _visited_lock_for_path(cls, path: Path) -> Lock:
        """Obtiene un lock compartido para el archivo de URLs visitadas.

        Args:
            path: Ruta del archivo a sincronizar.

        Returns:
            Lock: Lock asociado a la ruta.
        """
        key = str(path.resolve())
        with cls._VISITED_PATH_LOCKS_GUARD:
            lock = cls._VISITED_PATH_LOCKS.get(key)
            if lock is None:
                lock = Lock()
                cls._VISITED_PATH_LOCKS[key] = lock
            return lock

    def respect_delay(self, url: str) -> None:
        """Respeta el retraso entre requests para un host.

        Args:
            url: URL destino de la solicitud.

        Returns:
            None
        """
        host = urlparse(url).netloc.lower()
        now = time.monotonic()
        last = self.last_request_at.get(host)
        if last is not None:
            sleep_time = self.config.request_delay - (now - last)
            if sleep_time > 0:
                time.sleep(sleep_time)
        self.last_request_at[host] = time.monotonic()

    def record_error(self, url: str, depth: int, reason: str, detail: str | None = None) -> None:
        """Registra un error del crawler y actualiza estadisticas.

        Args:
            url: URL afectada.
            depth: Profundidad de exploracion.
            reason: Codigo corto del error.
            detail: Detalle adicional opcional.

        Returns:
            None
        """
        self.stats["errors"] += 1
        self.logger.warning("Error [%s] %s (depth=%d): %s", reason, url, depth, detail or "")

    def fetch(self, url: str, depth: int) -> tuple[str, str, str] | None:
        """Descarga una URL y devuelve URL final, HTML y content-type.

        Args:
            url: URL original a descargar.
            depth: Profundidad de la URL en el arbol de crawling.

        Returns:
            tuple[str, str, str] | None: URL final, HTML y content-type, o `None`.
        """
        self.respect_delay(url)
        try:
            response = self.session.get(url, timeout=self.config.timeout, allow_redirects=True)
        except requests.RequestException as exc:
            self.record_error(url, depth, "request-exception", str(exc))
            return None

        final_url = self.policies.normalize_url(url, response.url) or response.url
        if response.status_code >= 400:
            self.record_error(
                final_url,
                depth,
                "http_error",
                f"status_code={response.status_code}"
            )
            return None

        if not self.policies.is_allowed(final_url):
            self.stats["skipped_policy"] += 1
            return None

        response.encoding = response.apparent_encoding or response.encoding
        content_type = response.headers.get("Content-Type", "").lower()
        return final_url, response.text, content_type

    def crawl(self) -> dict[str, int]:
        """Ejecuta el crawl completo y devuelve estadisticas agregadas.

        Returns:
            dict[str, int]: Conteos de paginas, errores, enlaces y documentos.
        """
        start_time = datetime.now(UTC)
        queue: deque[tuple[str, int, str | None]] = deque()
        visited: set[str] = set()

        for seed in self.config.seed_urls:
            normalized = self.policies.normalize_url(seed, seed)
            if not normalized:
                continue
            if not self.policies.is_allowed(normalized):
                self.stats["skipped_policy"] += 1
                continue
            queue.append((normalized, 0, None))
            self.queued.add(normalized)

        while queue and self.stats["pages_fetched"] < self.config.max_pages:
            current_url, depth, parent_url = queue.popleft()
            self.queued.discard(current_url)

            if current_url in visited:
                self.stats["skipped_duplicate"] += 1
                continue

            if not self.policies.is_allowed(current_url):
                self.stats["skipped_policy"] += 1
                continue

            if not self.policies.is_allowed_by_robots(current_url, self.config.user_agent):
                self.stats["skipped_robots"] += 1
                continue

            if self.config.persist_visited and current_url in self.visited_urls:
                visited.add(current_url)
                self.stats["urls_visited"] += 1
                self.stats["skipped_persisted"] += 1

                fetched = self.fetch(current_url, depth)
                if fetched is None:
                    continue
                final_url, html, content_type = fetched
                if "text/html" not in content_type:
                    self.stats["skipped_non_html"] += 1
                    continue
                self.stats["pages_fetched"] += 1

                links = extract_links(html, final_url, self.policies)
                self.stats["links_discovered"] += len(links)
                for link in links:
                    if link in visited or link in self.queued:
                        continue
                    queue.append((link, depth + 1, final_url))
                    self.queued.add(link)
                    self.stats["links_enqueued"] += 1

                if self.config.persist_visited:
                    self.append_visited_url(final_url)

                self.print_progress()
                continue

            visited.add(current_url)
            self.stats["urls_visited"] += 1

            fetched = self.fetch(current_url, depth)
            if fetched is None:
                continue
            final_url, html, content_type = fetched

            if "text/html" not in content_type:
                self.stats["skipped_non_html"] += 1
                continue

            self.stats["pages_fetched"] += 1
            document = extract_document(html, final_url)
            document["depth"] = depth
            document["parent_url"] = parent_url

            save_document = True
            if not self._has_supported_language(document):
                self.stats["skipped_language"] += 1
                save_document = False
            if not self._has_minimum_words(document):
                self.stats["skipped_short_text"] += 1
                save_document = False

            if save_document:
                self.storage.append_document(document)
                self.stats["documents_saved"] += 1
                self.print_progress()
            if self.config.persist_visited:
                self.append_visited_url(final_url)

            if depth >= self.config.max_depth:
                continue

            links = extract_links(html, final_url, self.policies)
            self.stats["links_discovered"] += len(links)
            for link in links:
                if link in visited or link in self.queued:
                    continue
                queue.append((link, depth + 1, final_url))
                self.queued.add(link)
                self.stats["links_enqueued"] += 1

        elapsed = (datetime.now(UTC) - start_time).total_seconds()
        self.logger.info(
            "Crawl completo | documentos=%d paginas=%d errores=%d urls=%d enqueued=%d segundos=%.1f",
            self.stats["documents_saved"],
            self.stats["pages_fetched"],
            self.stats["errors"],
            self.stats["urls_visited"],
            self.stats["links_enqueued"],
            elapsed,
        )
        return self.stats

    def _has_supported_language(self, document: dict[str, object]) -> bool:
        """Verifica si el documento esta en espanol o ingles.

        Args:
            document: Documento extraido del HTML.

        Returns:
            bool: `True` si el idioma es soportado.
        """
        language = str(document.get("language") or "").strip().lower()
        if not language:
            return False
        normalized = language.split("-")[0]
        return normalized in self._ALLOWED_LANGUAGE_PREFIXES

    def _has_minimum_words(self, document: dict[str, object]) -> bool:
        """Verifica si el documento supera el umbral minimo de palabras.

        Args:
            document: Documento extraido del HTML.

        Returns:
            bool: `True` si el conteo es suficiente.
        """
        raw_count = document.get("word_count")
        try:
            count = int(raw_count) if raw_count is not None else 0
        except (TypeError, ValueError):
            count = 0
        return count >= self._MIN_WORD_COUNT

    def print_progress(self) -> None:
        """Imprime progreso periodico del crawl.

        Returns:
            None
        """
        pages = self.stats["pages_fetched"]
        if pages <= 0:
            return

        by_pages = (
            self.config.progress_every_pages > 0
            and pages % self.config.progress_every_pages == 0
        )
        if not by_pages:
            return
        self.logger.info(
            "Progreso: pages=%d saved=%d queued=%d visited=%d",
            pages,
            self.stats["documents_saved"],
            len(self.queued),
            self.stats["urls_visited"],
        )

    def load_visited_urls(self, path: Path) -> set[str]:
        """Carga URLs visitadas desde disco.

        Args:
            path: Archivo de URLs visitadas.

        Returns:
            set[str]: URLs ya procesadas.
        """
        urls: set[str] = set()
        file_lock = self._visited_lock_for_path(path)
        try:
            if not path.exists():
                return urls
            with file_lock:
                with path.open("r", encoding="utf-8") as file:
                    for line in file:
                        url = line.strip()
                        if url:
                            urls.add(url)
        except OSError:
            return urls
        return urls

    def append_visited_url(self, url: str) -> None:
        """Agrega una URL al archivo compartido de visitadas.

        Args:
            url: URL a persistir.

        Returns:
            None
        """
        if not url or url in self.visited_urls:
            return
        self.visited_urls.add(url)
        path = self.config.visited_urls_path
        file_lock = self._visited_lock_for_path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with file_lock:
                with path.open("a", encoding="utf-8") as file:
                    file.write(url)
                    file.write("\n")
        except OSError:
            return
