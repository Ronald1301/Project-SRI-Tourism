from __future__ import annotations

import logging
from pathlib import Path
import time

from ddgs import DDGS
import requests

from src.utils.file_manager import load_visited_urls
from src.web_crawler.config import DEFAULT_VISITED_URLS_PATH
from src.web_crawler.scraper import extract_document

logger = logging.getLogger("src.web_crawler.web_search_client")


class DuckDuckGoWebSearchClient:
    """
    Lightweight web search client based on ddgs.
    """

    def __init__(
        self,
        *,
        timeout: float = 12.0,
        user_agent: str = "SRI-Tourism-WebSearch/1.0",
        visited_urls_path: Path | None = None,
    ) -> None:
        self.timeout = float(timeout)
        self.user_agent = user_agent
        self.visited_urls_path = Path(visited_urls_path or DEFAULT_VISITED_URLS_PATH)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def search(self, query: str, max_results: int = 5) -> list[dict[str, object]]:
        if not query or not query.strip():
            logger.warning("search() llamada con query vacia")
            return []

        documents: list[dict[str, object]] = []
        urls: list[str] = []
        seen_urls: set[str] = set()
        visited_urls = load_visited_urls(self.visited_urls_path)
        documents_count = 0

        logger.info("Consultando DuckDuckGo para: \"%s\" (max=%d)", query, max_results)
        try:
            with DDGS(timeout=self.timeout) as ddgs:
                hits = ddgs.text(query.strip(), max_results=max(int(max_results), 0))
                for hit in hits:
                    url = str(hit.get("href") or hit.get("url") or "").strip()
                    if not url:
                        continue
                    if url in seen_urls:
                        continue
                    if url in visited_urls:
                        continue

                    urls.append(url)
                    seen_urls.add(url)
                    if len(urls) >= max(int(max_results), 0):
                        break
        except Exception as exc:
            logger.warning("Error en busqueda DuckDuckGo: %s", exc)
            return []

        logger.info("DuckDuckGo retorno %d URLs unicas", len(urls))
        total = len(urls)
        for i, url in enumerate(urls, start=1):
            html = None
            logger.info("Extrayendo [%d/%d] %s", i, total, url)
            time.sleep(2)
            try:
                resp = self.session.get(
                    url,
                    headers={"User-Agent": self.user_agent},
                    timeout=self.timeout
                )
                if resp.status_code == 200:
                    html = resp.text
                else:
                    logger.info("  HTTP %d, saltando", resp.status_code)
                    continue
            except Exception as exc:
                logger.info("  Error HTTP: %s, saltando", exc)
                continue

            if html is None:
                continue

            document = extract_document(html, url)
            documents_count += 1
            logger.info("  Documento extraido (%d/%d)", documents_count, total)
            documents.append(document)
            self._append_visited_url(url)

        logger.info("Busqueda web completa: %d documentos extraidos de %d URLs", documents_count, total)
        return documents

    def _append_visited_url(self, url: str) -> None:
        if not url:
            return
        try:
            self.visited_urls_path.parent.mkdir(parents=True, exist_ok=True)
            with self.visited_urls_path.open("a", encoding="utf-8") as file:
                file.write(url)
                file.write("\n")
        except OSError:
            return
