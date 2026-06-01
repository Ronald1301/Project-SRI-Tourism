"""Busqueda web acotada y filtrada para ampliar el corpus del sistema."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time
from threading import Lock
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.stemmer import Stemmer
from src.preprocessing.tokenizer import Tokenizer, load_english_stopwords, load_spanish_stopwords
from src.web_crawler.config import DEFAULT_VISITED_URLS_PATH
from src.web_crawler.config import CrawlerConfig
from src.web_crawler.policies import CrawlPolicies
from src.web_crawler.scraper import extract_document, extract_links
from src.utils.file_manager import load_visited_urls
from src.web_crawler.url_importance_policy import URLImportancePolicy

logger = logging.getLogger("src.web_crawler.web_search_client")


@dataclass(frozen=True)
class SearchCandidate:
    """URL candidata para exploracion BFS durante la busqueda web."""

    url: str
    depth: int
    title: str = ""
    snippet: str = ""


_VISITED_PATH_LOCKS: dict[str, Lock] = {}
_VISITED_PATH_LOCKS_GUARD = Lock()


def _lock_for_path(path: Path) -> Lock:
    """Obtiene un lock compartido para la ruta dada.

    Args:
        path: Ruta que se desea sincronizar.

    Returns:
        Lock: Lock asociado a la ruta.
    """
    key = str(path.resolve())
    with _VISITED_PATH_LOCKS_GUARD:
        lock = _VISITED_PATH_LOCKS.get(key)
        if lock is None:
            lock = Lock()
            _VISITED_PATH_LOCKS[key] = lock
        return lock


class DuckDuckGoWebSearchClient:
    """Cliente de busqueda web basado en DuckDuckGo y politicas de filtrado."""

    EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(
        self,
        *,
        timeout: float = 12.0,
        user_agent: str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
        visited_urls_path: Path | None = None,
        url_importance_policy: URLImportancePolicy | None = None,
        cuba_min_score: float = 0.40,
        query_min_score: float = 0.50,
        min_word_count: int = 200,
        max_boilerplate_ratio: float = 0.55,
        max_link_density: float = 0.25,
        request_delay_seconds: float = 0.4,
        max_pages: int = 30,
        follow_links_depth: int = 1,
    ) -> None:
        """Inicializa el cliente de busqueda web.

        Args:
            timeout: Tiempo maximo por request HTTP.
            user_agent: Cadena User-Agent usada en las solicitudes.
            visited_urls_path: Archivo compartido de URLs visitadas.
            url_importance_policy: Politica heuristica de importancia.
            cuba_min_score: Umbral minimo para el primer filtro semantico contra "Cuba".
            query_min_score: Umbral minimo para el segundo filtro semantico contra la query.
            min_word_count: Minimo de palabras para conservar un documento.
            max_boilerplate_ratio: Umbral maximo de boilerplate.
            max_link_density: Umbral maximo de densidad de enlaces.
            request_delay_seconds: Pausa entre solicitudes al mismo host.
            max_pages: Maximo de paginas procesadas.
            follow_links_depth: Profundidad maxima de seguimiento de enlaces.

        Returns:
            None
        """
        self.timeout = float(timeout)
        self.user_agent = user_agent
        self.visited_urls_path = Path(visited_urls_path or DEFAULT_VISITED_URLS_PATH)
        self.url_importance_policy = url_importance_policy or URLImportancePolicy()
        self.cuba_min_score = max(float(cuba_min_score), 0.0)
        self.query_min_score = max(float(query_min_score), 0.0)
        self.min_word_count = max(int(min_word_count), 1)
        self.max_boilerplate_ratio = min(max(float(max_boilerplate_ratio), 0.0), 1.0)
        self.max_link_density = min(max(float(max_link_density), 0.0), 1.0)
        self.request_delay_seconds = max(float(request_delay_seconds), 0.0)
        self.max_pages = max(int(max_pages), 1)
        self.follow_links_depth = max(int(follow_links_depth), 0)
        self._cleaner = TextCleaner()
        self._english_stopwords = load_english_stopwords()
        self._spanish_stopwords = load_spanish_stopwords()
        self._english_tokenizer = Tokenizer(language="english", stopwords=self._english_stopwords)
        self._spanish_tokenizer = Tokenizer(language="spanish", stopwords=self._spanish_stopwords)
        self._english_stemmer = Stemmer(language="english")
        self._spanish_stemmer = Stemmer(language="spanish")
        self.policies = CrawlPolicies(
            CrawlerConfig(
                seed_urls=["https://example.com"],
                allowed_domains=set(),
                max_depth=max(self.follow_links_depth, 1),
                max_pages=1000,
                request_delay=self.request_delay_seconds,
                timeout=self.timeout,
                user_agent=user_agent,
                output_dir=Path("data/raw"),
                obey_robots=True,
                save_html=False,
                include_url_patterns=[],
                exclude_url_patterns=[],
                allowed_schemes=("http", "https"),
                progress_every_pages=10,
                persist_visited=False,
                visited_urls_path=self.visited_urls_path,
                max_redirects=10,
            )
        )
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self._embedding_model: SentenceTransformer | None = None
        self._embedding_model_error: str | None = None

    def search(self, query: str, max_results: int = 5) -> list[dict[str, object]]:
        """Busca documentos web y aplica dos filtros consecutivos.

        Args:
            query: Consulta del usuario.
            max_results: Numero maximo de resultados iniciales a pedir a DuckDuckGo HTML.

        Returns:
            list[dict[str, object]]: Documentos filtrados y rankeados.
        """
        if not query or not query.strip():
            logger.warning("search() llamada con query vacia")
            return []

        ddg_query = self._build_ddg_query(query)
        documents: list[dict[str, object]] = []
        queue: deque[SearchCandidate] = deque()
        seen_urls: set[str] = set()
        visited_urls = load_visited_urls(self.visited_urls_path)
        documents_count = 0
        pages_processed = 0

        logger.info("Consultando DuckDuckGo HTML para: \"%s\" (max=%d)", ddg_query, max_results)
        hits = self._search_duckduckgo_html(ddg_query, max_results=max(int(max_results), 0))
        for hit in hits:
            url = str(hit.get("url") or "").strip()
            title = str(hit.get("title") or "").strip()
            snippet = str(hit.get("snippet") or "").strip()
            if not url:
                continue
            if url in seen_urls:
                continue
            if url in visited_urls:
                continue
            decision = self.url_importance_policy.evaluate(
                url=url,
                query=query,
                title=title,
                snippet=snippet,
            )
            if not decision.is_important:
                logger.info(
                    "URL descartada por baja importancia (score=%.4f, reason=%s): %s",
                    decision.score,
                    decision.reason,
                    url,
                )
                continue

            seen_urls.add(url)
            queue.append(SearchCandidate(url=url, depth=0, title=title, snippet=snippet))
            if len(queue) >= min(max(int(max_results), 0), self.max_pages):
                break

        logger.info("DuckDuckGo HTML retorno %d URLs unicas", len(queue))
        while queue and pages_processed < self.max_pages:
            candidate = queue.popleft()
            if candidate.url in visited_urls:
                continue
            if candidate.url in seen_urls and candidate.depth > 0:
                # Ya fue descubierta desde otro enlace, pero no necesariamente visitada.
                pass

            if not self.policies.is_allowed(candidate.url):
                logger.info("URL descartada por politicas generales: %s", candidate.url)
                continue
            if not self.policies.is_allowed_by_robots(candidate.url, self.user_agent):
                logger.info("URL descartada por robots.txt: %s", candidate.url)
                continue

            html = self._fetch_html(candidate.url, timeout=self.timeout)
            if html is None:
                continue

            pages_processed += 1
            document = extract_document(html, candidate.url)
            document = self._apply_document_quality_filters(html, document)
            if document is None:
                continue
            documents_count += 1
            logger.info("  Documento extraido (%d)", documents_count)
            documents.append(document)
            self._append_visited_url(candidate.url)
            visited_urls.add(candidate.url)

            if candidate.depth < self.follow_links_depth:
                link_candidates = self._extract_link_candidates(html, candidate.url)
                for link_url, link_title, link_snippet in link_candidates:
                    if pages_processed >= self.max_pages:
                        break
                    if link_url in visited_urls or link_url in seen_urls:
                        continue
                    if not self.policies.is_allowed(link_url):
                        continue
                    if not self.policies.is_allowed_by_robots(link_url, self.user_agent):
                        continue
                    decision = self.url_importance_policy.evaluate(
                        url=link_url,
                        query=query,
                        title=link_title,
                        snippet=link_snippet,
                    )
                    if not decision.is_important:
                        continue
                    seen_urls.add(link_url)
                    queue.append(SearchCandidate(url=link_url, depth=candidate.depth + 1, title=link_title, snippet=link_snippet))

            if self.request_delay_seconds > 0:
                time.sleep(self.request_delay_seconds)

        logger.info(
            "Busqueda web completa: %d documentos extraidos | paginas_procesadas=%d | depth_limit=%d",
            documents_count,
            pages_processed,
            self.follow_links_depth,
        )
        filtered_documents = self._apply_embedding_filter(
            query=query,
            documents=documents,
            max_results=max_results,
        )
        logger.info(
            "Embedding filter aplicado: %d -> %d documentos (cuba>=%.2f, query>=%.2f)",
            len(documents),
            len(filtered_documents),
            self.cuba_min_score,
            self.query_min_score,
        )
        return filtered_documents

    def _search_duckduckgo_html(self, query: str, *, max_results: int) -> list[dict[str, str]]:
        """Consulta la pagina HTML de DuckDuckGo y extrae resultados.

        Args:
            query: Consulta ya enriquecida para DuckDuckGo.
            max_results: Maximo de resultados a devolver.

        Returns:
            list[dict[str, str]]: Resultados con `url`, `title` y `snippet`.
        """
        if max_results <= 0:
            return []

        search_url = "https://html.duckduckgo.com/html/"
        try:
            response = self.session.get(
                search_url,
                params={"q": query},
                timeout=self.timeout,
                allow_redirects=True,
            )
        except Exception as exc:
            logger.warning("Error consultando DuckDuckGo HTML: %s", exc)
            return []

        if response.status_code != 200:
            logger.warning("DuckDuckGo HTML respondio HTTP %d", response.status_code)
            return []

        response.encoding = response.apparent_encoding or response.encoding
        soup = BeautifulSoup(response.text, "lxml")
        results: list[dict[str, str]] = []

        for result in soup.select("div.result"):
            link = result.select_one("a.result__a")
            if link is None:
                continue

            href = str(link.get("href") or "").strip()
            url = self._normalize_duckduckgo_result_url(href)
            if not url:
                continue

            title = link.get_text(" ", strip=True)
            snippet_node = result.select_one(".result__snippet")
            snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""

            results.append({"url": url, "title": title, "snippet": snippet})
            if len(results) >= max_results:
                break

        return results

    def _normalize_duckduckgo_result_url(self, href: str) -> str:
        """Convierte un enlace de resultado de DuckDuckGo a URL final.

        Args:
            href: Enlace obtenido del HTML de DuckDuckGo.

        Returns:
            str: URL final o cadena vacia si no puede resolverse.
        """
        if not href:
            return ""

        normalized_href = href.strip()
        if normalized_href.startswith("//"):
            normalized_href = urljoin("https:", normalized_href)

        parsed = urlparse(normalized_href)
        if parsed.scheme in {"http", "https"} and parsed.netloc and "duckduckgo.com" not in parsed.netloc:
            return normalized_href

        query = parse_qs(parsed.query)
        uddg = query.get("uddg")
        if uddg:
            return unquote(uddg[0]).strip()

        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return normalized_href

        return ""

    def _fetch_html(self, url: str, *, timeout: float) -> str | None:
        """Descarga HTML si la URL responde correctamente.

        Args:
            url: URL objetivo.
            timeout: Timeout en segundos para la peticion.

        Returns:
            str | None: HTML descargado o `None` en caso de error.
        """
        logger.info("Extrayendo %s", url)
        try:
            resp = self.session.get(
                url,
                headers={"User-Agent": self.user_agent},
                timeout=timeout,
                allow_redirects=True,
            )
        except Exception as exc:
            logger.info("  Error HTTP: %s, saltando", exc)
            return None

        if resp.status_code != 200:
            logger.info("  HTTP %d, saltando", resp.status_code)
            return None

        content_type = (resp.headers.get("Content-Type") or "").lower()
        if "text/html" not in content_type:
            logger.info("  Contenido no HTML (%s), saltando", content_type)
            return None

        resp.encoding = resp.apparent_encoding or resp.encoding
        return resp.text

    def _extract_link_candidates(self, html: str, base_url: str) -> list[tuple[str, str, str]]:
        """Obtiene candidatos de enlace a partir de un HTML.

        Args:
            html: Contenido HTML.
            base_url: URL base para resolver enlaces relativos.

        Returns:
            list[tuple[str, str, str]]: Tuplas `(url, title, snippet)`.
        """
        links = extract_links(html, base_url, self.policies)
        if not links:
            return []

        # Reutiliza la extracción base del crawler para enlaces; el texto del anchor no
        # siempre está disponible aquí, por eso se usa la URL como señal principal.
        return [(link_url, "", "") for link_url in links]

    def _append_visited_url(self, url: str) -> None:
        """Anade una URL al archivo compartido de visitadas.

        Args:
            url: URL a registrar.

        Returns:
            None
        """
        if not url:
            return
        try:
            self.visited_urls_path.parent.mkdir(parents=True, exist_ok=True)
            file_lock = _lock_for_path(self.visited_urls_path)
            with file_lock:
                with self.visited_urls_path.open("a", encoding="utf-8") as file:
                    file.write(url)
                    file.write("\n")
        except OSError:
            return

    def _apply_document_quality_filters(
        self,
        html: str,
        document: dict[str, object],
    ) -> dict[str, object] | None:
        """Aplica filtros de calidad a un documento descargado.

        Args:
            html: HTML original del documento.
            document: Documento estructurado extraido.

        Returns:
            dict[str, object] | None: Documento enriquecido o `None`.
        """
        content_text = str(document.get("content_text") or "").strip()
        combined_text = "\n".join(
            [
                str(document.get("title") or ""),
                str(document.get("summary") or ""),
                content_text,
            ]
        ).strip()
        word_count = int(document.get("word_count") or self._count_words(content_text))
        if word_count < self.min_word_count:
            logger.info(
                "Documento descartado por texto corto (%d < %d): %s",
                word_count,
                self.min_word_count,
                document.get("url"),
            )
            return None

        detected_language = self._detect_language(document, combined_text)
        if detected_language not in {"es", "en"}:
            logger.info(
                "Documento descartado por idioma no soportado (%s): %s",
                detected_language,
                document.get("url"),
            )
            return None

        boilerplate_ratio = self._compute_boilerplate_ratio(html)
        if boilerplate_ratio > self.max_boilerplate_ratio:
            logger.info(
                "Documento descartado por boilerplate ratio alto (%.3f > %.3f): %s",
                boilerplate_ratio,
                self.max_boilerplate_ratio,
                document.get("url"),
            )
            return None

        link_density = self._compute_link_density(html)
        if link_density > self.max_link_density:
            logger.info(
                "Documento descartado por link density alta (%.3f > %.3f): %s",
                link_density,
                self.max_link_density,
                document.get("url"),
            )
            return None

        filtered = dict(document)
        filtered["language"] = detected_language
        filtered["boilerplate_ratio"] = round(float(boilerplate_ratio), 4)
        filtered["link_density"] = round(float(link_density), 4)
        filtered["word_count"] = word_count
        return filtered

    def _detect_language(self, document: dict[str, object], text: str) -> str:
        """Detecta si el texto parece estar en espanol o ingles.

        Args:
            document: Documento estructurado.
            text: Texto combinado para inferencia.

        Returns:
            str: Codigo de idioma inferido.
        """
        raw_language = str(document.get("language") or "").strip().lower()
        if raw_language:
            normalized = raw_language.replace("_", "-")
            prefix = normalized.split("-", 1)[0]
            if prefix in {"es", "en"}:
                return prefix

        cleaned = self._cleaner.clean(text)
        if not cleaned:
            return "unknown"

        english_tokens = self._english_tokenizer.tokenize(cleaned)
        spanish_tokens = self._spanish_tokenizer.tokenize(cleaned)
        if not english_tokens and not spanish_tokens:
            return "unknown"

        english_hits = sum(1 for token in english_tokens if token in self._english_stopwords)
        spanish_hits = sum(1 for token in spanish_tokens if token in self._spanish_stopwords)

        if english_hits == 0 and spanish_hits == 0:
            return "unknown"
        if english_hits > spanish_hits:
            return "en"
        if spanish_hits > english_hits:
            return "es"
        return "en" if english_hits else "unknown"

    def _compute_boilerplate_ratio(self, html: str) -> float:
        """Calcula la fraccion de texto asociada a boilerplate.

        Args:
            html: HTML a evaluar.

        Returns:
            float: Ratio en el rango `[0.0, 1.0]`.
        """
        soup = BeautifulSoup(html, "lxml")
        total_text = soup.get_text(" ", strip=True)
        total_words = len(total_text.split())
        if total_words <= 0:
            return 1.0

        boilerplate_words = 0
        for selector in ("nav", "header", "footer", "aside", "form", "menu"):
            for node in soup.find_all(selector):
                boilerplate_words += len(node.get_text(" ", strip=True).split())

        return min(1.0, float(boilerplate_words) / float(total_words))

    def _compute_link_density(self, html: str) -> float:
        """Calcula la densidad de enlaces del HTML.

        Args:
            html: HTML a evaluar.

        Returns:
            float: Ratio de enlaces sobre texto total.
        """
        soup = BeautifulSoup(html, "lxml")
        total_text = soup.get_text(" ", strip=True)
        total_words = len(total_text.split())
        if total_words <= 0:
            return 1.0

        anchor_words = 0
        for anchor in soup.find_all("a"):
            anchor_words += len(anchor.get_text(" ", strip=True).split())

        return min(1.0, float(anchor_words) / float(total_words))

    def _apply_embedding_filter(
        self,
        *,
        query: str,
        documents: list[dict[str, object]],
        max_results: int,
    ) -> list[dict[str, object]]:
        """Filtra documentos usando embeddings multilingues y dos umbrales.

        Args:
            query: Consulta del usuario.
            documents: Documentos candidatos ya descargados.
            max_results: Maximo de documentos a devolver.

        Returns:
            list[dict[str, object]]: Documentos ordenados por relevancia.
        """
        if not documents:
            return []

        model = self._get_embedding_model()
        if model is None:
            logger.warning("No se pudo cargar el modelo de embeddings; se omite el filtro semantico")
            return []

        doc_payload_by_id: dict[str, dict[str, object]] = {}
        doc_texts: list[str] = []
        doc_ids: list[str] = []

        for item in documents:
            doc_id = str(item.get("doc_id") or "").strip()
            if not doc_id:
                continue
            combined_text = "\n".join(
                [
                    str(item.get("title") or ""),
                    str(item.get("summary") or ""),
                    str(item.get("content_text") or ""),
                ]
            ).strip()
            if not combined_text:
                continue
            doc_ids.append(doc_id)
            doc_texts.append(combined_text)
            doc_payload_by_id[doc_id] = dict(item)

        if not doc_texts:
            return []

        doc_embeddings = self._encode_texts(model, doc_texts)
        if doc_embeddings.size == 0:
            return []

        cuba_embedding = self._encode_texts(model, ["Cuba"])
        if cuba_embedding.size == 0:
            logger.warning("No se pudo generar embedding de referencia para Cuba")
            return []

        query_embedding = self._encode_texts(model, [query])
        if query_embedding.size == 0:
            logger.warning("No se pudo generar embedding para la query")
            return []

        cuba_scores = doc_embeddings @ cuba_embedding[0]
        query_scores = doc_embeddings @ query_embedding[0]

        scored_documents: list[dict[str, object]] = []
        for idx, doc_id in enumerate(doc_ids):
            payload = dict(doc_payload_by_id.get(doc_id, {}))
            if not payload:
                continue
            cuba_score = float(cuba_scores[idx])
            query_score = float(query_scores[idx])
            final_score = query_score
            payload["cuba_score"] = round(cuba_score, 4)
            payload["query_score"] = round(query_score, 4)
            payload["topic_relevance_percent"] = round(query_score * 100.0, 2)
            payload["semantic_score"] = round(final_score, 4)
            payload["score"] = round(final_score, 4)
            scored_documents.append(payload)

        if not scored_documents:
            return []

        after_cuba = [item for item in scored_documents if float(item.get("cuba_score", 0.0)) >= self.cuba_min_score]
        after_query = [item for item in after_cuba if float(item.get("query_score", 0.0)) >= self.query_min_score]
        after_query.sort(key=lambda item: float(item.get("query_score", 0.0)), reverse=True)
        return after_query[: max(int(max_results), 1)]

    def _build_ddg_query(self, query: str) -> str:
        """Construye la consulta para DuckDuckGo con contexto turistico sobre Cuba.

        Args:
            query: Consulta original del usuario.

        Returns:
            str: Consulta enriquecida.
        """
        normalized = " ".join(str(query or "").split())
        if not normalized:
            return "Cuba turismo"
        return f"Cuba turismo {normalized}"

    def _encode_texts(self, model: SentenceTransformer, texts: list[str]) -> np.ndarray:
        """Convierte una lista de textos en embeddings normalizados.

        Args:
            model: Modelo de Sentence Transformers.
            texts: Lista de textos a codificar.

        Returns:
            np.ndarray: Matriz `(n, d)` de embeddings normalizados.
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return np.ascontiguousarray(array)

    def _get_embedding_model(self) -> SentenceTransformer | None:
        """Carga el modelo semantico de forma diferida.

        Returns:
            SentenceTransformer | None: Modelo listo o `None` si no se pudo cargar.
        """
        if self._embedding_model is not None:
            return self._embedding_model
        if self._embedding_model_error is not None:
            return None
        try:
            self._embedding_model = SentenceTransformer(self.EMBEDDING_MODEL_NAME)
            return self._embedding_model
        except Exception as exc:
            self._embedding_model_error = f"{type(exc).__name__}: {exc}"
            logger.warning("No se pudo cargar el modelo '%s': %s", self.EMBEDDING_MODEL_NAME, exc)
            return None

    def _extract_tokens(self, text: object, language: str | None = None) -> list[str]:
        """Extrae tokens normalizados usando el preprocesador del proyecto.

        Args:
            text: Texto de entrada.
            language: Idioma preferido para aplicar stemming y stopwords.

        Returns:
            list[str]: Lista de tokens limpios, sin stopwords y con stemming.
        """
        cleaned = self._cleaner.clean(text)
        if not cleaned:
            return []

        normalized_language = (language or "").strip().lower()
        if normalized_language in {"es", "spa", "spanish", "espanol", "español"}:
            tokenizer = self._spanish_tokenizer
            stemmer = self._spanish_stemmer
        else:
            tokenizer = self._english_tokenizer
            stemmer = self._english_stemmer

        tokens = tokenizer.tokenize(cleaned)
        tokens = tokenizer.remove_stopwords(tokens)
        return stemmer.stem_tokens(tokens)

    def _count_words(self, text: object) -> int:
        """Cuenta palabras usando la segmentacion del preprocesador.

        Args:
            text: Texto de entrada.

        Returns:
            int: Numero de palabras validas.
        """
        cleaned = self._cleaner.clean(text)
        if not cleaned:
            return 0
        return len(self._english_tokenizer.tokenize(cleaned))
