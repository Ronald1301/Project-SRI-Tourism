"""Politicas de normalizacion y filtrado para URLs del crawler."""

from __future__ import annotations

import logging
import re
from urllib.parse import urljoin,urlparse,urlunparse
from urllib.robotparser import RobotFileParser
from .config import CrawlerConfig

logger = logging.getLogger("src.web_crawler.policies")

class CrawlPolicies:
    """Agrupa reglas de URL, dominio, patrones y robots.txt.

    La clase se usa tanto por el crawler principal como por la busqueda web
    para decidir si una URL debe descargarse.
    """

    def __init__(self,config : CrawlerConfig) -> None:
        """Inicializa las politicas a partir de una configuracion.

        Args:
            config: Configuracion base del crawler.

        Returns:
            None
        """
        self.config = config
        self.robots_cache : dict[str,RobotFileParser] = {}
        self.include_patterns = [re.compile(p,flags=re.IGNORECASE) for p in config.include_url_patterns]
        self.exclude_patterns = [re.compile(p,flags=re.IGNORECASE) for p in config.exclude_url_patterns]
    
    @staticmethod
    def normalize_url(base_url : str,href : str) -> str | None:
        """Normaliza un enlace HTML absoluto o relativo.

        Args:
            base_url: URL base desde la que se resuelve el enlace.
            href: Valor del atributo `href`.

        Returns:
            URL normalizada o `None` si el enlace no es util.
        """
        if not href:
            return None
        href = href.strip()
        if not href or href.startswith(("javascript:","mailto:","tel:")):
            return None
        
        absolute = urljoin(base_url,href)
        parsed = urlparse(absolute)
        if not parsed.netloc:
            return None
        
        cleaned = parsed._replace(fragment="",params="")
        return urlunparse(cleaned)
    
    def is_allowed_scheme(self, url : str) -> bool:
        """Indica si el esquema de la URL esta permitido.

        Args:
            url: URL a evaluar.

        Returns:
            bool: `True` si el esquema es aceptado.
        """
        scheme = urlparse(url).scheme.lower()
        return scheme in self.config.allowed_schemes
    
    def is_allowed_domain(self , url : str) -> bool:
        """Indica si el dominio pertenece al conjunto permitido.

        Args:
            url: URL a evaluar.

        Returns:
            bool: `True` si el dominio esta permitido o no hay restriccion.
        """
        if not self.config.allowed_domains:
            return True
        
        host = (urlparse(url).hostname or "").lower()
        return any(host == domain or host.endswith(f".{domain}") for domain in self.config.allowed_domains)
    
    def is_allowed_by_patterns(self, url : str) -> bool:
        """Aplica patrones de inclusion y exclusion sobre la URL.

        Args:
            url: URL a evaluar.

        Returns:
            bool: `True` si la URL pasa las reglas de patron.
        """
        if self.include_patterns and not any(p.search(url) for p in self.include_patterns):
            return False
        if self.exclude_patterns and any(p.search(url) for p in self.exclude_patterns):
            return False
        return True
    
    def is_allowed(self , url : str) -> bool:
        """Combina esquema, dominio y patrones en una sola validacion.

        Args:
            url: URL a evaluar.

        Returns:
            bool: `True` si la URL puede procesarse.
        """
        return self.is_allowed_scheme(url) and self.is_allowed_domain(url) and self.is_allowed_by_patterns(url)
    
    def get_robots_parser(self , url : str) -> RobotFileParser:
        """Obtiene y cachea el parser de robots.txt para un host.

        Args:
            url: URL que determina el host de robots.txt.

        Returns:
            RobotFileParser: Parser cacheado o recien descargado.
        """
        parsed = urlparse(url)
        host_key = f"{parsed.scheme}://{parsed.netloc}".lower()
        parser = self.robots_cache.get(host_key)
        if parser is not None:
            return parser

        robots_url = f"{host_key}/robots.txt"
        logger.info("Obteniendo robots.txt: %s", robots_url)
        parser = RobotFileParser()
        parser.set_url(robots_url)
        try:
            parser.read()
            logger.info("robots.txt cargado exitosamente: %s", robots_url)
        except Exception as exc:
            logger.warning("Error al obtener/parsear robots.txt %s: %s", robots_url, exc)
        self.robots_cache[host_key] = parser
        return parser
    
    def is_allowed_by_robots(self, url : str, user_agent : str) -> bool: 
        """Verifica robots.txt para una URL dada.

        Args:
            url: URL a evaluar.
            user_agent: User-Agent usado en la consulta.

        Returns:
            bool: `True` si robots.txt permite la descarga.
        """
        if not self.config.obey_robots:
            return True
        parser = self.get_robots_parser(url)
        if parser is None:
            return True
        try:
            allowed = parser.can_fetch(user_agent,url)
            if not allowed:
                logger.info("URL descartada por robots.txt: %s", url)
            return allowed
        except Exception as exc:
            logger.warning("Error al consultar robots.txt para %s: %s", url, exc)
            return True
