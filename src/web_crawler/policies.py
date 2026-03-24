from __future__ import annotations

import re
from urllib.parse import urljoin,urlparse,urlunparse
from urllib.robotparser import RobotFileParser
from .config import CrawlerConfig

class CrawlPolicies:
    def __init__(self,config : CrawlerConfig) -> None:
        self.config = config
        self.robots_cache : dict[str,RobotFileParser] = {}
        self.include_patterns = [re.compile(p,flags=re.IGNORECASE) for p in config.include_url_patterns]
        self.exclude_patterns = [re.compile(p,flags=re.IGNORECASE) for p in config.exclude_url_patterns]
    
    @staticmethod
    def normalize_url(base_url : str,href : str) -> str | None:
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
        scheme = urlparse(url).scheme.lower()
        return scheme in self.config.allowed_schemes
    
    def is_allowed_domain(self , url : str) -> bool:
        if not self.config.allowed_domains:
            return True
        
        host = (urlparse(url).hostname or "").lower()
        return any(host == domain or host.endswith(f".{domain}") for domain in self.config.allowed_domains)
    
    def is_allowed_by_patterns(self, url : str) -> bool:
        if self.include_patterns and not any(p.search(url) for p in self.include_patterns):
            return False
        if self.exclude_patterns and any(p.search(url) for p in self.exclude_patterns):
            return False
        return True
    
    def is_allowed(self , url : str) -> bool:
        return self.is_allowed_scheme(url) and self.is_allowed_domain(url) and self.is_allowed_by_patterns(url)
    
    def get_robots_parser(self , url : str) -> RobotFileParser:
        parsed = urlparse(url)
        host_key = f"{parsed.scheme}://{parsed.netloc}".lower()
        parser = self.robots_cache.get(host_key)
        if parser is not None:
            return parser
        
        robots_url = f"{host_key}/robots.txt"
        parser = RobotFileParser()
        parser.set_url(robots_url)
        try:
            parser.read()
        except Exception:
            pass
        self.robots_cache[host_key] = parser
        return parser
    
    def is_allowed_by_robots(self, url : str, user_agent : str) -> bool:
        if not self.config.obey_robots:
            return True
        parser = self.get_robots_parser(url)
        try:
            return parser.can_fetch(user_agent,url)
        except Exception:
            return True