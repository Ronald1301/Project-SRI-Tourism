from __future__ import annotations

from pathlib import Path

from .config import CrawlerConfig, DEFAULT_EXCLUDE_URL_PATTERNS

# Edit these defaults to control the crawler when running:
#   python3 main.py crawl

SEED_URLS = [
    "https://es.wikivoyage.org/wiki/París",
    # "https://es.wikivoyage.org/wiki/Barcelona",
    # "https://es.wikivoyage.org/wiki/Roma",
    # "https://es.wikivoyage.org/wiki/Londres",
    # "https://es.wikivoyage.org/wiki/Nueva_York",
    # "https://es.wikivoyage.org/wiki/Madrid",
    # "https://es.wikivoyage.org/wiki/La_Habana",
    # "https://es.wikivoyage.org/wiki/México",
    # "https://es.wikivoyage.org/wiki/Buenos_Aires",
    # "https://es.wikivoyage.org/wiki/Tokio",
]

ALLOWED_DOMAINS = {
    "es.wikivoyage.org",
    "en.wikivoyage.org",
}

MAX_DEPTH = 2
MAX_PAGES = 100
REQUEST_DELAY = 1.0
TIMEOUT = 10.0
USER_AGENT = "SRI-Tourism-Crawler/1.0 (+academic-project)"
OUTPUT_DIR = Path("data/raw/crawl")
OBEY_ROBOTS = False
SAVE_HTML = True

INCLUDE_URL_PATTERNS: list[str] = []
EXCLUDE_URL_PATTERNS: list[str] = DEFAULT_EXCLUDE_URL_PATTERNS.copy()


def build_default_config() -> CrawlerConfig:
    return CrawlerConfig.from_iterables(
        seed_urls=SEED_URLS,
        allowed_domains=ALLOWED_DOMAINS,
        max_depth=MAX_DEPTH,
        max_pages=MAX_PAGES,
        request_delay=REQUEST_DELAY,
        timeout=TIMEOUT,
        user_agent=USER_AGENT,
        output_dir=OUTPUT_DIR,
        obey_robots=OBEY_ROBOTS,
        save_html=SAVE_HTML,
        include_url_patterns=INCLUDE_URL_PATTERNS,
        exclude_url_patterns=EXCLUDE_URL_PATTERNS,
    )
