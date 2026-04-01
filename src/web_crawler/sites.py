from __future__ import annotations

from pathlib import Path

from .config import CrawlerConfig, DEFAULT_EXCLUDE_URL_PATTERNS


class BaseSite:
    name = "base"
    seed_urls: list[str] = []
    allowed_domains: set[str] = set()
    include_url_patterns: list[str] = []
    exclude_url_patterns: list[str] = []

    max_depth = 2
    max_pages = 2000
    request_delay = 1.0
    timeout = 10.0
    user_agent = "SRI-Tourism-Crawler/1.0 (+academic-project)"
    output_dir = Path("data/raw/crawl")
    obey_robots = False
    save_html = False

    @classmethod
    def build_config(cls) -> CrawlerConfig:
        exclude_patterns = DEFAULT_EXCLUDE_URL_PATTERNS.copy()
        if cls.exclude_url_patterns:
            exclude_patterns.extend(cls.exclude_url_patterns)

        return CrawlerConfig.from_iterables(
            seed_urls=cls.seed_urls,
            allowed_domains=cls.allowed_domains,
            max_depth=cls.max_depth,
            max_pages=cls.max_pages,
            request_delay=cls.request_delay,
            timeout=cls.timeout,
            user_agent=cls.user_agent,
            output_dir=cls.output_dir,
            obey_robots=cls.obey_robots,
            save_html=cls.save_html,
            include_url_patterns=cls.include_url_patterns,
            exclude_url_patterns=exclude_patterns,
        )


class WikivoyageSite(BaseSite):
    name = "wikivoyage"
    seed_urls = [
        "https://es.wikivoyage.org/wiki/La_Habana",
    ]
    allowed_domains = {"es.wikivoyage.org",
    "en.wikivoyage.org",}
    include_url_patterns = [r"/wiki/"]
    exclude_url_patterns = [
        r"/wiki/Special:",
        r"/wiki/Template:",
        r"/wiki/File:",
        r"/wiki/Category:",
        r"/wiki/Categor[ií]a:",
    ]

class VisitarCubaSite(BaseSite):
    name = "visitarcuba"
    seed_urls = [
        "https://www.visitarcuba.org/"
    ]
    allowed_domains = {
        "visitarcuba.org"
    }
    include_url_patterns = []
    exclude_url_patterns = [
        # ========== DIRECTORIOS INTERNOS ==========
    r"/archivos/",
    r"/Images/",
    r"/wp-admin/",
    r"/wp-includes/",
    r"/wp-content/",
    
    # ========== PÁGINAS SIN CONTENO TURÍSTICO ==========
    r"/search\.php",
    r"/contacto\.php",
    r"/precios-especiales-para-agencias-y-tour-operadores",
    r"/favicon\.ico",
    r"/robots\.txt",
    
    # ========== SECCIONES NO RELEVANTES ==========
    r"/tag/",
    r"/category/",
    r"/author/",
    r"/feed",
    r"/rss",
    r"/comments",
    r"\?s=",           # Búsquedas
    r"\?page=",        # Paginación innecesaria
    r"\?p=",           # Posts por ID
    
    # ========== PUBLICIDAD Y B2B ==========
    r"precios-especiales-para-agencias",
    r"/anuncios/",
    r"/adsense",
    
    # ========== REDES SOCIALES Y EXTERNOS ==========
    r"facebook\.com",
    r"twitter\.com",
    r"instagram\.com",
    r"youtube\.com",
    r"whatsapp",
    r"chatra\.io",
    
    # ========== VERSIONES EN OTROS IDIOMAS ==========
    # (si solo quieres español)
    r"tripcuba\.org",      # inglés
    r"cubavoyage\.org",    # francés
    r"viaggiarecuba\.com", # italiano
    r"turismoemcuba\.com", # portugués
    r"visitarcuba\.ru",    # ruso
    
    # ========== PARÁMETROS DE SEGUIMIENTO ==========
    r"\?utm_",
    r"\&utm_",
    r"\?fbclid=",
    r"\?gclid=",
    r"\?ref=",
    
    # ========== CONTENIDO DUPLICADO / PLANTILLAS ==========
    r"/layouts/",
    r"/Templates/",
    r"/Connections/",
    r"/admin",
    
    # ========== JAVASCRIPT Y CALLBACKS ==========
    r"javascript:",
    r"#",
    r"void\(",
]
SITE_REGISTRY = {
    WikivoyageSite.name: WikivoyageSite,
    VisitarCubaSite.name : VisitarCubaSite
}

DEFAULT_SITE_NAME = VisitarCubaSite.name


def get_site_class(name: str) -> type[BaseSite] | None:
    if not name:
        return None
    return SITE_REGISTRY.get(name.strip().lower())


def build_default_config(site_name: str | None = None) -> CrawlerConfig:
    site_name = site_name or DEFAULT_SITE_NAME
    site_cls = get_site_class(site_name)
    if site_cls is None:
        raise ValueError(f"Unknown site: {site_name}")
    return site_cls.build_config()
