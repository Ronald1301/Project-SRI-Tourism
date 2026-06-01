"""Presets de sitios para el modulo de crawling."""

from __future__ import annotations

from pathlib import Path

from .config import CrawlerConfig, DEFAULT_EXCLUDE_URL_PATTERNS


class BaseSite:
    """Clase base para describir el preset de un sitio web."""

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
        """Construye un `CrawlerConfig` a partir del preset del sitio.

        Returns:
            CrawlerConfig: Configuracion lista para ejecutar el crawler.
        """
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

class VisitarCubaSite(BaseSite):
    """Preset de crawling para el sitio Visitar Cuba."""

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
    
    # ========== PÁGINAS SIN CONTENIDO TURÍSTICO ==========
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


class CubatravelSite(BaseSite):
    """Preset de crawling para el sitio Cuba Travel."""

    name = "cubatravel"
    seed_urls = [
        "https://www.cuba.travel/",
    ]
    allowed_domains = {
        "www.cuba.travel",
        "cuba.travel",
        "www.cubatravel.cu",
        "cubatravel.cu",
    }
    include_url_patterns = []
    exclude_url_patterns = [
        # Archivos estáticos
        r"\.css$",
        r"\.js$",
        r"\.json$",
        r"\.xml$",
        r"\.png$",
        r"\.jpg$",
        r"\.jpeg$",
        r"\.gif$",
        r"\.svg$",
        r"\.ico$",
        
        # Directorios internos
        r"/DesktopModules/",
        r"/Portals/",
        r"/LinkClick/",
        r"/DependencyHandler/",
        r"/WebResource",
        r"/ScriptResource",
        
        # Parámetros de query
        r"\?",
        
        # Módulos especiales
        r"/BookingEngine/",
        r"/Activity/Search",
        r"/Hotel/Search",
        r"/House/Search",
        
        # Formularios y APIs
        r"/survey",
        r"/api/",
        
        # Otros idiomas
        r"/en/",
        r"/fr/",
        r"/de/",
        r"/ru/",
        
        # Funciones de usuario
        r"/login",
        r"/register",
        r"/profile",
        
        r"facebook\.com",
        r"twitter\.com",
        r"instagram\.com",
        r"youtube\.com",
        r"linkedin\.com",
        r"/tag/",
        r"/author/",
        r"/feed",
        r"/rss",
        r"\?utm_",
        r"\&utm_",
        r"\?fbclid=",
        r"\?gclid=",
        r"javascript:",
        r"void\(",
        r"#",
    ]


class InfoturSite(BaseSite):
    """Preset de crawling para el sitio Infotur."""

    name = "infotur"
    seed_urls = [
        "https://infotur.cu/",
    ]
    allowed_domains = {
        "www.infotur.cu",
        "infotur.cu",
    }
    include_url_patterns = []
    exclude_url_patterns = [
        # ========== ARCHIVOS ESTÁTICOS ==========
        r"\.css",
        r"\.js", 
        r"\.json",
        r"\.xml",
        r"\.png",
        r"\.jpg",
        r"\.jpeg",
        r"\.gif",
        r"\.svg",
        r"\.ico",
        r"\.webp",
        r"\.pdf", 
        r"\.mp4",
        r"\.mp3",
        
        # ========== RECURSOS INTERNOS DE NUXT ==========
        r"/_nuxt/",
        r"/api/",
        r"/favicon",
        r"/logo.png",
        
        # ========== OTROS IDIOMAS (si solo quieres español) ==========
        r"/ru/",
        r"/en/",
        r"/de/",
        r"/fr/",
        
        # ========== PARÁMETROS DE QUERY ==========
        r"\?",
        
        # ========== SECCIONES NO RELEVANTES ==========
        r"/tag/",
        r"/category/",
        r"/author/",
        r"/feed",
        r"/rss",
        
        # ========== REDES SOCIALES ==========
        r"facebook.com",
        r"twitter.com",
        r"instagram.com",
        r"youtube.com",
        r"linkedin.com",
        
        # ========== PARÁMETROS DE SEGUIMIENTO ==========
        r"\?utm_",
        r"\&utm_",
        r"\?fbclid=",
        r"\?gclid=",
        
        # ========== JAVASCRIPT Y CALLBACKS ==========
        r"javascript:",
        r"void\(",
        r"#",
        
        # ========== ADMIN Y BACKEND ==========
        r"/wp-admin/",
        r"/wp-includes/",
        r"/wp-content/",
    ]


SITE_REGISTRY = {
    VisitarCubaSite.name: VisitarCubaSite,
    CubatravelSite.name: CubatravelSite,
    InfoturSite.name: InfoturSite,
}

DEFAULT_SITE_NAME = InfoturSite.name


def get_site_class(name: str) -> type[BaseSite] | None:
    """Resuelve una clase de sitio por nombre.

    Args:
        name: Nombre logico del sitio.

    Returns:
        type[BaseSite] | None: Clase asociada o `None`.
    """
    if not name:
        return None
    return SITE_REGISTRY.get(name.strip().lower())


def build_default_config(site_name: str | None = None) -> CrawlerConfig:
    """Construye la configuracion por defecto de un sitio.

    Args:
        site_name: Nombre del sitio. Si es `None`, usa el sitio por defecto.

    Returns:
        CrawlerConfig: Configuracion del preset elegido.

    Raises:
        ValueError: Si el sitio no existe en el registro.
    """
    site_name = site_name or DEFAULT_SITE_NAME
    site_cls = get_site_class(site_name)
    if site_cls is None:
        raise ValueError(f"Unknown site: {site_name}")
    return site_cls.build_config()
