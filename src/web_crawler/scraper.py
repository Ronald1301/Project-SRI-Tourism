"""Extraccion estructurada de HTML y metadatos para el crawler."""

from __future__ import annotations

import json
from datetime import UTC,datetime
from hashlib import sha1
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from .policies import CrawlPolicies

def safe_text(value : object) -> str | None:
    """Convierte un valor arbitrario a texto limpio o `None`.

    Args:
        value: Valor de entrada a normalizar.

    Returns:
        str | None: Texto sin espacios extremos o `None` si queda vacio.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None

def flatten_jsonld(node : object) -> list[dict]:
    """Aplana una estructura JSON-LD anidada.

    Args:
        node: Nodo JSON-LD arbitrario.

    Returns:
        list[dict]: Lista plana de diccionarios JSON-LD.
    """
    if node is None:
        return []
    if isinstance(node,list):
        flattened : list[dict] = []
        for item in node:
            flattened.extend(flatten_jsonld(item))
        return flattened
    if isinstance(node,dict):
        if "@graph" in node:
            return flatten_jsonld(node["@graph"])
        return [node]
    return []

def load_jsonld(soup : BeautifulSoup) -> list[dict]:
    """Extrae todos los bloques JSON-LD de un documento HTML.

    Args:
        soup: Objeto BeautifulSoup del HTML.

    Returns:
        list[dict]: Registros JSON-LD parseados y aplanados.
    """
    output : list[dict] = []
    for script in soup.select('script[type="application/ld+json"]'):
        raw = (script.string or script.get_text() or "").strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        output.extend(flatten_jsonld(parsed))
    return output 

def first_non_null(*values : object) -> str | None:
    """Devuelve el primer valor textual util de una secuencia.

    Args:
        *values: Valores candidatos.

    Returns:
        str | None: Primer texto no vacio o `None`.
    """
    for value in values:
        candidate = safe_text(value)
        if candidate:
            return candidate
    return None

def extract_jsonld_fields(records : list[dict]) -> dict[str , object]:
    """Resume campos relevantes desde un conjunto de registros JSON-LD.

    Args:
        records: Lista de objetos JSON-LD.

    Returns:
        dict[str, object]: Metadatos consolidados del documento.
    """
    entity_name = None
    review_author = None
    review_date = None
    rating = None
    location = None
    entity_type = None
    tags: list[str] = []

    for record in records:
        r_type = record.get("@type")
        if isinstance(r_type,list):
            r_type = ",".join(str(x) for x in r_type)
        if not entity_type and r_type:
            entity_type = str(r_type)
        
        entity_name = entity_name or safe_text(record.get("name"))

        if isinstance(record.get("author"),dict):
            review_author = review_author or safe_text(record["author"].get("name"))
        review_author = review_author or safe_text(record.get("author"))

        review_date = review_date or safe_text(record.get("datePublished"))

        if isinstance(record.get("aggregateRating"),dict):
            rating = rating or safe_text(record['aggregateRating'].get("ratingValue"))
        if isinstance(record.get("reviewRating"),dict):
            rating = rating or safe_text(record['reviewRating'].get("ratingValue"))
        rating = rating or safe_text(record.get("ratingValue"))

        address = record.get("address")
        if isinstance(address,dict):
            location = location or first_non_null(
                address.get("addressLocality"),
                address.get("addressRegion"),
                address.get("addressCountry"),
            )
        location = location or safe_text(record.get("location"))

        if isinstance(record.get("keywords"),str):
            tags.extend([token.strip() for token in record["keywords"].split(",") if token.strip()])
    
    unique_tags = sorted(set(tags))
    return {
        "entity_name" : entity_name,
        "review_author" : review_author,
        "review_date" : review_date,
        "rating" : rating,
        "location" : location,
        "content_type" : entity_type,
        "tags" : unique_tags
    }

def extract_links(html : str , base_url : str, policies : CrawlPolicies) -> set[str]:
    """Extrae enlaces permitidos desde un HTML.

    Args:
        html: Contenido HTML.
        base_url: URL base para resolver enlaces relativos.
        policies: Politicas de filtrado y normalizacion.

    Returns:
        set[str]: Conjunto de URLs normalizadas y permitidas.
    """
    soup = BeautifulSoup(html,"lxml")
    links : set[str] = set()
    for anchor in soup.find_all("a",href=True):
        normalized = policies.normalize_url(base_url,anchor["href"])
        if not normalized:
            continue
        if policies.is_allowed(normalized):
            links.add(normalized)
    return links

def extract_document(html : str, url : str) -> dict[str,object]:
    """Construye un documento estructurado a partir de HTML.

    Args:
        html: Contenido HTML descargado.
        url: URL original del recurso.

    Returns:
        dict[str, object]: Documento con titulo, resumen, texto, idioma y metadatos.
    """
    soup = BeautifulSoup(html,"lxml")
    jsonld = load_jsonld(soup)
    jsonld_fields = extract_jsonld_fields(jsonld)

    title = first_non_null(
        soup.find("meta", property="og:title") and soup.find("meta",property="og:title").get("content"),
        soup.title and soup.title.get_text(strip=True),
    )
    summary = first_non_null(
        soup.find("meta",attrs={"name":"description"}) and soup.find("meta",attrs={"name":"description"}).get("content"),
        soup.find("meta",property="og:description") and soup.find("meta",property="og:description").get("content"),
    )

    text_blocks = []
    article = soup.find("article")
    if article:
        text_blocks.extend(node.get_text(" ",strip=True) for node in article.find_all(["p","li"]))
    if not text_blocks:
        text_blocks.extend(node.get_text(" ",strip=True) for node in soup.find_all("p"))
    text = "\n".join(block for block in text_blocks if block)

    images : list[str] = []
    for image in soup.find_all("img",src=True):
        src = image["src"].strip()
        if src and src not in images:
            images.append(src)
        if len(images) == 10:
            break
    
    lang = None
    if soup.html and soup.html.has_attr("lang"):
        lang = safe_text(soup.html["lang"])
    
    content_type = jsonld_fields.get("content_type") or "web_page"

    record = {
        "doc_id" : sha1(url.encode("utf-8")).hexdigest(),
        "url" : url,
        "domain" : urlparse(url).netloc.lower(),
        "title" : title,
        "summary" : summary,
        "content_text" : text,
        "word_count" : len(text.split()),
        "language" : lang,
        "content_type" : content_type,
        "entity_name" : jsonld_fields.get("entity_name") or title,
        "review_author" : jsonld_fields.get("review_author"),
        "review_date" : jsonld_fields.get("review_date"),
        "rating" : jsonld_fields.get("rating"),
        "location" : jsonld_fields.get("location"),
        "tags" : jsonld_fields.get("tags") or [],
        "images" : images,
        "scraped_at" : datetime.now(UTC).isoformat(),
    }
    return record
