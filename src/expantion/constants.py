from __future__ import annotations

import re
from pathlib import Path

DEFAULT_FEEDBACK_PATH = Path("data/feedback/query_feedback.json")
WORD_RE = re.compile(r"[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ]{3,}")

STOPWORDS_ES = {
    "aqui",
    "algo",
    "ante",
    "como",
    "con",
    "cual",
    "cuando",
    "donde",
    "desde",
    "este",
    "esta",
    "estos",
    "estas",
    "para",
    "pero",
    "porque",
    "que",
    "sin",
    "sobre",
    "todo",
    "una",
    "uno",
    "unos",
    "unas",
    "ver",
    "visitar",
    "viaje",
    "viajar",
    "turismo",
    "turistico",
    "turistica",
    "cuba",
    "cubano",
    "cubana",
    "lugar",
    "lugares",
    "mejor",
    "mejores",
}

STOPWORDS_EN = {
    "about",
    "and",
    "any",
    "best",
    "can",
    "cuba",
    "cuban",
    "do",
    "does",
    "for",
    "from",
    "get",
    "good",
    "how",
    "into",
    "near",
    "place",
    "places",
    "see",
    "some",
    "the",
    "there",
    "these",
    "this",
    "those",
    "tourism",
    "tourist",
    "travel",
    "trip",
    "visit",
    "what",
    "when",
    "where",
    "which",
    "with",
}

DOMAIN_SYNONYMS_ES = {
    "playa": ["cayo", "balneario"],
    "playas": ["cayo", "balneario"],
    "hotel": ["alojamiento", "hostal"],
    "hoteles": ["alojamiento", "hostal"],
    "restaurante": ["gastronomia", "comida"],
    "restaurantes": ["gastronomia", "comida"],
    "habana": ["malecon", "vedado"],
    "vieja": ["casco historico", "centro historico"],
    "museo": ["galeria", "patrimonio"],
    "museos": ["galeria", "patrimonio"],
    "excursion": ["tour", "ruta"],
    "excursiones": ["tour", "ruta"],
    "naturaleza": ["parque", "sendero"],
    "familia": ["ninos", "familiar"],
}

DOMAIN_SYNONYMS_EN = {
    "beach": ["playa", "cayo"],
    "beaches": ["playas", "cayo"],
    "hotel": ["hoteles", "lodging"],
    "hotels": ["hotel", "lodging"],
    "restaurant": ["restaurante", "gastronomy"],
    "restaurants": ["restaurantes", "gastronomy"],
    "havana": ["habana", "malecon"],
    "old": ["vieja", "historic center"],
    "museum": ["museo", "gallery"],
    "museums": ["museos", "gallery"],
    "tour": ["excursion", "route"],
    "tours": ["excursiones", "route"],
    "nature": ["naturaleza", "park"],
    "family": ["familia", "children"],
}

IMPLICIT_EVENT_GROUPS = {
    "copy_url": "source_interaction",
    "copiar_url": "source_interaction",
    "open_source": "source_interaction",
    "abrir_fuente": "source_interaction",
    "source_interaction": "source_interaction",
}

IMPLICIT_WEIGHTS = {
    "source_interaction": 0.65,
    "view_result": 0.2,
    "dwell": 0.35,
}


def get_stopwords(language: str = "auto") -> set[str]:
    if language == "english":
        return set(STOPWORDS_EN)
    if language == "spanish":
        return set(STOPWORDS_ES)
    return set(STOPWORDS_ES) | set(STOPWORDS_EN)


def get_domain_synonyms(language: str = "auto") -> dict[str, list[str]]:
    if language == "english":
        return dict(DOMAIN_SYNONYMS_EN)
    if language == "spanish":
        return dict(DOMAIN_SYNONYMS_ES)
    return {**DOMAIN_SYNONYMS_ES, **DOMAIN_SYNONYMS_EN}
