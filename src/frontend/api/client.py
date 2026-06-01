import os

import requests
from src.frontend.api.mock_client import mock_search

USE_MOCK = False

BASE_URL = os.getenv("SRI_API_BASE_URL", "http://localhost:8000").rstrip("/")


def _friendly_request_error(response_text: str) -> str:
    lowered = response_text.lower()
    if "documents.jsonl" in lowered:
        return (
            "La busqueda no pudo comenzar porque faltan los documentos base del proyecto. "
            "Primero genera los datos con el crawler o con el pipeline."
        )
    if "tfidf_matrix" in lowered or "lsi_model.pkl" in lowered or "data/index" in lowered:
        return (
            "La busqueda no pudo comenzar porque faltan los indices del recuperador. "
            "Primero construye el TF-IDF y entrena el modelo LSI desde el pipeline del proyecto."
        )
    return (
        "El servidor recibio la consulta, pero no pudo generar una respuesta valida. "
        "Revisa que la API y los datos del proyecto esten preparados."
    )


def search(query, top_k=5, page=1):

    if USE_MOCK:
        return mock_search(query, top_k, page)

    try:
        response = requests.post(
            f"{BASE_URL}/search",
            json={
                "query": query,
                "top_k": top_k,
                "explanations": True,
            },
            timeout=10
        )
        if response.status_code >= 400:
            return {"error": _friendly_request_error(response.text)}
        payload = response.json()
        normalized_results = [
            {
                "doc_id": item.get("doc_id"),
                "title": item.get("title"),
                "url": item.get("url"),
                "score": item.get("score", 0.0),
                "snippet": item.get("summary") or "",
                "content_text": item.get("content_text"),
                "rating": item.get("rating"),
                "location": item.get("location"),
                "explanation": item.get("explanation"),
                "source": "api",
                "content_type": "document",
            }
            for item in payload.get("results", [])
        ]
        return {
            "results": normalized_results,
            "answer": payload.get("answer"),
            "expansion": payload.get("expansion"),
            "total": payload.get("total", len(normalized_results)),
            "has_more": False,
        }

    except requests.exceptions.Timeout:
        return {"error": "La busqueda tardo demasiado en responder."}
    except requests.exceptions.ConnectionError:
        return {"error": "No se pudo conectar con la API local del proyecto."}
    except requests.exceptions.RequestException:
        return {"error": "No se pudo completar la comunicacion con el servidor de busqueda."}


def send_explicit_feedback(query, doc_id, relevance, *, expanded_query=None):
    if USE_MOCK:
        return {"status": "ok", "counted": True}

    try:
        response = requests.post(
            f"{BASE_URL}/feedback",
            json={
                "query": query,
                "doc_id": doc_id,
                "relevance": int(relevance),
                "expanded_query": expanded_query,
            },
            timeout=5,
        )
        if response.status_code >= 400:
            return {"error": _friendly_request_error(response.text)}
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "No se pudo guardar la valoracion porque la API local no esta disponible."}
    except requests.exceptions.RequestException:
        return {"error": "No se pudo guardar la valoracion del resultado."}


def send_implicit_feedback(query, doc_id, event):
    if USE_MOCK:
        return {"status": "ok", "counted": True}

    try:
        response = requests.post(
            f"{BASE_URL}/feedback/implicit",
            json={
                "query": query,
                "doc_id": doc_id,
                "event": event,
            },
            timeout=5,
        )
        if response.status_code >= 400:
            return {"error": _friendly_request_error(response.text)}
        return response.json()
    except requests.exceptions.RequestException:
        return {"error": "No se pudo registrar la interaccion con el resultado."}
