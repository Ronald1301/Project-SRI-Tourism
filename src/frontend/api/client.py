
import requests
from src.frontend.api.mock_client import mock_search

USE_MOCK = False

BASE_URL = "http://localhost:8000"


def _friendly_request_error(response_text: str) -> str:
    lowered = response_text.lower()
    if "documents.jsonl" in lowered:
        return (
            "La busqueda no pudo comenzar porque faltan los documentos base del proyecto. "
            "Primero genera los datos con el crawler o con el pipeline."
        )
    return (
        "El servidor recibio la consulta, pero no pudo generar una respuesta valida. "
        "Revisa que la API y los datos del proyecto esten preparados."
    )


def search(query, mode, top_k=5, page=1):

    if USE_MOCK:
        return mock_search(query, mode, top_k, page)

    try:
        response = requests.post(
            f"{BASE_URL}/search",
            json={
                "query": query,
                "search_mode": mode,
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
            "total": payload.get("total", len(normalized_results)),
            "has_more": False,
        }

    except requests.exceptions.Timeout:
        return {"error": "La busqueda tardo demasiado en responder."}
    except requests.exceptions.ConnectionError:
        return {"error": "No se pudo conectar con la API local del proyecto."}
    except requests.exceptions.RequestException:
        return {"error": "No se pudo completar la comunicacion con el servidor de busqueda."}
