
import requests
from src.frontend.api.mock_client import mock_search

USE_MOCK = False

BASE_URL = "http://localhost:8000"


def search(query, mode, top_k=5, page=1):

    if USE_MOCK:
        return mock_search(query, mode, top_k, page)

    try:
        response = requests.post(
            f"{BASE_URL}/search",
            json={
                "query": query,
                "search_mode": "lsi",
                "top_k": top_k,
            },
            timeout=10
        )
        response.raise_for_status()
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

    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
