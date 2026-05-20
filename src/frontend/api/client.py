
import requests
from src.frontend.api.mock_client import mock_search

USE_MOCK = True

BASE_URL = "http://localhost:8000"

def search(query, mode, top_k=5, page=1):

    if USE_MOCK:
        return mock_search(query, mode, top_k, page)

    try:
        response = requests.post(
            f"{BASE_URL}/search",
            json={
                "query": query,
                "mode": mode,
                "top_k": top_k,
                "page": page
            },
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
