import random
import time

def mock_search(query, mode, top_k=5, page=1):
    time.sleep(0.8)  # simula latencia

    docs = []
    for i in range(top_k):
        docs.append({
            "title": f"Documento {i + 1 + (page-1)*top_k}",
            "snippet": f"Este documento habla sobre {query} y su impacto en sistemas modernos.",
            "score": random.uniform(0.5, 1.0),
            "source": "mock_source"
        })

    return {
        "results": docs,
        "answer": f"Respuesta generada (RAG) para: '{query}'. Este es un resumen inteligente basado en los documentos recuperados.",
        "has_more": page < 3
    }