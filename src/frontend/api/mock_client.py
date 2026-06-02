import random
import time

def mock_search(query, top_k=5, generate_answer=True):
    time.sleep(0.8)  # simula latencia

    docs = []
    for i in range(top_k):
        docs.append({
            "title": f"Documento {i + 1}",
            "snippet": f"Este documento habla sobre {query} y su impacto en sistemas modernos.",
            "score": random.uniform(0.5, 1.0),
            "source": "mock_source"
        })

    answer = (
        f"Respuesta local para: '{query}'. Este resumen se construyo solo con los documentos recuperados."
        if not generate_answer
        else f"Respuesta generada (RAG) para: '{query}'. Este es un resumen inteligente basado en los documentos recuperados."
    )

    return {
        "results": docs,
        "answer": answer,
        "domain": {
            "query": query,
            "status": "IN_DOMAIN",
            "fast_decision": "IN_DOMAIN",
            "used_llm": False,
            "llm_result": None,
            "message": None,
            "model": None,
            "features": {},
        },
        "events": [
            {"event_type": "processing", "stage": "checking_domain", "message": "Analizando consulta..."},
            {"event_type": "processing", "stage": "searching_local", "message": "Buscando en base de datos local..."},
            {"event_type": "processing", "stage": "generation", "message": "Generacion local." if not generate_answer else "Generando respuesta..."},
            {"event_type": "processing", "stage": "done", "message": "Busqueda completada."},
        ],
        "has_more": False
    }
