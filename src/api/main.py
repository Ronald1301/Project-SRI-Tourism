import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import SearchRequest, SearchResponse, DocumentResult
from src.api.service.rag_service import get_rag_service

logger = logging.getLogger("src.api.main")

app = FastAPI(title="SRI Tourism API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    logger.info("POST /search | query=\"%s\" | mode=%s | top_k=%d", request.query, request.search_mode, request.top_k)
    service = get_rag_service()
    documents, answer = service.search(
        query=request.query,
        search_mode=request.search_mode,
        top_k=request.top_k,
        include_explanations=request.explanations,
    )

    results = [
        DocumentResult(
            doc_id=doc.doc_id,
            title=doc.title,
            url=doc.url,
            score=doc.score,
            summary=doc.summary,
            content_text=doc.content_text,
            rating=doc.metadata.get("rating") if doc.metadata else None,
            location=doc.metadata.get("location") if doc.metadata else None,
            explanation=doc.metadata.get("explanation") if doc.metadata else None,
        )
        for doc in documents
    ]

    logger.info("Respuesta enviada | %d resultados | answer_len=%d", len(results), len(answer or ""))
    return SearchResponse(results=results, answer=answer, total=len(results))
