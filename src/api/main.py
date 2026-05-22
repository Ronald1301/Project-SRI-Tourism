import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import SearchRequest, SearchResponse, DocumentResult
from src.api.service.rag_service import get_rag_service


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
    service = get_rag_service()
    documents, answer = service.search(
        query=request.query,
        search_mode=request.search_mode,
        top_k=request.top_k,
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
        )
        for doc in documents
    ]

    return SearchResponse(results=results, answer=answer, total=len(results))