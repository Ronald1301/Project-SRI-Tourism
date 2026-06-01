import logging
import sys
from contextlib import asynccontextmanager
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

from src.api.bootstrap import ensure_api_artifacts
from src.api.models import (
    DocumentResult,
    FeedbackRequest,
    FeedbackResponse,
    ImplicitFeedbackRequest,
    SearchRequest,
    SearchResponse,
)
from src.api.service.rag_service import get_rag_service

logger = logging.getLogger("src.api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prepara los artefactos de recuperacion antes de levantar la API.

    Args:
        app: Instancia de FastAPI sobre la que se administra el ciclo de vida.

    Yields:
        None: Permite que la aplicacion atienda requests una vez lista.
    """
    logger.info("Preparando artefactos de recuperacion para la API...")
    try:
        summary = ensure_api_artifacts()
        get_rag_service()
        app.state.retrieval_bootstrap = summary
        logger.info(
            "Artefactos listos | vector_db_built=%s | lsi_built=%s",
            summary.get("vector_db_built"),
            summary.get("lsi", {}).get("built"),
        )
        yield
    except Exception as exc:
        logger.exception("No fue posible preparar los artefactos de recuperacion: %s", exc)
        raise
    finally:
        logger.info("Cerrando ciclo de vida de la API.")


app = FastAPI(title="SRI Tourism API", version="1.0.0", lifespan=lifespan)

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
    logger.info("POST /search | query=\"%s\" | top_k=%d", request.query, request.top_k)
    service = get_rag_service()
    documents, answer, expansion = service.search(
        query=request.query,
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
    return SearchResponse(
        results=results,
        answer=answer,
        total=len(results),
        expansion=expansion,
        events=rag_result.events,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest):
    service = get_rag_service()
    service.add_explicit_feedback(
        query=request.query,
        doc_id=request.doc_id,
        relevance=1 if int(request.relevance) > 0 else 0,
        expanded_query=request.expanded_query,
    )
    logger.info(
        "Feedback explicito | query=\"%s\" | doc_id=%s | relevance=%s",
        request.query,
        request.doc_id,
        request.relevance,
    )
    return FeedbackResponse(status="ok", counted=True)


@app.post("/feedback/implicit", response_model=FeedbackResponse)
def implicit_feedback(request: ImplicitFeedbackRequest):
    service = get_rag_service()
    _, counted = service.add_implicit_feedback(
        query=request.query,
        doc_id=request.doc_id,
        event=request.event,
    )
    logger.info(
        "Feedback implicito | query=\"%s\" | doc_id=%s | event=%s | counted=%s",
        request.query,
        request.doc_id,
        request.event,
        counted,
    )
    return FeedbackResponse(status="ok", counted=counted)
