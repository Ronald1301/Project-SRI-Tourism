import logging
import sys
import asyncio
import json
import threading
from contextlib import asynccontextmanager
from queue import Queue
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
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from src.api.bootstrap import ensure_api_artifacts
from src.api.models import (
    DocumentResult,
    FeedbackRequest,
    FeedbackResponse,
    ImplicitFeedbackRequest,
    ProcessingEvent,
    QueryExpansionInfo,
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


def _build_search_response(
    documents,
    answer,
    expansion,
    domain,
    events,
) -> SearchResponse:
    expansion_model = None
    if expansion:
        expansion_model = QueryExpansionInfo.model_validate(expansion)

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

    return SearchResponse(
        results=results,
        answer=answer,
        total=len(results),
        expansion=expansion_model,
        domain=domain,
        events=[ProcessingEvent.model_validate(item) for item in events],
    )


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    logger.info("POST /search | query=\"%s\" | top_k=%d", request.query, request.top_k)
    service = get_rag_service()
    documents, answer, expansion, domain, events = service.search(
        query=request.query,
        top_k=request.top_k,
        include_explanations=request.explanations,
        generate_answer=request.generate_answer,
    )

    if domain.get("status") == "OUT_OF_DOMAIN":
        logger.info("Consulta fuera de dominio detectada antes del retrieval")
        response = _build_search_response([], None, None, domain, events)
        return response

    response = _build_search_response(documents, answer, expansion, domain, events)
    logger.info("Respuesta enviada | %d resultados | answer_len=%d", len(response.results), len(answer or ""))
    return response


@app.get("/query-stream")
async def query_stream(
    q: str,
    top_k: int = 5,
    explanations: bool = False,
    generate_answer: bool = True,
):
    logger.info("GET /query-stream | query=\"%s\" | top_k=%d", q, top_k)
    service = get_rag_service()
    event_queue: Queue[dict[str, object]] = Queue()
    sentinel = object()

    def publish(event: dict[str, object]) -> None:
        event_queue.put(event)

    def worker() -> None:
        try:
            documents, answer, expansion, domain, events = service.search(
                query=q,
                top_k=top_k,
                include_explanations=explanations,
                generate_answer=generate_answer,
                event_sink=publish,
            )
            response = _build_search_response(documents, answer, expansion, domain, events)
            event_queue.put(
                {
                    "type": "result",
                    "data": jsonable_encoder(response),
                }
            )
        except Exception as exc:  # pragma: no cover - depende del entorno/IO
            logger.exception("query-stream fallo: %s", exc)
            event_queue.put(
                {
                    "type": "error",
                    "message": str(exc),
                }
            )
        finally:
            event_queue.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()

    async def event_generator():
        stage_alias = {
            "checking_domain": "validating_domain",
            "searching_local": "local_search",
            "searching_web": "web_search",
            "generating_answer": "generating_answer",
            "request": "validating_domain",
            "retrieval": "local_search",
            "analysis": "validating_domain",
            "web_search": "web_search",
            "generation": "generating_answer",
            "done": "done",
            "out_of_domain": "out_of_domain",
        }
        while True:
            item = await asyncio.to_thread(event_queue.get)
            if item is sentinel:
                break

            if item.get("type") == "result":
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                break

            if item.get("type") == "error":
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                break

            stage = str(item.get("stage") or "")
            payload = {
                "type": "status",
                "step": stage_alias.get(stage, stage or "processing"),
                "message": item.get("message") or "",
                "progress": item.get("progress"),
                "data": item.get("data"),
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


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
