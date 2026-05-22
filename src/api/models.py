from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    search_mode: str = "lsi"
    top_k: int = 5


class DocumentResult(BaseModel):
    doc_id: str
    title: str
    url: str | None
    score: float
    summary: str | None
    content_text: str | None
    rating: str | None
    location: str | None


class SearchResponse(BaseModel):
    results: list[DocumentResult]
    answer: str | None = None
    total: int