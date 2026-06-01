from typing import Any, Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    explanations: bool = False


class ProcessingEvent(BaseModel):
    event_type: str
    message: str
    stage: str
    timestamp: str
    progress: float | None = None
    data: dict[str, Any] | None = None


class ExplanationComponent(BaseModel):
    name: str
    value: float
    weight: float
    contribution: float
    category: Literal["base", "signal"]


class ExplanationAdjustment(BaseModel):
    name: str
    value: float
    contribution: float


class ExplanationNormalization(BaseModel):
    name: str
    method: str
    fields: list[str]


class ExplanationExactMatches(BaseModel):
    full_query_phrase: bool


class RankingExplanation(BaseModel):
    schema_version: str
    final_score: float
    base_score: float
    signal_total: float
    components: list[ExplanationComponent]
    boosts: list[ExplanationAdjustment]
    penalties: list[ExplanationAdjustment]
    normalizations: list[ExplanationNormalization]
    weights: dict[str, float]
    signals: dict[str, float]
    exact_matches: ExplanationExactMatches


class DocumentResult(BaseModel):
    doc_id: str
    title: str
    url: str | None
    score: float
    summary: str | None
    content_text: str | None
    rating: str | None
    location: str | None
    explanation: RankingExplanation | None = None


class QueryExpansionInfo(BaseModel):
    enabled: bool = True
    original_query: str
    expanded_query: str
    selected_query: str | None = None
    selected_strategy: str = "raw"
    terms: list[str] = Field(default_factory=list)
    term_scores: dict[str, float] = Field(default_factory=dict)
    method: str = "hybrid"
    applied: bool = False


class SearchResponse(BaseModel):
    results: list[DocumentResult]
    answer: str | None = None
    total: int
    expansion: QueryExpansionInfo | None = None
    events: list[ProcessingEvent] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    query: str
    doc_id: str
    relevance: int
    expanded_query: str | None = None


class ImplicitFeedbackRequest(BaseModel):
    query: str
    doc_id: str
    event: str


class FeedbackResponse(BaseModel):
    status: str
    counted: bool = True
