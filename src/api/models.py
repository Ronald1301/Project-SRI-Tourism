from typing import Literal

from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    search_mode: str = "lsi"
    top_k: int = 5
    explanations: bool = False


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


class SearchResponse(BaseModel):
    results: list[DocumentResult]
    answer: str | None = None
    total: int
