"""Typed API request and response models."""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class UploadResponse(BaseModel):
    dataset_id: str
    filename: str
    record_count: int
    columns: list[str]


class SalesDataResponse(BaseModel):
    dataset_id: str
    filename: str
    record_count: int
    columns: list[str]
    data: list[dict[str, Any]]


class KpiResponse(BaseModel):
    dataset_id: str
    kpis: dict[str, int | float]


class AnalyticsResponse(BaseModel):
    dataset_id: str
    monthly_revenue: list[dict[str, Any]]
    product_performance: list[dict[str, Any]]
    revenue_by_region: list[dict[str, Any]]


class ForecastResponse(BaseModel):
    dataset_id: str
    history: list[dict[str, Any]]
    forecast: list[dict[str, Any]]
    evaluation: dict[str, int | float]


class SegmentsResponse(BaseModel):
    dataset_id: str
    segments: list[dict[str, Any]]
    customers: list[dict[str, Any]]


class AnomaliesResponse(BaseModel):
    dataset_id: str
    anomaly_count: int
    anomalies: list[dict[str, Any]]


class AskRequest(BaseModel):
    dataset_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("dataset_id", "question")
    @classmethod
    def require_nonblank_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("This field must contain non-whitespace text.")
        return stripped


class AskResponse(BaseModel):
    dataset_id: str
    supported: bool
    analysis_type: str | None = None
    answer: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    page_count: int
    chunk_count: int


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    created_at: str


class DocumentAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_id: str | None = Field(default=None, min_length=1, max_length=64)

    @field_validator("question", "document_id")
    @classmethod
    def require_nonblank_text(cls, value: str | None) -> str | None:
        if value is None:
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("This field must contain non-whitespace text.")
        return stripped


class DocumentSource(BaseModel):
    document_id: str
    filename: str
    page_number: int
    similarity: float


class DocumentAskResponse(BaseModel):
    answer: str
    sources: list[DocumentSource]
