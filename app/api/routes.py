"""FastAPI endpoints for sales uploads, analytics, ML, and grounded questions."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from app.ai.analyst import answer_natural_language_question
from app.ai.errors import LLMConfigurationError, LLMServiceError
from app.analytics.kpis import calculate_kpis
from app.analytics.sales_analysis import (
    product_performance,
    revenue_by_month,
    revenue_by_region,
)
from app.api.schemas import (
    AnalyticsResponse,
    AnomaliesResponse,
    AskRequest,
    AskResponse,
    DocumentAskRequest,
    DocumentAskResponse,
    DocumentSummary,
    DocumentUploadResponse,
    ForecastResponse,
    KpiResponse,
    SalesDataResponse,
    SegmentsResponse,
    UploadResponse,
)
from app.data import DataValidationError, load_and_clean_sales_data
from app.database import (
    DatabaseDataset,
    DatabaseUnavailableError,
    SalesRepository,
)
from app.database.database import get_default_session_factory
from app.rag import (
    DocumentError,
    DocumentStoreError,
    answer_document_question,
    delete_document,
    ingest_pdf,
    list_documents,
)
from app.ml.anomaly_detection import detect_revenue_anomalies
from app.ml.errors import MLAnalysisError
from app.ml.forecasting import forecast_monthly_revenue
from app.ml.segmentation import segment_customers

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


sales_repository = SalesRepository(get_default_session_factory())
router = APIRouter(tags=["sales"])
documents_router = APIRouter(prefix="/documents", tags=["documents"])
DATASET_NOT_FOUND = {404: {"description": "Dataset ID does not exist."}}
DATABASE_UNAVAILABLE = {503: {"description": "The configured database is unavailable."}}
ANALYSIS_UNAVAILABLE = {422: {"description": "The dataset does not have enough valid data for this analysis."}}


class _NamedBytesIO(BytesIO):
    def __init__(self, filename: str, content: bytes) -> None:
        super().__init__(content)
        self.name = filename


def _get_dataset(dataset_id: str) -> DatabaseDataset:
    try:
        stored = sales_repository.get_dataset(dataset_id)
    except DatabaseUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    if stored is None:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload the file again.")
    return stored


def _records(data: pd.DataFrame) -> list[dict[str, object]]:
    """Convert pandas values and timestamps to JSON-compatible Python values."""
    return json.loads(data.to_json(orient="records", date_format="iso", force_ascii=False))


def _clean_uploaded_file(filename: str, content: bytes) -> pd.DataFrame:
    """Validate and clean upload bytes using the shared ingestion pipeline."""
    safe_name = Path(filename.replace("\\", "/")).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="The uploaded file must have a filename.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The uploaded file exceeds the 10 MiB size limit.")
    try:
        return load_and_clean_sales_data(_NamedBytesIO(safe_name, content))
    except DataValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def _store_dataset(filename: str, sales: pd.DataFrame) -> str:
    try:
        return sales_repository.save_dataset(filename, sales)
    except DatabaseUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and validate a sales file",
    responses={
        400: {"description": "The upload has no filename."},
        413: {"description": "The upload exceeds the 10 MiB limit."},
        422: {"description": "The file is unsupported or has invalid sales data."},
        **DATABASE_UNAVAILABLE,
    },
)
def upload_sales_file(file: UploadFile = File(...)) -> UploadResponse:
    """Accept a CSV or modern Excel file and persist cleaned records in the database."""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="The uploaded file must have a filename.")
        content = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="The uploaded file exceeds the 10 MiB size limit.")
        sales = _clean_uploaded_file(file.filename, content)
        filename = Path(file.filename.replace("\\", "/")).name
        dataset_id = _store_dataset(filename, sales)
    finally:
        file.file.close()
    return UploadResponse(
        dataset_id=dataset_id,
        filename=filename,
        record_count=len(sales),
        columns=list(sales.columns),
    )


@router.get(
    "/data/{dataset_id}",
    response_model=SalesDataResponse,
    summary="Get cleaned sales records",
    responses={**DATASET_NOT_FOUND, **DATABASE_UNAVAILABLE},
)
def get_sales_data(dataset_id: str) -> SalesDataResponse:
    stored = _get_dataset(dataset_id)
    return SalesDataResponse(
        dataset_id=dataset_id,
        filename=stored.filename,
        record_count=len(stored.sales),
        columns=list(stored.sales.columns),
        data=_records(stored.sales),
    )


@router.delete(
    "/data/{dataset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an uploaded dataset",
    responses={**DATASET_NOT_FOUND, **DATABASE_UNAVAILABLE},
)
def delete_sales_data(dataset_id: str) -> None:
    try:
        deleted = sales_repository.delete_dataset(dataset_id)
    except DatabaseUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    if not deleted:
        raise HTTPException(status_code=404, detail="Dataset not found.")


@router.get(
    "/kpis/{dataset_id}",
    response_model=KpiResponse,
    summary="Calculate sales KPIs",
    responses={**DATASET_NOT_FOUND, **DATABASE_UNAVAILABLE},
)
def get_sales_kpis(dataset_id: str) -> KpiResponse:
    stored = _get_dataset(dataset_id)
    return KpiResponse(dataset_id=dataset_id, kpis=calculate_kpis(stored.sales))


@router.get(
    "/analytics/{dataset_id}",
    response_model=AnalyticsResponse,
    summary="Get grouped sales summaries",
    responses={**DATASET_NOT_FOUND, **DATABASE_UNAVAILABLE},
)
def get_sales_analytics(dataset_id: str) -> AnalyticsResponse:
    stored = _get_dataset(dataset_id)
    sales = stored.sales
    return AnalyticsResponse(
        dataset_id=dataset_id,
        monthly_revenue=_records(revenue_by_month(sales)),
        product_performance=_records(product_performance(sales)),
        revenue_by_region=_records(revenue_by_region(sales)),
    )


@router.get(
    "/forecast/{dataset_id}",
    response_model=ForecastResponse,
    summary="Forecast monthly revenue",
    responses={**DATASET_NOT_FOUND, **DATABASE_UNAVAILABLE, **ANALYSIS_UNAVAILABLE},
)
def get_sales_forecast(
    dataset_id: str,
    horizon: int = Query(default=3, ge=1, le=12),
) -> ForecastResponse:
    stored = _get_dataset(dataset_id)
    try:
        result = forecast_monthly_revenue(stored.sales, horizon=horizon)
    except MLAnalysisError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return ForecastResponse(
        dataset_id=dataset_id,
        history=_records(result.history),
        forecast=_records(result.forecast),
        evaluation={
            "mae": result.evaluation.mae,
            "rmse": result.evaluation.rmse,
            "holdout_count": result.evaluation.holdout_count,
        },
    )


@router.get(
    "/segments/{dataset_id}",
    response_model=SegmentsResponse,
    summary="Segment customers using RFM values",
    responses={**DATASET_NOT_FOUND, **DATABASE_UNAVAILABLE, **ANALYSIS_UNAVAILABLE},
)
def get_customer_segments(
    dataset_id: str,
    n_clusters: int = Query(default=3, ge=2, le=20),
) -> SegmentsResponse:
    stored = _get_dataset(dataset_id)
    try:
        result = segment_customers(stored.sales, n_clusters=n_clusters)
    except MLAnalysisError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return SegmentsResponse(
        dataset_id=dataset_id,
        segments=_records(result.segments),
        customers=_records(result.customers),
    )


@router.get(
    "/anomalies/{dataset_id}",
    response_model=AnomaliesResponse,
    summary="Detect unusual revenue transactions",
    responses={**DATASET_NOT_FOUND, **DATABASE_UNAVAILABLE, **ANALYSIS_UNAVAILABLE},
)
def get_sales_anomalies(dataset_id: str) -> AnomaliesResponse:
    stored = _get_dataset(dataset_id)
    try:
        anomalies = detect_revenue_anomalies(stored.sales)
    except MLAnalysisError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return AnomaliesResponse(
        dataset_id=dataset_id,
        anomaly_count=len(anomalies),
        anomalies=_records(anomalies),
    )


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Answer a supported business question",
    responses={
        **DATASET_NOT_FOUND,
        **DATABASE_UNAVAILABLE,
        422: {"description": "The question is invalid or its requested analysis is unavailable."},
        502: {"description": "The LLM provider request failed."},
    },
)
def ask_business_question(request: AskRequest) -> AskResponse:
    stored = _get_dataset(request.dataset_id)
    try:
        result = answer_natural_language_question(stored.sales, request.question)
    except LLMConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except LLMServiceError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    analysis = result.analysis
    return AskResponse(
        dataset_id=request.dataset_id,
        supported=analysis.intent is not None,
        analysis_type=analysis.intent,
        answer=result.answer,
        evidence=analysis.evidence,
        message=analysis.message,
    )


@documents_router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a PDF document",
    responses={
        413: {"description": "The PDF exceeds the 15 MiB limit."},
        422: {"description": "The PDF is invalid, encrypted, or contains no extractable text."},
        503: {"description": "The document store is unavailable."},
    },
)
def upload_pdf_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    try:
        if not file.filename:
            raise HTTPException(status_code=422, detail="The uploaded PDF must have a filename.")
        content = file.file.read(15 * 1024 * 1024 + 1)
        if len(content) > 15 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="The PDF exceeds the 15 MiB upload limit.")
        try:
            result = ingest_pdf(file.filename, content)
        except DocumentError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except DocumentStoreError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
    finally:
        file.file.close()
    return DocumentUploadResponse(**result)


@documents_router.get("", response_model=list[DocumentSummary], summary="List indexed PDF documents")
def get_documents() -> list[DocumentSummary]:
    try:
        return [DocumentSummary(**item) for item in list_documents()]
    except DocumentStoreError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@documents_router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an indexed PDF and its text chunks",
    responses={404: {"description": "Document ID does not exist."}, 503: {"description": "The document store is unavailable."}},
)
def delete_pdf_document(document_id: str) -> None:
    try:
        deleted = delete_document(document_id)
    except DocumentStoreError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")


@documents_router.post(
    "/ask",
    response_model=DocumentAskResponse,
    summary="Answer a question from indexed PDF documents",
    responses={
        422: {"description": "The question is invalid."},
        502: {"description": "The LLM provider request failed."},
        503: {"description": "The LLM or document store is unavailable."},
    },
)
def ask_documents(request: DocumentAskRequest) -> DocumentAskResponse:
    try:
        result = answer_document_question(request.question, document_id=request.document_id)
    except DocumentError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except LLMConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except DocumentStoreError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except LLMServiceError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return DocumentAskResponse(**result)
