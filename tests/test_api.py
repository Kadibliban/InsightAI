from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.api import routes
from app.api.routes import (
    _clean_uploaded_file,
    _records,
    delete_sales_data,
    get_customer_segments,
    get_sales_analytics,
    get_sales_anomalies,
    get_sales_data,
    get_sales_forecast,
    get_sales_kpis,
    upload_sales_file,
)
from app.api.schemas import AskRequest, AskResponse, DocumentAskRequest, DocumentAskResponse
from app.database import SalesRepository, create_database_engine
from app.database.database import create_session_factory
from app.database.errors import DatabaseUnavailableError
from app.data import load_and_clean_sales_data
from app.main import app, health_check


@pytest.fixture
def sales() -> pd.DataFrame:
    sample = Path(__file__).parents[1] / "data" / "sample_sales.csv"
    return load_and_clean_sales_data(sample)


@pytest.fixture(autouse=True)
def use_sqlite_repository(monkeypatch):
    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    repository = SalesRepository(create_session_factory(engine))
    monkeypatch.setattr(routes, "sales_repository", repository)
    yield repository
    engine.dispose()


def stored_sample(sales: pd.DataFrame) -> str:
    return routes.sales_repository.save_dataset("sample_sales.csv", sales)


def test_health_function_returns_ok():
    assert health_check() == {"status": "ok"}


def test_openapi_lists_expected_api_endpoints():
    openapi = app.openapi()
    paths = openapi["paths"]

    assert {"post"} <= set(paths["/upload"])
    assert {"get", "delete"} <= set(paths["/data/{dataset_id}"])
    for path in (
        "/kpis/{dataset_id}",
        "/analytics/{dataset_id}",
        "/forecast/{dataset_id}",
        "/segments/{dataset_id}",
        "/anomalies/{dataset_id}",
        "/ask",
        "/documents/upload",
        "/documents",
        "/documents/{document_id}",
        "/documents/ask",
        "/health",
    ):
        assert path in paths
    upload_description = paths["/upload"]["post"]["description"]
    assert "persist cleaned records in the database" in upload_description
    assert "503" in paths["/upload"]["post"]["responses"]
    assert "503" in paths["/ask"]["post"]["responses"]


def test_upload_loader_reuses_validation_and_cleans_csv(sales):
    path = Path(__file__).parents[1] / "data" / "sample_sales.csv"
    loaded = _clean_uploaded_file("folder\\sample_sales.csv", path.read_bytes())

    assert len(loaded) == len(sales)
    assert loaded["revenue"].sum() == sales["revenue"].sum()


def test_upload_loader_returns_readable_validation_error():
    with pytest.raises(HTTPException) as error:
        _clean_uploaded_file("bad.csv", b"date,customer\nnot,a sales file\n")

    assert error.value.status_code == 422
    assert "Missing required columns" in error.value.detail


def test_upload_loader_rejects_oversized_content():
    with pytest.raises(HTTPException) as error:
        _clean_uploaded_file("big.csv", b"x" * (10 * 1024 * 1024 + 1))

    assert error.value.status_code == 413


def test_upload_endpoint_reads_a_bounded_file_and_stores_cleaned_data(sales):
    sample_path = Path(__file__).parents[1] / "data" / "sample_sales.csv"

    class FakeUpload:
        filename = "sample_sales.csv"

        def __init__(self, content: bytes):
            self.file = BytesIO(content)

    upload = FakeUpload(sample_path.read_bytes())

    response = upload_sales_file(upload)

    assert response.filename == "sample_sales.csv"
    assert response.record_count == len(sales)
    assert upload.file.closed


def test_sales_upload_stores_only_the_client_filename():
    sample_path = Path(__file__).parents[1] / "data" / "sample_sales.csv"

    class FakeUpload:
        filename = r"C:\private\reports\sample_sales.csv"

        def __init__(self):
            self.file = BytesIO(sample_path.read_bytes())

    response = upload_sales_file(FakeUpload())

    assert response.filename == "sample_sales.csv"


def test_upload_endpoint_closes_file_for_oversized_request():
    class FakeUpload:
        filename = "large.csv"

        def __init__(self):
            self.file = BytesIO(b"x" * (10 * 1024 * 1024 + 1))

    upload = FakeUpload()
    with pytest.raises(HTTPException) as error:
        upload_sales_file(upload)

    assert error.value.status_code == 413
    assert upload.file.closed


def test_repository_persists_an_immutable_snapshot_and_deletes():
    source = pd.DataFrame(
        [{"date": "2025-01-01", "customer": "C1", "product": "P1", "category": "C", "region": "East", "quantity": 1, "unit_price": 10, "revenue": 10}]
    )
    cleaned = _clean_uploaded_file("one.csv", source.to_csv(index=False).encode())
    repository = routes.sales_repository
    dataset_id = repository.save_dataset("one.csv", cleaned)
    cleaned.loc[0, "revenue"] = 20

    stored = repository.get_dataset(dataset_id)
    assert stored.sales.loc[0, "revenue"] == 10
    assert repository.delete_dataset(dataset_id) is True
    assert repository.delete_dataset(dataset_id) is False


def test_unknown_dataset_returns_404():
    with pytest.raises(HTTPException) as error:
        get_sales_kpis("unknown")

    assert error.value.status_code == 404


def test_unavailable_database_is_reported_as_503(monkeypatch):
    class UnavailableRepository:
        def get_dataset(self, dataset_id):
            raise DatabaseUnavailableError("The configured database is unavailable.")

    monkeypatch.setattr(routes, "sales_repository", UnavailableRepository())

    with pytest.raises(HTTPException) as error:
        get_sales_kpis("dataset-id")

    assert error.value.status_code == 503
    assert "configured database is unavailable" in error.value.detail


def test_data_endpoint_returns_json_safe_records(sales):
    dataset_id = stored_sample(sales)

    response = get_sales_data(dataset_id)

    assert response.record_count == len(sales)
    assert response.data[0]["date"].startswith("2025-")
    assert response.data[0]["customer"] == sales.iloc[0]["customer"]


def test_kpi_endpoint_returns_calculated_values(sales):
    dataset_id = stored_sample(sales)

    response = get_sales_kpis(dataset_id)

    assert response.kpis["total_revenue"] == pytest.approx(sales["revenue"].sum())
    assert response.kpis["sales_transactions"] == len(sales)


def test_analytics_endpoint_returns_grouped_summaries(sales):
    dataset_id = stored_sample(sales)

    response = get_sales_analytics(dataset_id)

    assert sum(row["revenue"] for row in response.monthly_revenue) == pytest.approx(
        sales["revenue"].sum()
    )
    assert response.product_performance
    assert response.revenue_by_region


def test_forecast_endpoint_returns_prediction_and_evaluation(sales):
    dataset_id = stored_sample(sales)

    response = get_sales_forecast(dataset_id, horizon=2)

    assert len(response.forecast) == 2
    assert response.evaluation["mae"] >= 0
    assert response.evaluation["holdout_count"] >= 1


def test_forecast_endpoint_maps_model_errors_to_422(sales):
    one_month = sales[pd.to_datetime(sales["date"]).dt.month == 1]
    dataset_id = stored_sample(one_month)

    with pytest.raises(HTTPException) as error:
        get_sales_forecast(dataset_id)

    assert error.value.status_code == 422


def test_segments_endpoint_returns_profiles_and_customers(sales):
    dataset_id = stored_sample(sales)

    response = get_customer_segments(dataset_id, n_clusters=3)

    assert response.segments
    assert len(response.customers) == sales["customer"].nunique()


def test_anomalies_endpoint_returns_records_and_count(sales):
    dataset_id = stored_sample(sales)

    response = get_sales_anomalies(dataset_id)

    assert response.anomaly_count == len(response.anomalies)


def test_delete_endpoint_removes_dataset(sales):
    dataset_id = stored_sample(sales)

    delete_sales_data(dataset_id)

    with pytest.raises(HTTPException) as error:
        get_sales_data(dataset_id)
    assert error.value.status_code == 404


def test_ask_request_trims_fields_and_rejects_blank_values():
    assert AskRequest(dataset_id=" id ", question=" What is revenue? ").question == "What is revenue?"
    with pytest.raises(ValidationError):
        AskRequest(dataset_id="id", question="  ")


def test_dataframe_record_serializer_converts_nulls_and_dates():
    data = pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "value": [None]})

    result = _records(data)
    assert pd.to_datetime(result[0]["date"]) == pd.Timestamp("2025-01-01")
    assert result[0]["value"] is None


def test_ask_endpoint_returns_analysis_and_python_evidence(monkeypatch, sales):
    dataset_id = stored_sample(sales)
    analysis = SimpleNamespace(
        intent="total_revenue",
        evidence={"total_revenue": 654300.0},
        message=None,
    )
    monkeypatch.setattr(
        routes,
        "answer_natural_language_question",
        lambda *args, **kwargs: SimpleNamespace(analysis=analysis, answer="Finding: revenue is 654,300."),
    )

    response = routes.ask_business_question(
        AskRequest(dataset_id=dataset_id, question="What was our total revenue?")
    )

    assert isinstance(response, AskResponse)
    assert response.supported is True
    assert response.analysis_type == "total_revenue"
    assert response.evidence["total_revenue"] == 654300.0


def test_ask_endpoint_returns_unsupported_guidance_without_provider(monkeypatch, sales):
    dataset_id = stored_sample(sales)
    analysis = SimpleNamespace(intent=None, evidence={}, message="Unsupported question.")
    monkeypatch.setattr(
        routes,
        "answer_natural_language_question",
        lambda *args, **kwargs: SimpleNamespace(analysis=analysis, answer=None),
    )

    response = routes.ask_business_question(
        AskRequest(dataset_id=dataset_id, question="Tell me a joke")
    )

    assert response.supported is False
    assert response.message == "Unsupported question."


def test_ask_endpoint_maps_missing_llm_configuration_to_503(monkeypatch, sales):
    dataset_id = stored_sample(sales)

    def unavailable(*args, **kwargs):
        from app.ai.errors import LLMConfigurationError

        raise LLMConfigurationError("GROQ_API_KEY is not configured.")

    monkeypatch.setattr(routes, "answer_natural_language_question", unavailable)

    with pytest.raises(HTTPException) as error:
        routes.ask_business_question(
            AskRequest(dataset_id=dataset_id, question="What was our revenue?")
        )
    assert error.value.status_code == 503


def test_pdf_upload_endpoint_returns_index_summary_and_closes_file(monkeypatch):
    class FakeUpload:
        filename = "report.pdf"

        def __init__(self):
            self.file = BytesIO(b"%PDF-1.7 test")

    monkeypatch.setattr(
        routes,
        "ingest_pdf",
        lambda filename, content: {
            "document_id": "doc-id",
            "filename": filename,
            "page_count": 2,
            "chunk_count": 3,
        },
    )
    upload = FakeUpload()

    response = routes.upload_pdf_document(upload)

    assert response.document_id == "doc-id"
    assert response.filename == "report.pdf"
    assert response.chunk_count == 3
    assert upload.file.closed


def test_pdf_upload_endpoint_rejects_oversized_file_and_closes_it():
    class FakeUpload:
        filename = "large.pdf"

        def __init__(self):
            self.file = BytesIO(b"x" * (15 * 1024 * 1024 + 1))

    upload = FakeUpload()
    with pytest.raises(HTTPException) as error:
        routes.upload_pdf_document(upload)

    assert error.value.status_code == 413
    assert upload.file.closed


def test_document_list_and_delete_endpoints(monkeypatch):
    monkeypatch.setattr(
        routes,
        "list_documents",
        lambda: [
            {
                "document_id": "doc-id",
                "filename": "report.pdf",
                "page_count": 2,
                "chunk_count": 3,
                "created_at": "2026-10-08T10:00:00+00:00",
            }
        ],
    )
    response = routes.get_documents()
    assert response[0].document_id == "doc-id"
    assert response[0].filename == "report.pdf"

    monkeypatch.setattr(routes, "delete_document", lambda document_id: document_id == "doc-id")
    assert routes.delete_pdf_document("doc-id") is None
    with pytest.raises(HTTPException) as error:
        routes.delete_pdf_document("unknown")
    assert error.value.status_code == 404


def test_document_ask_endpoint_returns_answer_and_sources(monkeypatch):
    monkeypatch.setattr(
        routes,
        "answer_document_question",
        lambda question, document_id=None: {
            "answer": "Riara University [report.pdf, page 1].",
            "sources": [
                {
                    "document_id": document_id,
                    "filename": "report.pdf",
                    "page_number": 1,
                    "similarity": 0.72,
                }
            ],
        },
    )

    response = routes.ask_documents(
        DocumentAskRequest(question="Where does the student study?", document_id="doc-id")
    )

    assert isinstance(response, DocumentAskResponse)
    assert "Riara University" in response.answer
    assert response.sources[0].page_number == 1


def test_document_ask_endpoint_maps_unavailable_llm_to_503(monkeypatch):
    from app.ai.errors import LLMConfigurationError

    def unavailable(*args, **kwargs):
        raise LLMConfigurationError("GROQ_API_KEY is not configured.")

    monkeypatch.setattr(routes, "answer_document_question", unavailable)
    with pytest.raises(HTTPException) as error:
        routes.ask_documents(DocumentAskRequest(question="Question?"))

    assert error.value.status_code == 503
