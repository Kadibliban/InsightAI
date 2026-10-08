from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_mock_engine, inspect

from app.data import load_and_clean_sales_data
from app.database import Base, SalesRepository, create_database_engine, get_database_url
from app.database.database import create_session_factory
from app.database.errors import DatabaseUnavailableError


def make_repository(url: str = "sqlite+pysqlite:///:memory:"):
    engine = create_database_engine(url)
    repository = SalesRepository(create_session_factory(engine))
    return engine, repository


def test_database_url_uses_sqlite_default_when_not_configured(monkeypatch):
    from app.database import database

    monkeypatch.setattr(database, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    assert get_database_url() == "sqlite:///./insightai.db"


def test_database_url_reads_environment_without_modification(monkeypatch):
    from app.database import database

    monkeypatch.setattr(database, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://user:pass@localhost:5432/insightai")

    assert get_database_url().startswith("postgresql+psycopg2://")


def test_empty_explicit_database_url_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        get_database_url("  ")


def test_sqlite_repository_persists_and_reloads_sales_dataset():
    sample_path = Path(__file__).parents[1] / "data" / "sample_sales.csv"
    sales = load_and_clean_sales_data(sample_path)
    engine, repository = make_repository()
    try:
        dataset_id = repository.save_dataset("sample_sales.csv", sales)
        loaded = repository.get_dataset(dataset_id)

        assert loaded is not None
        assert loaded.id == dataset_id
        assert loaded.filename == "sample_sales.csv"
        assert loaded.record_count == len(sales)
        assert loaded.sales["revenue"].sum() == pytest.approx(sales["revenue"].sum())
        assert loaded.sales["date"].tolist() == pd.to_datetime(sales["date"]).tolist()
    finally:
        engine.dispose()


def test_delete_removes_metadata_and_sales_rows():
    sample_path = Path(__file__).parents[1] / "data" / "sample_sales.csv"
    sales = load_and_clean_sales_data(sample_path)
    engine, repository = make_repository()
    try:
        dataset_id = repository.save_dataset("sample_sales.csv", sales)

        assert repository.delete_dataset(dataset_id) is True
        assert repository.get_dataset(dataset_id) is None
        assert repository.delete_dataset(dataset_id) is False
    finally:
        engine.dispose()


def test_schema_has_dataset_metadata_and_sales_rows():
    engine, repository = make_repository()
    try:
        repository.initialize()
        inspector = inspect(engine)

        assert set(inspector.get_table_names()) == {"datasets", "sales_records"}
        assert "filename" in {column["name"] for column in inspector.get_columns("datasets")}
        assert "revenue" in {column["name"] for column in inspector.get_columns("sales_records")}
        assert inspector.get_foreign_keys("sales_records")[0]["referred_table"] == "datasets"
    finally:
        engine.dispose()


def test_schema_compiles_for_postgresql_dialect_without_network_connection():
    statements = []
    engine = create_mock_engine(
        "postgresql+psycopg2://user:pass@localhost/insightai",
        lambda statement, *args, **kwargs: statements.append(
            str(statement.compile(dialect=engine.dialect))
        ),
    )
    Base.metadata.create_all(engine)

    assert any("CREATE TABLE datasets" in statement for statement in statements)
    assert any("CREATE TABLE sales_records" in statement for statement in statements)


def test_unavailable_database_errors_are_sanitized():
    assert "password" not in str(DatabaseUnavailableError("The database is unavailable."))
