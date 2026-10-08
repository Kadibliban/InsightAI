"""Transactional persistence for uploaded sales datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any
from uuid import uuid4

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from app.database.database import Base
from app.database.errors import DatabaseUnavailableError
from app.database.models import DatasetModel, SalesRecordModel


@dataclass(frozen=True)
class DatabaseDataset:
    """A dataset and its cleaned sales rows loaded from persistent storage."""

    id: str
    filename: str
    record_count: int
    sales: pd.DataFrame


class SalesRepository:
    """Store and retrieve normalized datasets using SQLAlchemy sessions."""

    def __init__(self, sessions: sessionmaker) -> None:
        self._sessions = sessions
        self._initialized = False

    def initialize(self) -> None:
        """Create tables when absent. A migration framework is deferred."""
        if self._initialized:
            return
        try:
            bind = self._sessions.kw["bind"]
            Base.metadata.create_all(bind)
        except (SQLAlchemyError, KeyError, AttributeError) as error:
            raise DatabaseUnavailableError(
                "The configured database is unavailable or could not initialize its schema."
            ) from error
        self._initialized = True

    def save_dataset(self, filename: str, sales: pd.DataFrame) -> str:
        """Persist dataset metadata and every cleaned row atomically."""
        self.initialize()
        dataset_id = str(uuid4())
        safe_filename = filename.replace("\\", "/").rsplit("/", 1)[-1][:255]
        records: list[dict[str, Any]] = []
        for row in sales.itertuples(index=False):
            records.append(
                {
                    "date": pd.Timestamp(row.date).date(),
                    "customer": str(row.customer),
                    "product": str(row.product),
                    "category": str(row.category),
                    "region": str(row.region),
                    "quantity": float(row.quantity),
                    "unit_price": float(row.unit_price),
                    "revenue": float(row.revenue),
                }
            )
        try:
            with self._sessions.begin() as session:
                dataset = DatasetModel(
                    id=dataset_id,
                    filename=safe_filename,
                    record_count=len(records),
                )
                dataset.sales_records = [SalesRecordModel(**record) for record in records]
                session.add(dataset)
        except SQLAlchemyError as error:
            raise DatabaseUnavailableError(
                "The dataset could not be saved to the configured database."
            ) from error
        return dataset_id

    def get_dataset(self, dataset_id: str) -> DatabaseDataset | None:
        """Load metadata and rows, returning None when the ID does not exist."""
        self.initialize()
        try:
            with self._sessions() as session:
                dataset = session.get(DatasetModel, dataset_id)
                if dataset is None:
                    return None
                rows = session.scalars(
                    select(SalesRecordModel)
                    .where(SalesRecordModel.dataset_id == dataset_id)
                    .order_by(SalesRecordModel.id)
                ).all()
                sales = pd.DataFrame(
                    [
                        {
                            "date": pd.Timestamp(row.date),
                            "customer": row.customer,
                            "product": row.product,
                            "category": row.category,
                            "region": row.region,
                            "quantity": row.quantity,
                            "unit_price": row.unit_price,
                            "revenue": row.revenue,
                        }
                        for row in rows
                    ],
                    columns=[
                        "date",
                        "customer",
                        "product",
                        "category",
                        "region",
                        "quantity",
                        "unit_price",
                        "revenue",
                    ],
                )
                return DatabaseDataset(
                    id=dataset.id,
                    filename=dataset.filename,
                    record_count=dataset.record_count,
                    sales=sales,
                )
        except SQLAlchemyError as error:
            raise DatabaseUnavailableError(
                "The configured database could not retrieve the requested dataset."
            ) from error

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete dataset metadata and its sales rows in one transaction."""
        self.initialize()
        try:
            with self._sessions.begin() as session:
                dataset = session.get(DatasetModel, dataset_id)
                if dataset is None:
                    return False
                session.execute(
                    delete(SalesRecordModel).where(
                        SalesRecordModel.dataset_id == dataset_id
                    )
                )
                session.delete(dataset)
            return True
        except SQLAlchemyError as error:
            raise DatabaseUnavailableError(
                "The dataset could not be deleted from the configured database."
            ) from error
