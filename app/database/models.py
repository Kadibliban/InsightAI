"""Relational tables for uploaded sales datasets and their business records."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.database import Base


class DatasetModel(Base):
    """Metadata for one uploaded dataset."""

    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    record_count: Mapped[int] = mapped_column(Integer, nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    sales_records: Mapped[list[SalesRecordModel]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class SalesRecordModel(Base):
    """One validated sales row belonging to an uploaded dataset."""

    __tablename__ = "sales_records"
    __table_args__ = (
        Index("ix_sales_dataset_date", "dataset_id", "date"),
        Index("ix_sales_dataset_customer", "dataset_id", "customer"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    date: Mapped[date] = mapped_column(Date, nullable=False)
    customer: Mapped[str] = mapped_column(Text, nullable=False)
    product: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(Text, nullable=False)
    region: Mapped[str] = mapped_column(Text, nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    unit_price: Mapped[float] = mapped_column(Float, nullable=False)
    revenue: Mapped[float] = mapped_column(Float, nullable=False)

    dataset: Mapped[DatasetModel] = relationship(back_populates="sales_records")
