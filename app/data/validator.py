"""Schema validation and column normalization for sales data."""

import re

import pandas as pd

REQUIRED_COLUMNS = (
    "date",
    "customer",
    "product",
    "category",
    "region",
    "quantity",
    "unit_price",
    "revenue",
)


class DataValidationError(ValueError):
    """Raised when uploaded business data cannot be safely analyzed."""


def normalize_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with trimmed, lowercase, underscore-separated headers."""
    normalized = data.copy()
    columns = [
        re.sub(r"\s+", "_", str(column).strip().lower())
        for column in normalized.columns
    ]
    duplicates = sorted({column for column in columns if columns.count(column) > 1})
    if duplicates:
        names = ", ".join(duplicates)
        raise DataValidationError(f"Column names are duplicated after normalization: {names}.")
    normalized.columns = columns
    return normalized


def validate_sales_data(data: pd.DataFrame) -> None:
    """Validate that a dataset has the expected sales schema and records."""
    normalized = normalize_column_names(data)
    missing = [column for column in REQUIRED_COLUMNS if column not in normalized.columns]
    if missing:
        names = ", ".join(missing)
        raise DataValidationError(f"Missing required columns: {names}.")
    if normalized.empty or normalized.dropna(how="all").empty:
        raise DataValidationError("The dataset contains no sales records.")
