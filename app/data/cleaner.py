"""Clean sales records using explicit, explainable rules."""

import numpy as np
import pandas as pd

from app.data.validator import DataValidationError, normalize_column_names, validate_sales_data

TEXT_COLUMNS = ("customer", "product", "category", "region")
REQUIRED_TEXT_COLUMNS = ("customer", "product", "region")
NUMERIC_COLUMNS = ("quantity", "unit_price", "revenue")


def _row_numbers(mask: pd.Series, source_rows: pd.Series) -> str:
    positions = np.flatnonzero(mask.to_numpy())
    rows = [str(source_rows.iloc[position]) for position in positions[:5]]
    remainder = len(positions) - len(rows)
    suffix = f" and {remainder} more" if remainder else ""
    return ", ".join(rows) + suffix


def _raise_for_rows(message: str, mask: pd.Series, source_rows: pd.Series) -> None:
    if mask.any():
        raise DataValidationError(f"{message} Data rows: {_row_numbers(mask, source_rows)}.")


def clean_sales_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize and clean records; reject invalid values rather than guessing."""
    if not isinstance(data, pd.DataFrame):
        raise DataValidationError("Sales data must be provided as a pandas DataFrame.")

    cleaned = normalize_column_names(data)
    validate_sales_data(cleaned)
    source_rows = pd.Series(np.arange(len(cleaned)) + 2, index=cleaned.index)

    # Completely blank spreadsheet rows are harmless; remove them before validation.
    non_empty = ~cleaned.drop(columns=[]).isna().all(axis=1)
    cleaned = cleaned.loc[non_empty].copy()
    source_rows = source_rows.loc[non_empty]
    cleaned = cleaned.reset_index(drop=True)
    source_rows = source_rows.reset_index(drop=True)

    for column in TEXT_COLUMNS:
        cleaned[column] = cleaned[column].astype("string").str.strip()
        cleaned[column] = cleaned[column].replace("", pd.NA)

    missing_text = cleaned[list(REQUIRED_TEXT_COLUMNS)].isna().any(axis=1)
    _raise_for_rows("Missing customer, product, or region values.", missing_text, source_rows)
    cleaned["category"] = cleaned["category"].fillna("Uncategorized")

    raw_dates = cleaned["date"]
    cleaned["date"] = pd.to_datetime(raw_dates, errors="coerce", format="mixed")
    invalid_dates = cleaned["date"].isna()
    _raise_for_rows("Invalid or missing dates.", invalid_dates, source_rows)

    for column in NUMERIC_COLUMNS:
        raw_values = cleaned[column]
        blank_values = raw_values.astype("string").str.strip().eq("").fillna(False)
        raw_values = raw_values.mask(blank_values)
        cleaned[column] = pd.to_numeric(raw_values, errors="coerce")
        invalid_values = raw_values.notna() & cleaned[column].isna()
        _raise_for_rows(f"Non-numeric values in '{column}'.", invalid_values, source_rows)

    missing_values = cleaned[["quantity", "unit_price"]].isna().any(axis=1)
    _raise_for_rows("Missing quantity or unit_price values.", missing_values, source_rows)

    for column in NUMERIC_COLUMNS:
        finite = cleaned[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(finite).all():
            invalid_values = cleaned[column].notna() & ~np.isfinite(
                cleaned[column].fillna(0).astype(float)
            )
            _raise_for_rows(f"Non-finite values in '{column}'.", invalid_values, source_rows)

    invalid_quantity = cleaned["quantity"] <= 0
    _raise_for_rows("Quantity must be greater than zero.", invalid_quantity, source_rows)
    invalid_price = cleaned["unit_price"] < 0
    _raise_for_rows("unit_price cannot be negative.", invalid_price, source_rows)

    # Revenue can reflect discounts or refunds, so only derive it when it is absent.
    missing_revenue = cleaned["revenue"].isna()
    cleaned.loc[missing_revenue, "revenue"] = (
        cleaned.loc[missing_revenue, "quantity"]
        * cleaned.loc[missing_revenue, "unit_price"]
    )

    # Drop exact duplicate records after trimming text and converting data types.
    cleaned = cleaned.drop_duplicates(keep="first")
    return cleaned.reset_index(drop=True)
