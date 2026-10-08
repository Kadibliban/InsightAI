from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

from app.data import (
    DataValidationError,
    clean_sales_data,
    load_and_clean_sales_data,
    load_sales_data,
    validate_sales_data,
)


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-02"],
            "customer": [" C001 ", " C001 "],
            "product": ["Desk Lamp", "Desk Lamp"],
            "category": [None, None],
            "region": ["East", "East"],
            "quantity": ["2", "2"],
            "unit_price": ["4200", "4200"],
            "revenue": [" ", None],
        }
    )


def test_loads_and_cleans_csv(tmp_path: Path) -> None:
    source = tmp_path / "sales.csv"
    valid_frame().to_csv(source, index=False)

    cleaned = load_and_clean_sales_data(source)

    assert len(cleaned) == 1
    assert cleaned.loc[0, "customer"] == "C001"
    assert cleaned.loc[0, "category"] == "Uncategorized"
    assert cleaned.loc[0, "date"] == pd.Timestamp("2025-01-02")
    assert cleaned.loc[0, "revenue"] == 8400


def test_loads_excel_file(tmp_path: Path) -> None:
    source = tmp_path / "sales.xlsx"
    valid_frame().iloc[:1].to_excel(source, index=False)

    loaded = load_and_clean_sales_data(source)

    assert loaded.loc[0, "product"] == "Desk Lamp"
    assert loaded.loc[0, "revenue"] == 8400


def test_loads_uploaded_csv_buffer_from_start() -> None:
    upload = BytesIO(valid_frame().iloc[:1].to_csv(index=False).encode())
    upload.name = "upload.csv"
    upload.seek(10)

    loaded = load_sales_data(upload)

    assert len(loaded) == 1
    assert loaded.loc[0, "product"] == "Desk Lamp"


def test_loader_rejects_unsupported_file_type(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="Unsupported file type"):
        load_sales_data(tmp_path / "sales.json")


def test_loader_explains_legacy_excel_limit(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="Save the file as .xlsx"):
        load_sales_data(tmp_path / "sales.xls")


def test_normalizes_column_names_and_drops_blank_rows() -> None:
    frame = valid_frame().iloc[:1].copy()
    frame.columns = [" Date ", "Customer", "Product", "Category", "Region", "Quantity", "Unit Price", "Revenue"]
    frame.loc[1] = [None] * len(frame.columns)

    cleaned = clean_sales_data(frame)

    assert "unit_price" in cleaned.columns
    assert len(cleaned) == 1


def test_reports_missing_columns() -> None:
    frame = pd.DataFrame({"date": ["2025-01-01"]})

    with pytest.raises(DataValidationError, match="Missing required columns: customer"):
        validate_sales_data(frame)


def test_reports_duplicate_headers_after_normalization() -> None:
    frame = pd.DataFrame([["2025-01-01", "2025-01-02"]], columns=["Date", " date "])

    with pytest.raises(DataValidationError, match="duplicated after normalization"):
        clean_sales_data(frame)


def test_keeps_rows_with_distinct_additional_fields() -> None:
    frame = valid_frame().iloc[:1].copy()
    frame["order_id"] = ["A-1"]
    second = frame.copy()
    second["order_id"] = "A-2"

    cleaned = clean_sales_data(pd.concat([frame, second], ignore_index=True))

    assert len(cleaned) == 2


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("date", "not a date", "Invalid or missing dates"),
        ("quantity", "many", "Non-numeric values in 'quantity'"),
        ("unit_price", "many", "Non-numeric values in 'unit_price'"),
        ("customer", " ", "Missing customer, product, or region values"),
    ],
)
def test_reports_invalid_record_values(column: str, value: str, message: str) -> None:
    frame = valid_frame().iloc[:1].copy()
    frame.loc[0, column] = value

    with pytest.raises(DataValidationError, match=message):
        clean_sales_data(frame)


def test_rejects_nonpositive_quantity_and_negative_unit_price() -> None:
    frame = valid_frame().iloc[:1].copy()
    frame.loc[0, "quantity"] = "0"
    with pytest.raises(DataValidationError, match="Quantity must be greater than zero"):
        clean_sales_data(frame)

    frame = valid_frame().iloc[:1].copy()
    frame.loc[0, "unit_price"] = "-1"
    with pytest.raises(DataValidationError, match="unit_price cannot be negative"):
        clean_sales_data(frame)


def test_rejects_empty_data() -> None:
    frame = valid_frame().iloc[:0]

    with pytest.raises(DataValidationError, match="no sales records"):
        clean_sales_data(frame)


def test_sample_dataset_loads_and_cleans() -> None:
    sample = Path(__file__).parents[1] / "data" / "sample_sales.csv"

    cleaned = load_and_clean_sales_data(sample)

    assert len(cleaned) == 24
    assert cleaned["revenue"].sum() == 654300
    assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])
