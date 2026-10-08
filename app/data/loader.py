"""Load CSV and modern Excel sales files into pandas data frames."""

from pathlib import Path
from typing import BinaryIO

import pandas as pd

from app.data.cleaner import clean_sales_data
from app.data.validator import DataValidationError

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xlsm"}


def _file_name(source: str | Path | BinaryIO) -> str:
    name = getattr(source, "name", source)
    return str(name)


def load_sales_data(source: str | Path | BinaryIO) -> pd.DataFrame:
    """Read a CSV or .xlsx/.xlsm workbook; file-like uploads need a filename."""
    name = _file_name(source)
    extension = Path(name).suffix.lower()
    if extension == ".xls":
        raise DataValidationError(
            "Legacy .xls workbooks are not supported. Save the file as .xlsx and try again."
        )
    if extension not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DataValidationError(f"Unsupported file type '{extension or '(none)'}'. Use {allowed}.")

    try:
        if hasattr(source, "seek"):
            source.seek(0)
        if extension == ".csv":
            return pd.read_csv(source)
        return pd.read_excel(source, engine="openpyxl")
    except Exception as error:
        raise DataValidationError(
            f"Could not read '{Path(name).name}'. Check that it is a valid {extension} file."
        ) from error


def load_and_clean_sales_data(source: str | Path | BinaryIO) -> pd.DataFrame:
    """Load a supported file and return validated, cleaned sales records."""
    return clean_sales_data(load_sales_data(source))
