"""Filtering and grouped sales summaries used by the dashboard."""

from datetime import date

import pandas as pd


def filter_sales_data(
    sales: pd.DataFrame,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    regions: list[str] | None = None,
    products: list[str] | None = None,
) -> pd.DataFrame:
    """Return records matching the selected date, region, and product filters."""
    matches = pd.Series(True, index=sales.index)
    dates = pd.to_datetime(sales["date"])
    if start_date is not None:
        matches &= dates.dt.date >= start_date
    if end_date is not None:
        matches &= dates.dt.date <= end_date
    if regions is not None:
        matches &= sales["region"].isin(regions)
    if products is not None:
        matches &= sales["product"].isin(products)
    return sales.loc[matches].copy()


def revenue_by_month(sales: pd.DataFrame) -> pd.DataFrame:
    """Group revenue into calendar months for a trend chart."""
    if sales.empty:
        return pd.DataFrame(columns=["month", "revenue"])
    monthly = sales.assign(month=pd.to_datetime(sales["date"]).dt.to_period("M").dt.to_timestamp())
    return (
        monthly.groupby("month", as_index=False)["revenue"]
        .sum()
        .sort_values("month")
    )


def product_performance(sales: pd.DataFrame) -> pd.DataFrame:
    """Aggregate revenue and units sold by product, highest revenue first."""
    if sales.empty:
        return pd.DataFrame(columns=["product", "revenue", "units_sold"])
    return (
        sales.groupby("product", as_index=False)
        .agg(revenue=("revenue", "sum"), units_sold=("quantity", "sum"))
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )


def revenue_by_region(sales: pd.DataFrame) -> pd.DataFrame:
    """Aggregate revenue by region, highest revenue first."""
    if sales.empty:
        return pd.DataFrame(columns=["region", "revenue"])
    return (
        sales.groupby("region", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
