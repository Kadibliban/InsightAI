from datetime import date
from pathlib import Path

import pandas as pd

from app.analytics.kpis import calculate_kpis
from app.analytics.sales_analysis import (
    filter_sales_data,
    product_performance,
    revenue_by_month,
    revenue_by_region,
)
from app.data import load_and_clean_sales_data


def sample_sales() -> pd.DataFrame:
    sample = Path(__file__).parents[1] / "data" / "sample_sales.csv"
    return load_and_clean_sales_data(sample)


def test_kpis_match_sample_dataset() -> None:
    sales = sample_sales()

    kpis = calculate_kpis(sales)

    assert kpis["total_revenue"] == 654300
    assert kpis["sales_transactions"] == 24
    assert kpis["average_order_value"] == 27262.5
    assert kpis["customer_count"] == 8
    assert kpis["units_sold"] == sales["quantity"].sum()


def test_filters_combine_date_region_and_product() -> None:
    sales = sample_sales()

    filtered = filter_sales_data(
        sales,
        start_date=date(2025, 4, 1),
        end_date=date(2025, 6, 30),
        regions=["East"],
        products=["Standing Desk"],
    )

    assert len(filtered) == 2
    assert set(filtered["region"]) == {"East"}
    assert set(filtered["product"]) == {"Standing Desk"}
    assert filtered["revenue"].sum() == 84000


def test_grouped_chart_data_sums_to_filtered_revenue() -> None:
    sales = sample_sales()

    monthly = revenue_by_month(sales)
    products = product_performance(sales)
    regions = revenue_by_region(sales)

    assert len(monthly) == 6
    assert monthly["revenue"].sum() == sales["revenue"].sum()
    assert products.iloc[0]["revenue"] >= products.iloc[-1]["revenue"]
    assert products["revenue"].sum() == sales["revenue"].sum()
    assert regions.iloc[0]["revenue"] >= regions.iloc[-1]["revenue"]
    assert regions["revenue"].sum() == sales["revenue"].sum()


def test_empty_data_produces_zero_kpis_and_empty_chart_data() -> None:
    empty_sales = sample_sales().iloc[:0]

    kpis = calculate_kpis(empty_sales)

    assert kpis["total_revenue"] == 0
    assert kpis["sales_transactions"] == 0
    assert kpis["average_order_value"] == 0
    assert kpis["customer_count"] == 0
    assert revenue_by_month(empty_sales).empty
    assert product_performance(empty_sales).empty
    assert revenue_by_region(empty_sales).empty
