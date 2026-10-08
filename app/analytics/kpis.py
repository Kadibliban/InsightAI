"""Key performance indicators for the sales dashboard."""

from typing import TypedDict

import pandas as pd


class SalesKpis(TypedDict):
    total_revenue: float
    sales_transactions: int
    average_order_value: float
    customer_count: int
    units_sold: float


def calculate_kpis(sales: pd.DataFrame) -> SalesKpis:
    """Calculate dashboard KPIs from cleaned sales records.

    Each row represents one sales transaction because the starter schema does
    not include an order identifier.
    """
    total_revenue = float(sales["revenue"].sum()) if not sales.empty else 0.0
    sales_transactions = int(len(sales))
    return {
        "total_revenue": total_revenue,
        "sales_transactions": sales_transactions,
        "average_order_value": (
            total_revenue / sales_transactions if sales_transactions else 0.0
        ),
        "customer_count": int(sales["customer"].nunique()) if not sales.empty else 0,
        "units_sold": float(sales["quantity"].sum()) if not sales.empty else 0.0,
    }
