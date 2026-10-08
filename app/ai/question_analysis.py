"""Route natural-language questions to a small set of safe Python analyses."""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass

import pandas as pd

from app.analytics.kpis import calculate_kpis
from app.analytics.sales_analysis import product_performance, revenue_by_region
from app.ml.anomaly_detection import detect_revenue_anomalies
from app.ml.errors import MLAnalysisError
from app.ml.forecasting import forecast_monthly_revenue

SUPPORTED_QUESTION_TYPES = (
    "total revenue",
    "monthly revenue",
    "best-selling products",
    "regional revenue",
    "top customers by sales",
    "revenue decreases and contributors",
    "lowest-revenue products",
    "next-month forecast",
    "revenue anomalies",
)


@dataclass(frozen=True)
class QuestionAnalysis:
    """A deterministic intent choice and the calculated evidence for it."""

    intent: str | None
    evidence: dict[str, object]
    message: str | None = None


def identify_intent(question: str) -> str | None:
    """Map a natural-language question to a predefined analysis identifier."""
    if not isinstance(question, str) or not question.strip():
        return None

    text = re.sub(r"[^a-z0-9]+", " ", question.casefold()).strip()
    has_revenue = "revenue" in text or "sales" in text
    has_product = "product" in text or "products" in text
    has_month = _mentioned_month(question) is not None

    if any(word in text for word in ("anomaly", "anomalies", "unusual", "outlier")):
        return "anomalies"
    if any(word in text for word in ("forecast", "predict", "prediction", "expected", "next month")) and has_revenue:
        return "forecast"
    if has_product and any(word in text for word in ("underperform", "poorly", "weak", "lowest")):
        return "underperforming_products"
    if has_revenue and any(
        word in text
        for word in ("decrease", "decreased", "decreasing", "decline", "declined", "drop", "dropped", "fall", "fell")
    ):
        return "revenue_decrease"
    if has_product and any(
        phrase in text
        for phrase in ("best selling", "best seller", "most sold", "top selling", "sold the most")
    ):
        return "best_selling_products"
    if "region" in text and (has_revenue or any(word in text for word in ("top", "most", "highest"))):
        return "regional_revenue"
    if "customer" in text and any(
        word in text for word in ("spent", "spend", "valuable", "top", "most", "highest")
    ):
        return "top_customers"
    if has_revenue and (has_month or "monthly" in text):
        return "monthly_revenue"
    if has_revenue and any(word in text for word in ("total", "overall", "all")):
        return "total_revenue"
    if text in {"revenue", "what was our revenue", "how much revenue"}:
        return "total_revenue"
    return None


def analyze_business_question(sales: pd.DataFrame, question: str) -> QuestionAnalysis:
    """Calculate evidence for a supported question without executing model code."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Enter a business question.")
    if not isinstance(sales, pd.DataFrame):
        raise ValueError("Sales data must be a pandas DataFrame.")

    intent = identify_intent(question)
    if intent is None:
        examples = "; ".join(SUPPORTED_QUESTION_TYPES)
        return QuestionAnalysis(
            None,
            {},
            "I can answer questions about " + examples + ". Try rephrasing your question.",
        )
    if sales.empty:
        return QuestionAnalysis(intent, {}, "There are no sales records matching the current filters.")

    if intent == "total_revenue":
        kpis = calculate_kpis(sales)
        return QuestionAnalysis(
            intent,
            {"total_revenue": kpis["total_revenue"], "sales_records": kpis["sales_transactions"]},
        )
    if intent == "monthly_revenue":
        return QuestionAnalysis(intent, _monthly_revenue_evidence(sales, question))
    if intent == "best_selling_products":
        return QuestionAnalysis(intent, _best_selling_products(sales))
    if intent == "regional_revenue":
        return QuestionAnalysis(intent, _top_regions(sales))
    if intent == "top_customers":
        return QuestionAnalysis(intent, _top_customers(sales))
    if intent == "revenue_decrease":
        return QuestionAnalysis(intent, _revenue_decrease_evidence(sales, question))
    if intent == "underperforming_products":
        return QuestionAnalysis(intent, _underperforming_products(sales))
    if intent == "forecast":
        try:
            result = forecast_monthly_revenue(sales, horizon=1)
        except MLAnalysisError as error:
            return QuestionAnalysis(intent, {}, f"A forecast is unavailable: {error}")
        predicted = result.forecast.iloc[0]
        return QuestionAnalysis(
            intent,
            {
                "forecast_month": predicted["month"].date().isoformat(),
                "predicted_revenue": float(predicted["predicted_revenue"]),
                "holdout_mae": result.evaluation.mae,
                "holdout_rmse": result.evaluation.rmse,
                "method": "Linear monthly time trend evaluated on a chronological holdout.",
            },
        )
    if intent == "anomalies":
        try:
            anomalies = detect_revenue_anomalies(sales)
        except MLAnalysisError as error:
            return QuestionAnalysis(intent, {}, f"Anomaly detection is unavailable: {error}")
        return QuestionAnalysis(
            intent,
            {
                "method": "Tukey IQR bounds with a 1.5 multiplier.",
                "anomaly_count": int(len(anomalies)),
                "anomalies": [
                    {
                        "date": pd.to_datetime(row.date).date().isoformat(),
                        "product": str(row.product),
                        "region": str(row.region),
                        "revenue": float(row.revenue),
                        "direction": str(row.anomaly_direction),
                    }
                    for row in anomalies.head(5).itertuples(index=False)
                ],
            },
        )

    return QuestionAnalysis(None, {}, "This question type is not supported yet.")


def _mentioned_month(question: str) -> tuple[int, int | None] | None:
    month_names = {name.casefold(): number for number, name in enumerate(calendar.month_name) if name}
    month_names.update(
        {name.casefold(): number for number, name in enumerate(calendar.month_abbr) if name}
    )
    for name, number in month_names.items():
        if re.search(rf"\b{re.escape(name)}\b", question, flags=re.IGNORECASE):
            year_match = re.search(r"\b(19\d{2}|20\d{2}|21\d{2})\b", question)
            return number, int(year_match.group(1)) if year_match else None
    return None


def _monthly_frame(sales: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(sales["date"])
    monthly = sales.assign(month=dates.dt.to_period("M")).groupby("month")["revenue"].sum()
    months = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    return (
        monthly.reindex(months, fill_value=0.0)
        .rename("revenue")
        .rename_axis("month")
        .reset_index()
    )


def _monthly_revenue_evidence(sales: pd.DataFrame, question: str) -> dict[str, object]:
    monthly = _monthly_frame(sales)
    requested = _mentioned_month(question)
    matches = monthly
    if requested is not None:
        matches = matches[matches["month"].map(lambda value: value.month == requested[0])]
        if requested[1] is not None:
            matches = matches[matches["month"].map(lambda value: value.year == requested[1])]
    return {
        "requested_month": calendar.month_name[requested[0]] if requested else None,
        "requested_year": requested[1] if requested else None,
        "found": not matches.empty,
        "monthly_revenue": [
            {"month": value.strftime("%B %Y"), "revenue": float(revenue)}
            for value, revenue in zip(matches["month"].dt.to_timestamp(), matches["revenue"])
        ],
    }


def _best_selling_products(sales: pd.DataFrame) -> dict[str, object]:
    performance = product_performance(sales).sort_values("units_sold", ascending=False)
    if performance.empty:
        return {"found": False, "best_selling_products": []}
    max_units = performance.iloc[0]["units_sold"]
    leaders = performance[performance["units_sold"] == max_units]
    return {
        "measure": "total units sold",
        "units_sold": float(max_units),
        "best_selling_products": [str(product) for product in leaders["product"]],
    }


def _top_regions(sales: pd.DataFrame) -> dict[str, object]:
    regions = revenue_by_region(sales)
    if regions.empty:
        return {"found": False, "top_regions": []}
    max_revenue = regions.iloc[0]["revenue"]
    leaders = regions[regions["revenue"] == max_revenue]
    return {
        "measure": "total revenue",
        "revenue": float(max_revenue),
        "top_regions": [str(region) for region in leaders["region"]],
    }


def _top_customers(sales: pd.DataFrame) -> dict[str, object]:
    customers = (
        sales.groupby("customer", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
        .head(5)
    )
    return {
        "measure": "total revenue per customer",
        "top_customers": [
            {"customer": str(row.customer), "revenue": float(row.revenue)}
            for row in customers.itertuples(index=False)
        ],
        "note": "Customer labels are included only because the question asks for customer rankings.",
    }


def _underperforming_products(sales: pd.DataFrame) -> dict[str, object]:
    products = product_performance(sales).sort_values("revenue").head(5)
    return {
        "measure": "lowest total revenue in the selected data",
        "products": [
            {
                "product": str(row.product),
                "revenue": float(row.revenue),
                "units_sold": float(row.units_sold),
            }
            for row in products.itertuples(index=False)
        ],
        "limitation": "Sales targets, costs, and profit margins are not available, so low revenue alone does not establish poor business performance.",
    }


def _revenue_decrease_evidence(sales: pd.DataFrame, question: str) -> dict[str, object]:
    monthly = _monthly_frame(sales)
    monthly["prior_revenue"] = monthly["revenue"].shift(1)
    monthly["change"] = monthly["revenue"] - monthly["prior_revenue"]
    requested = _mentioned_month(question)
    candidates = monthly[monthly["change"] < 0]
    if requested is not None:
        candidates = candidates[candidates["month"].map(lambda value: value.month == requested[0])]
        if requested[1] is not None:
            candidates = candidates[candidates["month"].map(lambda value: value.year == requested[1])]
    candidates = candidates.dropna(subset=["prior_revenue"])
    if candidates.empty:
        return {
            "found_decrease": False,
            "requested_month": calendar.month_name[requested[0]] if requested else None,
            "note": "No month-over-month revenue decrease was found in the selected data.",
        }

    entries = []
    for row in candidates.itertuples(index=False):
        month_start = row.month.to_timestamp()
        previous_start = (row.month - 1).to_timestamp()
        current_sales = sales[pd.to_datetime(sales["date"]).dt.to_period("M") == row.month]
        previous_sales = sales[pd.to_datetime(sales["date"]).dt.to_period("M") == row.month - 1]
        entries.append(
            {
                "month": month_start.strftime("%B %Y"),
                "previous_month": previous_start.strftime("%B %Y"),
                "revenue": float(row.revenue),
                "previous_revenue": float(row.prior_revenue),
                "change": float(row.change),
                "change_percent": (
                    float(row.change / abs(row.prior_revenue) * 100)
                    if row.prior_revenue != 0
                    else None
                ),
                "largest_product_revenue_declines": _largest_group_declines(
                    previous_sales, current_sales, "product"
                ),
                "largest_region_revenue_declines": _largest_group_declines(
                    previous_sales, current_sales, "region"
                ),
            }
        )
    if requested is None:
        entries = entries[-1:]
    return {
        "found_decrease": True,
        "decreases": entries,
        "interpretation_limit": "These are measured month-to-month changes by product and region; they do not establish why the changes happened.",
    }


def _largest_group_declines(
    previous_sales: pd.DataFrame, current_sales: pd.DataFrame, group_column: str
) -> list[dict[str, object]]:
    previous = previous_sales.groupby(group_column)["revenue"].sum()
    current = current_sales.groupby(group_column)["revenue"].sum()
    changes = current.subtract(previous, fill_value=0.0).sort_values()
    decreases = changes[changes < 0].head(3)
    return [
        {"name": str(name), "revenue_change": float(change)}
        for name, change in decreases.items()
    ]
