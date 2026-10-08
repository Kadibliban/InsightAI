"""Aggregate sales evidence and answer questions through a grounded LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.ai.llm import complete_chat
from app.ai.prompts import build_analyst_messages
from app.ai.question_analysis import QuestionAnalysis, analyze_business_question
from app.analytics.kpis import calculate_kpis
from app.analytics.sales_analysis import (
    product_performance,
    revenue_by_month,
    revenue_by_region,
)


@dataclass(frozen=True)
class NaturalLanguageAnswer:
    """Python analysis output and its optional LLM explanation."""

    analysis: QuestionAnalysis
    answer: str | None


def build_business_evidence(sales: pd.DataFrame) -> dict[str, object]:
    """Build compact aggregate evidence without sending raw customer records."""
    kpis = calculate_kpis(sales)
    monthly = revenue_by_month(sales)
    products = product_performance(sales).head(5)
    regions = revenue_by_region(sales).head(5)
    return {
        "record_count": len(sales),
        "date_range": (
            {
                "start": pd.to_datetime(sales["date"]).min().date().isoformat(),
                "end": pd.to_datetime(sales["date"]).max().date().isoformat(),
            }
            if not sales.empty
            else None
        ),
        "kpis": kpis,
        "monthly_revenue": [
            {"month": row.month.date().isoformat(), "revenue": float(row.revenue)}
            for row in monthly.itertuples(index=False)
        ],
        "top_products_by_revenue": [
            {
                "product": str(row.product),
                "revenue": float(row.revenue),
                "units_sold": float(row.units_sold),
            }
            for row in products.itertuples(index=False)
        ],
        "top_regions_by_revenue": [
            {"region": str(row.region), "revenue": float(row.revenue)}
            for row in regions.itertuples(index=False)
        ],
        "limitations": [
            "Average order value is calculated per sales record because no order identifier is available.",
            "Only the listed aggregate summaries are supplied; transaction-level and customer-level details are omitted.",
        ],
    }


def answer_business_question(
    question: str,
    evidence: dict[str, object],
    *,
    client: Any | None = None,
    model: str | None = None,
) -> str:
    """Answer a nonempty question using only caller-supplied calculated evidence."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Enter a business question.")
    if not isinstance(evidence, dict):
        raise ValueError("Business evidence must be a dictionary of calculated summaries.")
    messages = build_analyst_messages(question.strip(), evidence)
    return complete_chat(messages, client=client, model=model)


def answer_natural_language_question(
    sales: pd.DataFrame,
    question: str,
    *,
    client: Any | None = None,
    model: str | None = None,
) -> NaturalLanguageAnswer:
    """Route to Python first, then ask the LLM to explain calculated evidence."""
    analysis = analyze_business_question(sales, question)
    if analysis.intent is None or analysis.message is not None:
        return NaturalLanguageAnswer(analysis=analysis, answer=None)
    evidence = {
        "analysis_type": analysis.intent,
        "calculated_results": analysis.evidence,
    }
    answer = answer_business_question(question, evidence, client=client, model=model)
    return NaturalLanguageAnswer(analysis=analysis, answer=answer)
