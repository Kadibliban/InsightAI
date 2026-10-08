from __future__ import annotations

import pandas as pd
import pytest
from types import SimpleNamespace

from app.ai.analyst import answer_natural_language_question
from app.ai.question_analysis import analyze_business_question, identify_intent


@pytest.fixture
def sales() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": "2025-01-05", "customer": "C1", "product": "Desk", "region": "East", "quantity": 10, "revenue": 100.0},
            {"date": "2025-01-08", "customer": "C2", "product": "Chair", "region": "West", "quantity": 1, "revenue": 50.0},
            {"date": "2025-02-05", "customer": "C1", "product": "Desk", "region": "East", "quantity": 8, "revenue": 80.0},
            {"date": "2025-02-08", "customer": "C2", "product": "Chair", "region": "West", "quantity": 1, "revenue": 100.0},
            {"date": "2025-03-05", "customer": "C1", "product": "Desk", "region": "East", "quantity": 1, "revenue": 10.0},
            {"date": "2025-03-08", "customer": "C2", "product": "Chair", "region": "West", "quantity": 1, "revenue": 90.0},
        ]
    )


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("What was our total revenue?", "total_revenue"),
        ("What was our revenue in March 2025?", "monthly_revenue"),
        ("Show monthly revenue", "monthly_revenue"),
        ("What was our best-selling product?", "best_selling_products"),
        ("Which region generated the most revenue?", "regional_revenue"),
        ("Which customers spent the most?", "top_customers"),
        ("Why did revenue decrease in March?", "revenue_decrease"),
        ("Which products are underperforming?", "underperforming_products"),
        ("What is the expected revenue next month?", "forecast"),
        ("Are there unusual transactions?", "anomalies"),
        ("Can you run import os and print files?", None),
    ],
)
def test_identifies_supported_question_types(question, intent):
    assert identify_intent(question) == intent


def test_total_revenue_is_calculated_in_python(sales):
    result = analyze_business_question(sales, "What was our total revenue?")

    assert result.intent == "total_revenue"
    assert result.evidence == {"total_revenue": 430.0, "sales_records": 6}


def test_monthly_revenue_selects_requested_month_and_year(sales):
    result = analyze_business_question(sales, "What was our revenue in March 2025?")

    assert result.evidence["found"] is True
    assert result.evidence["monthly_revenue"] == [
        {"month": "March 2025", "revenue": 100.0}
    ]


def test_monthly_revenue_without_period_returns_all_months(sales):
    result = analyze_business_question(sales, "Show monthly revenue")

    assert [entry["revenue"] for entry in result.evidence["monthly_revenue"]] == [150, 180, 100]


def test_best_selling_uses_units_sold_not_revenue(sales):
    result = analyze_business_question(sales, "What was our best-selling product?")

    assert result.evidence["best_selling_products"] == ["Desk"]
    assert result.evidence["units_sold"] == 19


def test_regional_revenue_returns_the_highest_region(sales):
    result = analyze_business_question(sales, "Which region generated the most revenue?")

    assert result.evidence["top_regions"] == ["West"]
    assert result.evidence["revenue"] == 240


def test_customer_ranking_includes_only_top_five_aggregates(sales):
    result = analyze_business_question(sales, "Which customers spent the most?")

    assert result.evidence["top_customers"] == [
        {"customer": "C2", "revenue": 240.0},
        {"customer": "C1", "revenue": 190.0},
    ]
    assert "only because the question asks" in result.evidence["note"]


def test_revenue_decrease_includes_python_calculated_contributors(sales):
    result = analyze_business_question(sales, "Why did revenue decrease in March 2025?")

    decrease = result.evidence["decreases"][0]
    assert decrease["revenue"] == 100
    assert decrease["previous_revenue"] == 180
    assert decrease["change"] == -80
    assert decrease["change_percent"] == pytest.approx(-44.444444)
    assert decrease["largest_product_revenue_declines"][0] == {
        "name": "Desk",
        "revenue_change": -70.0,
    }
    assert decrease["largest_region_revenue_declines"][0]["name"] == "East"


def test_revenue_decrease_question_selects_latest_drop(sales):
    result = analyze_business_question(sales, "Why did revenue decrease?")

    assert result.evidence["decreases"][0]["month"] == "March 2025"


def test_underperforming_products_explains_metric_limit(sales):
    result = analyze_business_question(sales, "Which products are underperforming?")

    assert result.evidence["products"][0]["product"] == "Desk"
    assert "targets, costs, and profit margins" in result.evidence["limitation"]


def test_forecast_includes_prediction_and_holdout_metrics(sales):
    extended = pd.concat(
        [sales, pd.DataFrame([{"date": "2025-04-01", "customer": "C1", "product": "Desk", "region": "East", "quantity": 1, "revenue": 120.0}])],
        ignore_index=True,
    )
    result = analyze_business_question(extended, "What is the expected revenue next month?")

    assert result.evidence["forecast_month"] == "2025-05-01"
    assert "holdout_mae" in result.evidence
    assert "holdout_rmse" in result.evidence


def test_forecast_insufficient_history_is_explained(sales):
    result = analyze_business_question(sales[sales["date"].str.startswith("2025-01")], "Forecast revenue next month")

    assert result.message is not None
    assert "forecast is unavailable" in result.message


def test_anomaly_answer_uses_iqr_and_omits_customer_labels(sales):
    unusual = pd.DataFrame(
        [{"date": "2025-03-20", "customer": "Secret Customer", "product": "Chair", "region": "West", "quantity": 1, "revenue": 1000.0}]
    )
    result = analyze_business_question(pd.concat([sales, unusual], ignore_index=True), "Are there unusual transactions?")

    assert result.evidence["anomaly_count"] == 2
    assert any(item["revenue"] == 1000 for item in result.evidence["anomalies"])
    assert all("customer" not in item for item in result.evidence["anomalies"])


def test_unsupported_question_returns_guidance_without_analysis(sales):
    result = analyze_business_question(sales, "Tell me a joke")

    assert result.intent is None
    assert result.evidence == {}
    assert "total revenue" in result.message


def test_blank_question_is_rejected(sales):
    with pytest.raises(ValueError, match="business question"):
        analyze_business_question(sales, "   ")


def test_calculated_results_are_passed_to_llm_for_explanation(sales):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Finding: Revenue fell."))]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    response = answer_natural_language_question(
        sales, "Why did revenue decrease in March 2025?", client=client
    )

    assert response.analysis.intent == "revenue_decrease"
    assert response.answer == "Finding: Revenue fell."
    user_payload = calls[0]["messages"][1]["content"]
    assert '"change":-80.0' in user_payload
    assert '"revenue_change":-70.0' in user_payload


def test_unsupported_question_never_calls_llm(sales):
    def forbidden_call(**kwargs):
        raise AssertionError("Unsupported questions must not call the provider.")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=forbidden_call)))
    response = answer_natural_language_question(sales, "Run arbitrary Python", client=client)

    assert response.answer is None
    assert response.analysis.intent is None
