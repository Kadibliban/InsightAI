from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.ai import llm
from app.ai.analyst import answer_business_question, build_business_evidence
from app.ai.errors import LLMConfigurationError, LLMServiceError
from app.data import load_and_clean_sales_data


class FakeClient:
    def __init__(self, content: str = "Revenue was highest in March.", error=None):
        self.content = content
        self.error = error
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


def test_evidence_is_aggregated_and_matches_cleaned_data():
    sales = load_and_clean_sales_data("data/sample_sales.csv")

    evidence = build_business_evidence(sales)

    assert evidence["record_count"] == len(sales)
    assert evidence["kpis"]["total_revenue"] == pytest.approx(sales["revenue"].sum())
    assert len(evidence["monthly_revenue"]) > 0
    assert len(evidence["top_products_by_revenue"]) <= 5
    assert "customer" not in evidence
    assert "customer_name" not in evidence


def test_answer_passes_evidence_and_model_to_groq_client():
    client = FakeClient()
    evidence = {"kpis": {"total_revenue": 100.0}}

    answer = answer_business_question(
        "What is the revenue?", evidence, client=client, model="test-model"
    )

    assert answer == "Revenue was highest in March."
    assert client.calls[0]["model"] == "test-model"
    assert '"total_revenue":100.0' in client.calls[0]["messages"][1]["content"]


@pytest.mark.parametrize("question", ["", "  ", None])
def test_answer_rejects_empty_question(question):
    with pytest.raises(ValueError, match="business question"):
        answer_business_question(question, {}, client=FakeClient())


def test_answer_rejects_non_dictionary_evidence():
    with pytest.raises(ValueError, match="dictionary"):
        answer_business_question("Question?", [], client=FakeClient())


def test_missing_key_is_a_safe_configuration_error(monkeypatch):
    monkeypatch.setattr(llm, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(LLMConfigurationError, match="GROQ_API_KEY"):
        llm.create_groq_client()


def test_default_model_is_selected_without_configuration(monkeypatch):
    monkeypatch.setattr(llm, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    assert llm.configured_model() == "openai/gpt-oss-20b"


def test_provider_failure_is_sanitized():
    client = FakeClient(error=RuntimeError("secret-looking-provider-payload"))

    with pytest.raises(LLMServiceError) as error:
        answer_business_question("Question?", {}, client=client)

    assert "secret-looking-provider-payload" not in str(error.value)


def test_empty_provider_response_is_reported():
    with pytest.raises(LLMServiceError, match="empty answer"):
        answer_business_question("Question?", {}, client=FakeClient("  "))
