from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data import load_and_clean_sales_data
from app.ml.forecasting import forecast_monthly_revenue


def sample_sales() -> pd.DataFrame:
    sample = Path(__file__).parents[1] / "data" / "sample_sales.csv"
    return load_and_clean_sales_data(sample)


def test_forecast_aggregates_months_evaluates_holdout_and_predicts_horizon() -> None:
    result = forecast_monthly_revenue(sample_sales(), horizon=3)

    assert list(result.history.columns) == ["month", "revenue"]
    assert len(result.history) == 6
    assert result.history["revenue"].sum() == 654300
    assert result.evaluation.holdout_count == 2
    assert np.isfinite([result.evaluation.mae, result.evaluation.rmse]).all()
    assert len(result.holdout) == 2
    assert list(result.holdout.columns) == ["month", "actual_revenue", "predicted_revenue"]
    assert list(result.forecast.columns) == ["month", "predicted_revenue"]
    assert list(result.forecast["month"].dt.strftime("%Y-%m")) == [
        "2025-07", "2025-08", "2025-09"
    ]


def test_linear_trend_is_explainable_and_fills_calendar_month_gaps() -> None:
    sales = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-03-05", "2025-04-10"]),
            "revenue": [10.0, 30.0, 40.0],
        }
    )

    result = forecast_monthly_revenue(sales, horizon=2)

    assert result.history["revenue"].tolist() == [10.0, 0.0, 30.0, 40.0]
    assert result.forecast["predicted_revenue"].tolist() == pytest.approx([50.0, 62.0])
    assert result.evaluation.holdout_count == 1


@pytest.mark.parametrize(
    ("sales", "message"),
    [
        (pd.DataFrame(), "missing required columns"),
        (pd.DataFrame({"date": ["2025-01-01"], "revenue": [1]}), "three calendar months"),
        (
            pd.DataFrame(
                {"date": ["2025-01-01", "bad", "2025-03-01"], "revenue": [1, 2, 3]}
            ),
            "valid and non-missing",
        ),
        (
            pd.DataFrame(
                {"date": ["2025-01-01", "2025-02-01", "2025-03-01"], "revenue": [1, np.inf, 3]}
            ),
            "finite numeric",
        ),
    ],
)
def test_invalid_or_insufficient_sales_raise_clear_value_error(
    sales: pd.DataFrame, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        forecast_monthly_revenue(sales)


@pytest.mark.parametrize("horizon", [0, -1, 1.5, True, "2"])
def test_horizon_must_be_a_positive_integer(horizon: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        forecast_monthly_revenue(sample_sales(), horizon=horizon)  # type: ignore[arg-type]
