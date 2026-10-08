"""Explainable monthly revenue forecasts using a linear time trend."""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.ml.errors import MLAnalysisError


@dataclass(frozen=True)
class ForecastEvaluation:
    """Chronological holdout metrics for the monthly trend model."""

    mae: float
    rmse: float
    holdout_count: int


@dataclass(frozen=True)
class MonthlyRevenueForecast:
    """Monthly history, holdout results, and future forecast output."""

    history: pd.DataFrame
    evaluation: ForecastEvaluation
    holdout: pd.DataFrame
    forecast: pd.DataFrame


def _monthly_revenue(sales: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(sales, pd.DataFrame):
        raise MLAnalysisError("Sales data must be provided as a pandas DataFrame.")
    missing = {"date", "revenue"}.difference(sales.columns)
    if missing:
        raise MLAnalysisError(
            "Sales data is missing required columns: " + ", ".join(sorted(missing)) + "."
        )
    if sales.empty:
        raise MLAnalysisError("Sales data contains no records to forecast.")

    dates = pd.to_datetime(sales["date"], errors="coerce", format="mixed")
    if dates.isna().any():
        raise MLAnalysisError("Sales dates must all be valid and non-missing.")
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)

    revenue = pd.to_numeric(sales["revenue"], errors="coerce")
    numeric_revenue = revenue.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(numeric_revenue).all():
        raise MLAnalysisError("Sales revenue must contain only finite numeric values.")

    monthly = pd.DataFrame({"month": dates.dt.to_period("M"), "revenue": revenue})
    monthly = monthly.groupby("month", as_index=False)["revenue"].sum()
    full_months = pd.period_range(monthly["month"].min(), monthly["month"].max(), freq="M")
    monthly = (
        monthly.set_index("month")
        .reindex(full_months, fill_value=0.0)
        .rename_axis("month")
        .reset_index()
    )
    monthly["month"] = monthly["month"].dt.to_timestamp()
    monthly["revenue"] = monthly["revenue"].astype(float)
    if len(monthly) < 3:
        raise MLAnalysisError(
            "At least three calendar months of history are required for holdout evaluation."
        )
    return monthly


def forecast_monthly_revenue(
    sales: pd.DataFrame,
    *,
    horizon: int = 3,
) -> MonthlyRevenueForecast:
    """Evaluate and forecast monthly sales revenue with a linear time trend.

    Sales are summed by calendar month. Months without records within the date
    range are treated as zero revenue. The final 20% of months (at least one,
    leaving at least two training months) form a chronological holdout. After
    reporting holdout MAE and RMSE, the model is refit on all history to predict
    the requested future months. Predictions are left unbounded to preserve the
    fitted linear trend, including when the input contains refunds.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise MLAnalysisError("Forecast horizon must be a positive integer number of months.")

    history = _monthly_revenue(sales)
    values = history["revenue"].to_numpy(dtype=float)
    month_count = len(values)
    holdout_count = min(max(1, math.ceil(month_count * 0.2)), month_count - 2)
    split_at = month_count - holdout_count
    timeline = np.arange(month_count, dtype=float).reshape(-1, 1)

    evaluation_model = LinearRegression().fit(timeline[:split_at], values[:split_at])
    holdout_predictions = evaluation_model.predict(timeline[split_at:])
    actual_holdout = values[split_at:]
    holdout = history.iloc[split_at:][["month"]].copy().reset_index(drop=True)
    holdout["actual_revenue"] = actual_holdout
    holdout["predicted_revenue"] = holdout_predictions
    evaluation = ForecastEvaluation(
        mae=float(mean_absolute_error(actual_holdout, holdout_predictions)),
        rmse=float(np.sqrt(mean_squared_error(actual_holdout, holdout_predictions))),
        holdout_count=holdout_count,
    )

    final_model = LinearRegression().fit(timeline, values)
    future_timeline = np.arange(month_count, month_count + horizon, dtype=float)
    future_months = pd.period_range(
        history["month"].iloc[-1].to_period("M") + 1,
        periods=horizon,
        freq="M",
    ).to_timestamp()
    forecast = pd.DataFrame(
        {
            "month": future_months,
            "predicted_revenue": final_model.predict(future_timeline.reshape(-1, 1)),
        }
    )
    return MonthlyRevenueForecast(
        history=history,
        evaluation=evaluation,
        holdout=holdout,
        forecast=forecast,
    )
