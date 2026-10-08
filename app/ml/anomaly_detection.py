"""Explainable revenue anomaly detection using the interquartile range."""

import math

import pandas as pd

from app.ml.errors import MLAnalysisError


def detect_revenue_anomalies(
    sales: pd.DataFrame,
    *,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Return transactions outside Tukey IQR bounds with thresholds attached."""
    if "revenue" not in sales.columns:
        raise MLAnalysisError("Anomaly detection requires a 'revenue' column.")
    try:
        valid_multiplier = math.isfinite(iqr_multiplier) and iqr_multiplier > 0
    except (TypeError, ValueError):
        valid_multiplier = False
    if not valid_multiplier:
        raise MLAnalysisError("The IQR multiplier must be greater than zero.")
    if len(sales) < 4:
        raise MLAnalysisError("Anomaly detection needs at least 4 sales records.")

    revenue = pd.to_numeric(sales["revenue"], errors="coerce")
    values = revenue.to_numpy(dtype=float, na_value=float("nan"))
    if revenue.isna().any() or not all(math.isfinite(value) for value in values):
        raise MLAnalysisError("Revenue must contain valid numeric values for anomaly detection.")

    first_quartile = revenue.quantile(0.25)
    third_quartile = revenue.quantile(0.75)
    iqr = third_quartile - first_quartile
    lower_bound = float(first_quartile - iqr_multiplier * iqr)
    upper_bound = float(third_quartile + iqr_multiplier * iqr)
    mask = (revenue < lower_bound) | (revenue > upper_bound)

    anomalies = sales.loc[mask].copy()
    anomalies["anomaly_direction"] = anomalies["revenue"].map(
        lambda value: "Low" if value < lower_bound else "High"
    )
    anomalies["lower_bound"] = lower_bound
    anomalies["upper_bound"] = upper_bound
    return anomalies.sort_values("revenue", ascending=False).reset_index(drop=True)
