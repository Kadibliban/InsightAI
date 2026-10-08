import pandas as pd
import pytest

from app.ml.anomaly_detection import detect_revenue_anomalies
from app.ml.errors import MLAnalysisError


def test_iqr_detector_flags_high_and_low_transactions_with_bounds() -> None:
    sales = pd.DataFrame(
        {
            "customer": list("ABCDEFG"),
            "revenue": [10, 11, 12, 10, 13, 100, -100],
        }
    )

    anomalies = detect_revenue_anomalies(sales)

    assert set(anomalies["revenue"]) == {100, -100}
    assert set(anomalies["anomaly_direction"]) == {"High", "Low"}
    assert (anomalies["lower_bound"] < anomalies["upper_bound"]).all()


def test_iqr_detector_returns_empty_table_when_no_outliers() -> None:
    sales = pd.DataFrame({"revenue": [10, 11, 12, 13, 14, 15]})

    anomalies = detect_revenue_anomalies(sales)

    assert anomalies.empty
    assert {"anomaly_direction", "lower_bound", "upper_bound"}.issubset(anomalies.columns)


def test_anomaly_detection_reports_insufficient_data() -> None:
    with pytest.raises(MLAnalysisError, match="at least 4 sales records"):
        detect_revenue_anomalies(pd.DataFrame({"revenue": [10, 12, 100]}))


def test_anomaly_detection_validates_multiplier_and_column() -> None:
    with pytest.raises(MLAnalysisError, match="multiplier"):
        detect_revenue_anomalies(pd.DataFrame({"revenue": [1, 2, 3, 4]}), iqr_multiplier=0)
    with pytest.raises(MLAnalysisError, match="requires a 'revenue' column"):
        detect_revenue_anomalies(pd.DataFrame({"sales": [1, 2, 3, 4]}))
