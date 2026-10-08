from pathlib import Path

import pandas as pd
import pytest

from app.data import load_and_clean_sales_data
from app.ml.errors import MLAnalysisError
from app.ml.segmentation import calculate_rfm_features, segment_customers


def sample_sales() -> pd.DataFrame:
    sample = Path(__file__).parents[1] / "data" / "sample_sales.csv"
    return load_and_clean_sales_data(sample)


def test_rfm_features_are_calculated_per_customer() -> None:
    sales = sample_sales()

    rfm = calculate_rfm_features(sales)

    assert len(rfm) == 8
    assert set(rfm.columns) == {"customer", "recency_days", "frequency", "monetary"}
    assert (rfm["recency_days"] >= 0).all()
    assert rfm["frequency"].sum() == len(sales)
    assert rfm["monetary"].sum() == sales["revenue"].sum()


def test_segments_are_interpretable_and_cover_all_customers() -> None:
    sales = sample_sales()

    result = segment_customers(sales, n_clusters=3)

    assert len(result.customers) == 8
    assert result.segments["customer_count"].sum() == 8
    assert result.segments["segment"].nunique() == 3
    assert result.segments["interpretation"].str.contains("Average spend").all()
    assert result.customers["segment"].notna().all()


def test_segmentation_is_reproducible() -> None:
    sales = sample_sales()

    first = segment_customers(sales, n_clusters=3)
    second = segment_customers(sales, n_clusters=3)

    pd.testing.assert_frame_equal(first.customers, second.customers)


def test_segmentation_rejects_impossible_cluster_counts() -> None:
    with pytest.raises(MLAnalysisError, match="at least 2 clusters"):
        segment_customers(sample_sales(), n_clusters=1)
    with pytest.raises(MLAnalysisError, match="Need at least"):
        segment_customers(sample_sales(), n_clusters=9)


def test_segmentation_reports_identical_customer_profiles() -> None:
    sales = pd.DataFrame(
        {
            "customer": ["A", "B"],
            "date": ["2025-01-01", "2025-01-01"],
            "revenue": [100, 100],
        }
    )

    with pytest.raises(MLAnalysisError, match="distinct RFM profiles"):
        segment_customers(sales, n_clusters=2)
