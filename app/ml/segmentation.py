"""RFM feature calculation and explainable customer segmentation."""

from dataclasses import dataclass
from datetime import date
import math

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from app.ml.errors import MLAnalysisError

RFM_COLUMNS = ("recency_days", "frequency", "monetary")


@dataclass
class SegmentationResult:
    customers: pd.DataFrame
    segments: pd.DataFrame


def calculate_rfm_features(
    sales: pd.DataFrame,
    *,
    snapshot_date: date | None = None,
) -> pd.DataFrame:
    """Create one Recency/Frequency/Monetary row for each customer.

    Frequency counts sales records because the current schema has no order ID.
    The default snapshot is the day after the latest observed transaction.
    """
    if not isinstance(sales, pd.DataFrame):
        raise MLAnalysisError("Sales data must be provided as a pandas DataFrame.")
    required = {"customer", "date", "revenue"}
    missing = sorted(required - set(sales.columns))
    if missing:
        raise MLAnalysisError(f"RFM analysis requires columns: {', '.join(missing)}.")
    if sales.empty:
        raise MLAnalysisError("RFM analysis requires at least one sales record.")

    dates = pd.to_datetime(sales["date"], errors="coerce")
    revenue = pd.to_numeric(sales["revenue"], errors="coerce")
    finite_revenue = revenue.dropna().to_numpy(dtype=float)
    invalid = dates.isna() | revenue.isna() | sales["customer"].isna()
    if not all(math.isfinite(value) for value in finite_revenue):
        invalid |= ~revenue.map(lambda value: math.isfinite(value) if pd.notna(value) else True)
    if invalid.any():
        rows = [str(position + 2) for position, is_invalid in enumerate(invalid) if is_invalid][:5]
        raise MLAnalysisError(f"RFM analysis found missing or invalid values near data rows: {', '.join(rows)}.")

    customer_sales = pd.DataFrame(
        {"customer": sales["customer"].astype(str), "date": dates, "revenue": revenue}
    )
    snapshot = (
        pd.Timestamp(snapshot_date)
        if snapshot_date is not None
        else dates.max().normalize() + pd.Timedelta(days=1)
    )
    if snapshot < dates.max().normalize():
        raise MLAnalysisError("The RFM snapshot date cannot be earlier than the latest sales date.")
    features = customer_sales.groupby("customer", as_index=False).agg(
        last_purchase=("date", "max"),
        frequency=("date", "size"),
        monetary=("revenue", "sum"),
    )
    features["recency_days"] = (snapshot - features["last_purchase"]).dt.days
    return features[["customer", *RFM_COLUMNS]]


def segment_customers(
    sales: pd.DataFrame,
    *,
    n_clusters: int = 3,
    snapshot_date: date | None = None,
) -> SegmentationResult:
    """Scale RFM features, cluster customers, and summarize actual profiles."""
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, int) or n_clusters < 2:
        raise MLAnalysisError("Customer segmentation requires at least 2 clusters.")

    rfm = calculate_rfm_features(sales, snapshot_date=snapshot_date)
    if len(rfm) < n_clusters:
        raise MLAnalysisError(
            f"Need at least {n_clusters} customers for {n_clusters} clusters; found {len(rfm)}."
        )
    distinct_profiles = rfm[list(RFM_COLUMNS)].drop_duplicates().shape[0]
    if distinct_profiles < n_clusters:
        raise MLAnalysisError(
            f"Only {distinct_profiles} distinct RFM profiles are available; reduce the cluster count."
        )

    scaled = StandardScaler().fit_transform(rfm[list(RFM_COLUMNS)])
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    customers = rfm.copy()
    customers["cluster_id"] = model.fit_predict(scaled)

    profiles = customers.groupby("cluster_id", as_index=False).agg(
        customer_count=("customer", "size"),
        average_recency_days=("recency_days", "mean"),
        average_frequency=("frequency", "mean"),
        average_monetary=("monetary", "mean"),
    )
    profiles = profiles.sort_values(
        ["average_monetary", "average_frequency", "average_recency_days"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    profiles["segment"] = [f"Segment {index + 1}" for index in range(len(profiles))]
    profiles["interpretation"] = profiles.apply(
        lambda row: (
            f"Average spend {row['average_monetary']:,.2f}; "
            f"{row['average_frequency']:.1f} sales records per customer; "
            f"last purchase {row['average_recency_days']:.0f} days ago on average."
        ),
        axis=1,
    )
    labels = profiles.set_index("cluster_id")["segment"]
    customers["segment"] = customers["cluster_id"].map(labels)
    customers = customers.sort_values(["segment", "customer"]).reset_index(drop=True)
    profiles = profiles[
        [
            "segment",
            "customer_count",
            "average_recency_days",
            "average_frequency",
            "average_monetary",
            "interpretation",
        ]
    ]
    return SegmentationResult(customers=customers, segments=profiles)
