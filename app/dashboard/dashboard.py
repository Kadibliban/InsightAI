"""Streamlit sales dashboard for InsightAI."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from app.analytics.kpis import calculate_kpis
from app.analytics.sales_analysis import (
    filter_sales_data,
    product_performance,
    revenue_by_month,
    revenue_by_region,
)
from app.ai.analyst import answer_natural_language_question
from app.ai.errors import LLMConfigurationError, LLMServiceError
from app.data import DataValidationError, load_and_clean_sales_data
from app.ml.anomaly_detection import detect_revenue_anomalies
from app.ml.errors import MLAnalysisError
from app.ml.forecasting import forecast_monthly_revenue
from app.ml.segmentation import segment_customers
from app.rag import (
    DocumentError,
    DocumentStoreError,
    answer_document_question,
    delete_document,
    ingest_pdf,
    list_documents,
)

SAMPLE_FILE = Path(__file__).resolve().parents[2] / "data" / "sample_sales.csv"


def _format_amount(value: float) -> str:
    return f"{value:,.2f}"


def run_dashboard() -> None:
    st.set_page_config(page_title="InsightAI | Sales Dashboard", page_icon="📊", layout="wide")
    st.title("InsightAI Sales Dashboard")
    st.caption("Explore validated sales data with calculated business metrics.")

    with st.sidebar:
        st.header("Sales data")
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx", "xlsm"],
            help="Required columns: date, customer, product, category, region, quantity, unit_price, revenue.",
        )

    if uploaded_file is None:
        try:
            sales = load_and_clean_sales_data(SAMPLE_FILE)
        except DataValidationError as error:
            st.error(f"The sample dataset could not be loaded: {error}")
            st.stop()
        source_label = "Sample sales dataset"
    else:
        try:
            sales = load_and_clean_sales_data(uploaded_file)
        except DataValidationError as error:
            st.error(f"Could not use the uploaded file: {error}")
            st.stop()
        source_label = uploaded_file.name

    st.sidebar.caption(f"Data source: {source_label}")
    st.sidebar.caption(f"{len(sales):,} cleaned sales records")

    if sales.empty:
        st.info("No sales records are available to display.")
        return

    minimum_date = sales["date"].min().date()
    maximum_date = sales["date"].max().date()
    with st.sidebar:
        date_selection = st.date_input(
            "Date range",
            value=(minimum_date, maximum_date),
            min_value=minimum_date,
            max_value=maximum_date,
        )
        selected_regions = st.multiselect(
            "Regions", sorted(sales["region"].unique()), default=sorted(sales["region"].unique())
        )
        selected_products = st.multiselect(
            "Products", sorted(sales["product"].unique()), default=sorted(sales["product"].unique())
        )

    if isinstance(date_selection, (tuple, list)):
        start_date = date_selection[0] if date_selection else None
        end_date = date_selection[-1] if len(date_selection) > 1 else start_date
    else:
        start_date = end_date = date_selection

    filtered_sales = filter_sales_data(
        sales,
        start_date=start_date,
        end_date=end_date,
        regions=selected_regions,
        products=selected_products,
    )
    if filtered_sales.empty:
        st.info("No records match the selected filters. Adjust the date range, regions, or products.")
        return

    kpis = calculate_kpis(filtered_sales)
    metric_columns = st.columns(5)
    metric_columns[0].metric("Total revenue", _format_amount(kpis["total_revenue"]))
    metric_columns[1].metric("Sales transactions", f"{kpis['sales_transactions']:,}")
    metric_columns[2].metric("Average order value", _format_amount(kpis["average_order_value"]))
    metric_columns[3].metric("Customers", f"{kpis['customer_count']:,}")
    metric_columns[4].metric("Units sold", f"{kpis['units_sold']:,.0f}")
    st.caption(
        "Average order value is calculated per sales record because the source data has no order ID."
    )

    trend = revenue_by_month(filtered_sales)
    products = product_performance(filtered_sales)
    regions = revenue_by_region(filtered_sales)

    trend_column, product_column = st.columns(2)
    with trend_column:
        st.subheader("Revenue over time")
        figure = px.line(
            trend,
            x="month",
            y="revenue",
            markers=True,
            labels={"month": "Month", "revenue": "Revenue"},
            template="plotly_white",
        )
        figure.update_layout(margin=dict(l=10, r=10, t=15, b=10))
        st.plotly_chart(figure, width="stretch")

    with product_column:
        st.subheader("Product performance")
        figure = px.bar(
            products.sort_values("revenue"),
            x="revenue",
            y="product",
            orientation="h",
            labels={"product": "Product", "revenue": "Revenue"},
            template="plotly_white",
        )
        figure.update_layout(margin=dict(l=10, r=10, t=15, b=10))
        st.plotly_chart(figure, width="stretch")

    st.subheader("Revenue by region")
    figure = px.bar(
        regions,
        x="region",
        y="revenue",
        labels={"region": "Region", "revenue": "Revenue"},
        template="plotly_white",
    )
    figure.update_layout(margin=dict(l=10, r=10, t=15, b=10))
    st.plotly_chart(figure, width="stretch")

    with st.expander("Machine learning analysis"):
        forecast_tab, segment_tab, anomaly_tab = st.tabs(
            ["Sales forecast", "Customer segments", "Revenue anomalies"]
        )
        with forecast_tab:
            forecast_horizon = st.slider(
                "Months to forecast", min_value=1, max_value=6, value=3
            )
            try:
                forecast_result = forecast_monthly_revenue(
                    filtered_sales, horizon=forecast_horizon
                )
            except MLAnalysisError as error:
                st.info(f"Sales forecasting is unavailable: {error}")
            else:
                history_curve = forecast_result.history.rename(
                    columns={"revenue": "display_revenue"}
                ).assign(series="Historical")
                forecast_curve = pd.concat(
                    [
                        forecast_result.history.tail(1).rename(
                            columns={"revenue": "display_revenue"}
                        ),
                        forecast_result.forecast.rename(
                            columns={"predicted_revenue": "display_revenue"}
                        ),
                    ],
                    ignore_index=True,
                ).assign(series="Forecast")
                figure = px.line(
                    pd.concat([history_curve, forecast_curve], ignore_index=True),
                    x="month",
                    y="display_revenue",
                    color="series",
                    markers=True,
                    labels={"month": "Month", "display_revenue": "Revenue", "series": ""},
                    color_discrete_map={"Historical": "#2563EB", "Forecast": "#F59E0B"},
                    template="plotly_white",
                )
                figure.update_layout(margin=dict(l=10, r=10, t=15, b=10))
                st.plotly_chart(figure, width="stretch")
                evaluation_columns = st.columns(3)
                evaluation_columns[0].metric(
                    "Holdout MAE", _format_amount(forecast_result.evaluation.mae)
                )
                evaluation_columns[1].metric(
                    "Holdout RMSE", _format_amount(forecast_result.evaluation.rmse)
                )
                evaluation_columns[2].metric(
                    "Holdout months", forecast_result.evaluation.holdout_count
                )
                st.caption(
                    "Forecast uses a linear monthly trend. Holdout errors evaluate recent history; they do not guarantee future accuracy."
                )

        with segment_tab:
            customer_count = int(filtered_sales["customer"].nunique())
            if customer_count < 2:
                st.info("At least two customers are needed to create segments.")
            else:
                max_clusters = min(6, customer_count)
                cluster_count = st.slider(
                    "Number of customer segments",
                    min_value=2,
                    max_value=max_clusters,
                    value=min(3, max_clusters),
                )
                try:
                    segmentation = segment_customers(
                        filtered_sales, n_clusters=cluster_count
                    )
                    st.dataframe(
                        segmentation.segments,
                        width="content",
                        hide_index=True,
                    )
                    st.caption("Customers are grouped using standardized Recency, Frequency, and Monetary values.")
                    st.dataframe(
                        segmentation.customers,
                        width="stretch",
                        hide_index=True,
                    )
                except MLAnalysisError as error:
                    st.info(f"Customer segmentation is unavailable: {error}")

        with anomaly_tab:
            try:
                anomalies = detect_revenue_anomalies(filtered_sales)
            except MLAnalysisError as error:
                st.info(f"Anomaly detection is unavailable: {error}")
            else:
                if anomalies.empty:
                    st.success("No transactions fell outside the IQR anomaly bounds.")
                else:
                    st.dataframe(anomalies, width="content", hide_index=True)
                    st.caption(
                        "Transactions are flagged when revenue falls outside Q1 − 1.5×IQR or Q3 + 1.5×IQR."
                    )

    with st.expander("Ask the AI business analyst"):
        st.caption(
            "Questions are routed to predefined Python analyses using the current filters. "
            "Customer labels are sent only for customer-ranking questions. Verify important decisions against the dashboard."
        )
        question = st.text_area(
            "Business question",
            placeholder="Which month had the highest revenue, and what were the top products?",
            key="business_analyst_question",
        )
        if st.button("Ask analyst", type="primary", key="ask_business_analyst"):
            if not question.strip():
                st.info("Enter a business question first.")
            else:
                try:
                    with st.spinner("Calculating results and preparing an answer…"):
                        response = answer_natural_language_question(filtered_sales, question)
                except LLMConfigurationError as error:
                    st.error(str(error))
                except LLMServiceError as error:
                    st.error(str(error))
                else:
                    if response.answer is None:
                        st.info(response.analysis.message or "No supported analysis was selected.")
                    else:
                        st.markdown(response.answer)
                        with st.expander("Python-calculated evidence"):
                            st.json(
                                {
                                    "analysis_type": response.analysis.intent,
                                    "results": response.analysis.evidence,
                                }
                            )

    with st.expander("Ask questions about PDF documents"):
        st.caption(
            "Upload text-based PDFs, then ask questions answered from retrieved excerpts. "
            "Answers cite the source file and page. Scanned image PDFs need OCR first."
        )
        pdf_file = st.file_uploader(
            "Upload a PDF document",
            type=["pdf"],
            key="rag_pdf_upload",
            help="Maximum 15 MiB and 500 pages. Documents are stored locally in the project folder.",
        )
        if st.button("Index PDF", key="index_rag_pdf", disabled=pdf_file is None):
            try:
                with st.spinner("Extracting and indexing PDF text…"):
                    indexed = ingest_pdf(pdf_file.name, pdf_file.getvalue())
            except (DocumentError, DocumentStoreError) as error:
                st.error(str(error))
            else:
                st.success(
                    f"Indexed {indexed['filename']}: {indexed['page_count']} pages, "
                    f"{indexed['chunk_count']} text chunks."
                )

        try:
            documents = list_documents()
        except DocumentStoreError as error:
            st.error(str(error))
            documents = []
        if documents:
            document_options = {"All uploaded PDFs": None}
            document_options.update(
                {
                    f"{item['filename']} · {item['document_id'][:8]}": item["document_id"]
                    for item in documents
                }
            )
            selected_label = st.selectbox(
                "Search in",
                list(document_options),
                key="rag_document_selection",
            )
            question = st.text_area(
                "Document question",
                placeholder="What does the report say about last quarter's risks?",
                key="rag_document_question",
            )
            if st.button("Ask about documents", key="ask_rag_documents", type="primary"):
                if not question.strip():
                    st.info("Enter a question first.")
                else:
                    try:
                        with st.spinner("Searching documents and preparing an answer…"):
                            response = answer_document_question(
                                question,
                                document_id=document_options[selected_label],
                            )
                    except (DocumentError, DocumentStoreError, LLMConfigurationError, LLMServiceError) as error:
                        st.error(str(error))
                    else:
                        st.markdown(response["answer"])
                        if response["sources"]:
                            st.caption("Retrieved sources")
                            st.dataframe(
                                pd.DataFrame(response["sources"]),
                                width="stretch",
                                hide_index=True,
                            )
            with st.expander("Manage indexed PDFs"):
                for item in documents:
                    columns = st.columns([5, 1])
                    columns[0].write(
                        f"**{item['filename']}** · {item['page_count']} pages · "
                        f"{item['chunk_count']} chunks"
                    )
                    if columns[1].button("Delete", key=f"delete_document_{item['document_id']}"):
                        try:
                            delete_document(item["document_id"])
                        except DocumentStoreError as error:
                            st.error(str(error))
                        else:
                            st.rerun()
        else:
            st.info("Upload and index a PDF to start document Q&A.")

    with st.expander("View filtered sales records"):
        st.dataframe(
            filtered_sales.sort_values("date", ascending=False),
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    run_dashboard()
