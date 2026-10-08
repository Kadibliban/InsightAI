# InsightAI — AI Business Intelligence Assistant

InsightAI is a business intelligence app for exploring sales data through a dashboard, explainable machine-learning analyses, natural-language questions, and document search. Python calculates the business metrics; the language model explains those results instead of calculating or inventing them.

## Problem

Sales teams often need to combine spreadsheet cleanup, KPI reporting, trend analysis, and document lookup. InsightAI brings those tasks into one small app and keeps answers tied to calculated sales evidence or cited document excerpts.

## Features

- Upload and validate sales data from CSV and modern Excel files.
- Explore KPIs, monthly revenue, product performance, and regional results in Streamlit.
- Forecast monthly revenue, segment customers with RFM features, and flag unusual transactions.
- Ask supported business questions and inspect the Python-calculated evidence behind each response.
- Upload text-based PDFs and ask questions with source filename and page citations.
- Persist sales datasets in PostgreSQL or local SQLite; persist the PDF index in SQLite.

## Architecture

```mermaid
flowchart TD
    U[Business user] --> UI[Streamlit dashboard]
    U --> API[FastAPI endpoints]
    UI --> PY[Python analytics and ML]
    UI --> RAG[PDF extraction and retrieval]
    API --> VALIDATE[Sales validation and cleaning]
    VALIDATE --> PG[(PostgreSQL or SQLite sales data)]
    UI --> PG
    RAG --> DOC[(SQLite document index)]
    PY --> EVIDENCE[Calculated evidence]
    RAG --> EXCERPTS[Retrieved page excerpts]
    EVIDENCE --> LLM[Groq language model explains results]
    EXCERPTS --> LLM
    LLM --> UI
```

The dashboard and API are separate entry points. The dashboard calls the Python analysis and document modules directly; FastAPI exposes upload, persistence, and analysis routes for API clients. In Docker Compose, both connect to the same PostgreSQL service, while the document index uses a persistent SQLite volume.

## Technology stack

| Area | Tools |
| --- | --- |
| Dashboard | Streamlit, Plotly |
| API | FastAPI, Pydantic, Uvicorn |
| Data processing | Pandas, NumPy |
| Machine learning | scikit-learn |
| Persistence | SQLAlchemy, PostgreSQL, SQLite |
| Language model | Groq API |
| Document retrieval | pypdf, scikit-learn hashing vectorizer |
| Packaging and checks | Docker Compose, pytest |

## How it works

1. A user uploads a sales CSV or Excel file.
2. The ingestion pipeline normalizes headers, validates required values, parses dates and numbers, and removes exact duplicate rows.
3. The cleaned records are stored in the configured database and summarized in the dashboard.
4. Python computes supported KPI, trend, forecasting, segmentation, and anomaly results from the selected data.
5. For supported AI questions, the app sends calculated evidence to Groq for a plain-language explanation. For PDF questions, it retrieves relevant page excerpts and asks Groq to answer with citations.

## Engineering decisions

- Business numbers are calculated in Python before calling the language model; the model receives evidence to explain.
- Supported questions map to predefined analyses. The model cannot write or execute SQL or Python.
- Forecasting uses a small linear time-trend model so its assumptions and holdout error can be inspected.
- Document vectors use deterministic local hashing, avoiding a model download or separate embedding service. Retrieval is lexical, so wording that differs substantially from the source may match less well.
- PostgreSQL is used for shared sales persistence in Compose; SQLite keeps local setup simple and stores the document index.

## Database design

The SQLAlchemy schema has two sales tables: `datasets` stores upload metadata, and `sales_records` stores the cleaned rows linked to a dataset. Deleting a dataset also removes its sales rows. PostgreSQL is configured for the Docker Compose stack; local runs can use SQLite. PDF metadata, extracted page chunks, and hashed vectors are stored separately in the SQLite document index.

## Screenshots

No verified dashboard screenshot is included yet. Add screenshots here after capturing the running dashboard with representative sample data; the hosted deployment and its appearance still need a live check.

## Current status

The core application, Docker Compose setup, and Render dependency declaration are in the repository. The latest Render deployment and its live features still need verification. See [DEPLOYMENT.md](DEPLOYMENT.md) for the deployment setup and its access limitations.

## Requirements

- Python 3.12 or newer
- Git

## Setup

From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

The `.env` file is for local credentials and configuration. Keep it private; `.gitignore` excludes it. Add a Groq API key to enable the optional AI analyst and document Q&A:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

The model setting is optional and defaults to `openai/gpt-oss-20b`. Dashboard features other than the AI analyst work without an API key. The key is read from the process environment or the project `.env` file and is never displayed in the dashboard.

`DATABASE_URL` selects the database. The example file uses `sqlite:///./insightai.db` as a safe local default. To use PostgreSQL, set a private URL such as `postgresql+psycopg2://USER:PASSWORD@localhost:5432/insightai` in `.env`; replace both placeholders with your local database credentials. The SQLite file is ignored by Git. PostgreSQL tables are created automatically when the API first accesses the database.

## Load and clean sales data

Expected columns are `date`, `customer`, `product`, `category`, `region`, `quantity`, `unit_price`, and `revenue`. Headers are normalized to lowercase with underscores. Blank categories become `Uncategorized`; blank revenue is derived from quantity × unit price. Exact duplicate records are removed. Missing required values, invalid dates or numbers, nonpositive quantities, and negative unit prices produce readable validation errors. Revenue can be negative for refunds.

```python
from app.data import load_and_clean_sales_data

sales = load_and_clean_sales_data("data/sample_sales.csv")
print(sales.head())
```

CSV, `.xlsx`, and `.xlsm` are supported. Legacy `.xls` files should be saved as `.xlsx` first.

## Run the dashboard

```powershell
python -m streamlit run app/dashboard/dashboard.py
```

The dashboard shows total revenue, sales transactions, average order value, unique customers, units sold, monthly revenue, product performance, and revenue by region. Since the current data schema has no order identifier, average order value is total revenue divided by the number of sales records. The **Ask questions about PDF documents** section stores indexed document text and vectors locally in `data/rag_documents.sqlite3`.

## Machine learning methods

- **Forecasting:** Sums revenue by calendar month, fills missing months with zero, evaluates a linear time-trend model on the latest 20% of history (leaving at least two training months), and reports holdout MAE and RMSE. The model is then fit on all history to forecast future months. It is intentionally simple and does not model seasonality.
- **Customer segmentation:** Calculates Recency, Frequency (sales records), and Monetary value per customer, scales those features, and applies K-Means. The dashboard shows each segment's mean RFM characteristics so the groups can be interpreted from their measured values.
- **Anomaly detection:** Flags transaction revenue outside Tukey's 1.5×IQR bounds and displays the thresholds and direction. It is a screening rule, not a judgment that a transaction is erroneous.

These results appear in the dashboard's **Machine learning analysis** expander. Forecasting needs at least three calendar months, segmentation needs enough customers with distinct RFM profiles, and anomaly detection needs at least four transactions. When a filtered dataset is too small, the dashboard explains why that analysis is unavailable.

## Ask the AI business analyst

Open **Ask the AI business analyst**, enter a question, and click **Ask analyst**. The app selects a predefined Python analysis based on the question and calculates its results from the currently filtered records. The LLM receives those results and explains them. Raw sales rows are not sent. For customer-ranking questions, only the top five customer labels and their aggregated revenue are included. The model is instructed to stick to supplied evidence and disclose when it is insufficient. Expand **Python-calculated evidence** to inspect the exact results sent for explanation.

The analyst sends a request to Groq only after the button is clicked and a supported analysis succeeds. Without `GROQ_API_KEY`, it shows a configuration message. Provider behavior in automated tests is mocked; the test suite does not make API calls.

### Supported natural-language analyses

The question router selects only predefined Python operations. It supports total and monthly revenue, best-selling products by units, top region by revenue, top five customers by revenue, month-over-month revenue decreases with product and region changes, lowest-revenue products, next-month forecasts, and IQR revenue anomalies. Try questions such as “Which region generated the most revenue?”, “What was our revenue in March?”, or “Why did revenue decrease?”

Unsupported questions receive guidance instead of being passed to the model. The LLM never generates or executes Python. Month-over-month product and region changes are evidence of measured changes, not proof of their causes. Low product revenue does not establish poor performance without targets or margins. Customer labels are included only for customer-ranking questions.

Run the project tests with:

```powershell
python -m pytest -q --basetemp=.pytest-phase10-tmp tests
```

The test suite covers data ingestion and validation, analytics, forecasting, segmentation, anomaly detection, database persistence, sales and document API routes, LLM grounding with a mocked provider, and RAG extraction/retrieval/persistence. It makes no live Groq calls. Tests use a workspace-local temporary directory so they also work in restricted Windows environments.

## Run the API

```powershell
python -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/health` for the health response or `http://127.0.0.1:8000/docs` for the generated API documentation. Upload a CSV, `.xlsx`, or `.xlsm` file through `POST /upload` as multipart form field `file`; the response provides a `dataset_id` for subsequent requests.

API routes include `GET /data/{dataset_id}`, `DELETE /data/{dataset_id}`, `GET /kpis/{dataset_id}`, `GET /analytics/{dataset_id}`, `GET /forecast/{dataset_id}`, `GET /segments/{dataset_id}`, `GET /anomalies/{dataset_id}`, and `POST /ask`. The ask request body contains `dataset_id` and `question`. Validation errors use HTTP 422, unknown IDs return 404, unavailable LLM configuration returns 503, and upstream LLM failures return 502.

Document routes include `POST /documents/upload` (multipart `file` field), `GET /documents`, `DELETE /documents/{document_id}`, and `POST /documents/ask`. The ask body contains a `question` and an optional `document_id`; omitting the ID searches all indexed PDFs. Answers include retrieved source filenames and page numbers. Uploads are limited to 15 MiB and 500 pages. Encrypted PDFs and scanned PDFs without an OCR text layer are rejected with readable errors. Set `INSIGHTAI_DOCUMENT_DB` to change the SQLite document-store path.

Document retrieval uses a deterministic local word and word-pair hashing vectorizer with cosine similarity. It needs no external embedding API or model download, and its behavior is reproducible across restarts. It is lexical retrieval rather than a pretrained semantic embedding model, so paraphrases that share few words with the PDF may retrieve less relevant text. The LLM receives only the top retrieved excerpts, is told to treat PDF contents as untrusted data, and must cite source pages; review the cited pages for important decisions. Provider behavior should be configured with `GROQ_API_KEY` as described above.

Uploaded datasets and cleaned sales rows are persisted in the configured database. `DELETE /data/{dataset_id}` removes the metadata and associated sales rows in one transaction. KPI and ML outputs are calculated when requested rather than duplicated in storage.

## Project layout

```text
app/
  main.py       FastAPI application and health endpoint
  api/          Typed API schemas and sales/document routes
  database/     SQLAlchemy engine, dataset models, and persistence repository
  data/         Sales file loading, validation, and cleaning
  analytics/    KPI calculations and grouped sales summaries
  dashboard/    Streamlit dashboard interface
  ml/           Forecasting, RFM segmentation, and anomaly detection
  ai/           Groq client, evidence builder, and grounded analyst prompts
  rag/          PDF extraction, local vector storage, retrieval, and grounded answers
data/           Sample sales data
tests/          Automated checks
```

## Security and configuration

See [SECURITY.md](SECURITY.md) for the Phase 11 review, configuration rules, and limits. Keep `.env` private; only the empty-key `.env.example` belongs in version control. Uploaded PDF text and the user's document question are sent to Groq when answering a RAG question, so do not use confidential documents unless that data sharing is approved for your environment. The current API has no authentication or rate limiting and is intended for local or trusted use.

## Run with Docker Compose

Docker Compose starts PostgreSQL, the FastAPI service, and the Streamlit dashboard. The API and dashboard use the same image and source code but run as separate services. PostgreSQL waits for its health check before either app starts. API and dashboard ports are published only on `127.0.0.1` by default.

1. Copy `.env.example` to `.env` if you have not already done so.
2. Set `POSTGRES_PASSWORD` in `.env` to a unique password. Use URL-safe characters or percent-encode reserved characters because Compose also uses it in `DATABASE_URL`.
3. Optionally set `GROQ_API_KEY` to enable AI and document Q&A.
4. From the project directory, run:

```powershell
docker compose up --build
```

Open the dashboard at `http://127.0.0.1:8501`, the API at `http://127.0.0.1:8000`, and Swagger at `http://127.0.0.1:8000/docs`. Check service status in another terminal with `docker compose ps`. Stop the services with `Ctrl+C`; `docker compose down` removes the containers while preserving the PostgreSQL and RAG named volumes.

To change the published ports, set `API_PORT` or `DASHBOARD_PORT` in `.env`. The PostgreSQL port is not published to the host. The Compose stack stores SQL data in `postgres_data` and indexed PDF text/vectors in `rag_data`.

For a private server deployment, follow [DEPLOYMENT.md](DEPLOYMENT.md). The services remain bound to loopback; provide remote access only through a private VPN or an authenticated, access-restricted gateway.

## Known limitations and future work

- The API has no authentication, authorization, rate limiting, or per-user data isolation; keep access on a trusted private network.
- The business-question router supports a fixed set of analyses. Unsupported questions receive guidance instead of a general answer.
- Forecasting uses a linear trend and does not model seasonality. Anomaly detection is a screening rule, not a data-quality verdict.
- PDF retrieval is lexical and does not OCR scanned pages. Retrieved document text and the question are sent to Groq when generating an answer.
- Hosted deployment smoke checks and representative dashboard screenshots remain to be completed.

Potential follow-up work includes user access controls, retention and backup automation, semantic document retrieval, richer forecasting, and repeatable deployment checks.

## Development status

The repository is intentionally being built one phase at a time. Current limitations: the router supports a fixed set of intents; the hosted deployment has not been independently verified after the PostgreSQL driver update. The project does not save AI conversations or persist computed analysis outputs.
