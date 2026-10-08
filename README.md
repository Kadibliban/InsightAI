# InsightAI — AI Business Intelligence Assistant

InsightAI is a portfolio project for exploring business and sales data with analytics, machine learning, and grounded natural-language explanations. Development is proceeding in phases.

## Current phase

**Phase 9 — Retrieval-augmented generation (RAG).** The app accepts text-based PDFs, extracts and chunks page text, stores local text vectors persistently, retrieves relevant excerpts, and asks the configured LLM to answer with source-page citations. Sales datasets remain persisted through SQLAlchemy; PostgreSQL is supported as the target database and SQLite remains available for local development.

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

## Development status

The repository is intentionally being built one phase at a time. Current limitations: the router supports a fixed set of intents; a hosted deployment has not been created. The project does not save AI conversations or persist computed analysis outputs.
