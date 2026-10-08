# Deployment guide

This guide prepares InsightAI for a single Linux server using Docker Compose. The current application has no sign-in, authorization, or rate limiting, so keep self-hosted instances on a private network. Do not expose the dashboard or API directly to the public internet.

## Render API verification

The Render API is available at [https://insightai-kzvg.onrender.com](https://insightai-kzvg.onrender.com). On 2026-10-08, `GET /health` returned `{"status":"ok"}` and `/docs` loaded. A temporary upload of `data/sample_sales.csv` returned 24 records. KPI, analytics, forecast, segmentation, and anomaly endpoints all returned successfully; the KPI result included total revenue `654300.0`, 24 sales transactions, average order value `27262.5`, 8 customers, and 138 units. The temporary dataset was deleted. This confirms the PostgreSQL driver works for the API upload, read, and analysis paths. The service does not define a `/` homepage, so the base URL returns 404; use `/health` or `/docs`. The Streamlit dashboard is checked separately below; hosted LLM-backed flows have not been verified.

The Render API is publicly reachable and has no authentication or rate limiting. Do not upload confidential or personal data. Business questions and document answers can call the configured Groq account and consume its API quota.

## Render dashboard verification

The Streamlit dashboard is available at [https://insightai-dashboard.onrender.com/](https://insightai-dashboard.onrender.com/). On 2026-10-08, the dashboard page returned HTTP 200 and `/_stcore/health` returned `ok`. The README includes a cropped screenshot supplied for this deployment. The dashboard is publicly reachable without sign-in; use only sample or non-sensitive data. One grounded sales question and one synthetic PDF question were exercised through the live API; the full interactive widget workflow was not automated.

## Docker Compose verification

On 2026-10-08, `docker compose up --build` built both application images and started the API, Streamlit dashboard, and PostgreSQL services. The API and database reported healthy, the dashboard was running, and a follow-up `docker compose ps` confirmed the same states. The published API and dashboard ports were bound to `127.0.0.1`.

## Requirements

- A Linux server with Docker Engine and the Docker Compose plugin.
- Private network access, such as a VPN, or an authenticated gateway that restricts access to approved users.
- A Groq API key only if AI business analysis or document Q&A is needed.
- Sufficient server disk space for PostgreSQL data, uploaded document indexes, and backups.

## Configuration

Create `.env` beside `compose.yaml` from the example:

```sh
cp .env.example .env
```

Set these variables in `.env`:

| Variable | Required | Purpose |
| --- | --- | --- |
| `POSTGRES_PASSWORD` | Yes | Private PostgreSQL password. Use a strong URL-safe value because Compose also places it in `DATABASE_URL`. |
| `POSTGRES_DB` | No | Database name; defaults to `insightai`. |
| `POSTGRES_USER` | No | Database user; defaults to `insightai`. |
| `GROQ_API_KEY` | No | Enables LLM explanations and document Q&A. Keep it secret. |
| `GROQ_MODEL` | No | Groq model identifier; defaults to `openai/gpt-oss-20b`. |
| `API_PORT` | No | Host loopback port for FastAPI; defaults to `8000`. |
| `DASHBOARD_PORT` | No | Host loopback port for Streamlit; defaults to `8501`. |

Compose constructs `DATABASE_URL` from the PostgreSQL variables and uses the Docker service name `db`. The application can also use a direct SQLAlchemy URL outside Compose, for example `postgresql+psycopg2://USER:PASSWORD@HOST:5432/insightai`; percent-encode reserved characters in the username or password. A local non-container run defaults to `sqlite:///./insightai.db`.

`INSIGHTAI_DOCUMENT_DB` points to the SQLite document index. In Compose it is set to `/app/rag-data/rag_documents.sqlite3` and persists in the `rag_data` named volume. PostgreSQL records persist in `postgres_data`. Keep both volumes in backup and restore procedures.

Never commit `.env`, database passwords, API keys, uploaded customer data, or document indexes. `.gitignore` and `.dockerignore` exclude local secrets and database files.

## Start and check the services

From the project directory on the server:

```sh
docker compose up --build -d
docker compose ps
curl --fail http://127.0.0.1:8000/health
```

The API health endpoint should return `{"status":"ok"}`. On the server, the dashboard is available at `http://127.0.0.1:8501/`, and the API documentation is at `http://127.0.0.1:8000/docs`.

The Compose file publishes both application ports on loopback and does not publish PostgreSQL. For remote use, configure a private VPN or an authenticated gateway/reverse proxy that accepts traffic only from approved users and forwards to the loopback ports. Add TLS at that gateway when traffic crosses an untrusted network. Do not change the service bindings to `0.0.0.0` on an internet-facing host without first adding application authentication and abuse controls.

## Updates, logs, and data recovery

After updating the project files, rebuild and restart the services:

```sh
docker compose up --build -d
docker compose logs --tail=100 api dashboard db
```

Back up PostgreSQL and the `rag_data` volume before upgrades or maintenance. `docker compose down` preserves named volumes; `docker compose down --volumes` deletes them and should only be used when intentionally removing stored data. Verify backups by restoring them to a separate environment.

## Startup commands

The API container starts with `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`. The dashboard container starts with `streamlit run app/dashboard/dashboard.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true`. Docker Compose supplies these commands and restarts services after failures.

## Known limitations

- The Render API health, upload, KPI, analytics, ML, business-question, and RAG smoke paths, Streamlit dashboard availability, and local Docker Compose build/startup were checked. Full interactive widget automation was not performed. The steps in this guide run the application on a server you control.
- The Render API and dashboard are publicly reachable without sign-in, authorization, tenant separation, or rate limiting. Treat them as a sample-data demo; do not upload confidential or personal data. For self-hosting, restrict access to a trusted private network or a gateway that enforces access control.
- Users with access can upload sales files and PDFs. Document Q&A sends retrieved PDF text and the question to Groq; obtain approval for that data transfer before uploading confidential material.
- The application has no automated backup, retention, or restore system. Configure and periodically verify those operations for your server.
- The API and dashboard currently use separate processes and do not provide per-user isolation for stored datasets or documents.
