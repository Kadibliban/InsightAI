# Deployment guide

This guide prepares InsightAI for a single Linux server using Docker Compose. The current application has no sign-in, authorization, or rate limiting, so keep it on a private network. Do not expose the dashboard or API directly to the public internet.

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

- No public hosted instance has been created or validated yet. These steps run the application on a server you control.
- InsightAI has no user accounts, authorization, tenant separation, or rate limiting. The deployment is suitable only for a trusted private network or a gateway that enforces access control.
- Users with access can upload sales files and PDFs. Document Q&A sends retrieved PDF text and the question to Groq; obtain approval for that data transfer before uploading confidential material.
- The application has no automated backup, retention, or restore system. Configure and periodically verify those operations for your server.
- The API and dashboard currently use separate processes and do not provide per-user isolation for stored datasets or documents.
