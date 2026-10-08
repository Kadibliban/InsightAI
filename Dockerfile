FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    INSIGHTAI_DOCUMENT_DB=/app/rag-data/rag_documents.sqlite3

WORKDIR /app

RUN groupadd --system --gid 10001 insightai \
    && useradd --system --uid 10001 --gid insightai --home-dir /app insightai \
    && mkdir -p /app/rag-data \
    && chown -R insightai:insightai /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY --chown=insightai:insightai app/ ./app/
COPY --chown=insightai:insightai data/ ./data/

USER insightai

EXPOSE 8000 8501

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
