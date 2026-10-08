"""FastAPI application entry point for InsightAI."""

from fastapi import FastAPI

from app.api.routes import documents_router, router as sales_router

app = FastAPI(
    title="InsightAI",
    description="Sales data, business analytics, ML, and grounded question-answering API.",
    version="0.2.0",
)


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    """Report that the API process is responding."""
    return {"status": "ok"}


app.include_router(sales_router)
app.include_router(documents_router)
