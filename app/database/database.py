"""SQLAlchemy engine and session configuration."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import StaticPool

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_URL = "sqlite:///./insightai.db"


class Base(DeclarativeBase):
    """Base class for SQLAlchemy ORM models."""


def get_database_url(database_url: str | None = None) -> str:
    """Read the URL from configuration without exposing credentials."""
    if database_url is not None:
        if not database_url.strip():
            raise ValueError("DATABASE_URL must not be empty.")
        return database_url.strip()
    load_dotenv(_PROJECT_ROOT / ".env", override=False)
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL).strip() or DEFAULT_DATABASE_URL


def create_database_engine(database_url: str | None = None) -> Engine:
    """Build an engine for PostgreSQL or a local SQLite development database."""
    url = get_database_url(database_url)
    if url.startswith("sqlite"):
        if url.endswith(":memory:"):
            return create_engine(
                url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        return create_engine(url, connect_args={"check_same_thread": False})
    return create_engine(url, pool_pre_ping=True)


def create_session_factory(engine: Engine) -> sessionmaker:
    """Create a session factory bound to the supplied engine."""
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@lru_cache(maxsize=1)
def get_default_engine() -> Engine:
    """Return the process-wide configured engine (connections remain lazy)."""
    return create_database_engine()


@lru_cache(maxsize=1)
def get_default_session_factory() -> sessionmaker:
    return create_session_factory(get_default_engine())
