"""Errors exposed by the database layer."""


class DatabaseUnavailableError(RuntimeError):
    """Raised when the configured database cannot serve a request."""
