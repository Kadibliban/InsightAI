"""Errors exposed by the optional Groq business analyst."""


class LLMConfigurationError(RuntimeError):
    """Raised when required LLM configuration is unavailable."""


class LLMServiceError(RuntimeError):
    """Raised when an LLM provider request fails or returns no content."""
