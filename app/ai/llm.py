"""Small, lazily configured Groq chat-completion client."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.ai.errors import LLMConfigurationError, LLMServiceError

DEFAULT_MODEL = "openai/gpt-oss-20b"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configured_model(model: str | None = None) -> str:
    """Return the selected model, loading project configuration if present."""
    load_dotenv(_PROJECT_ROOT / ".env", override=False)
    selected = model or os.getenv("GROQ_MODEL") or DEFAULT_MODEL
    if not selected.strip():
        raise LLMConfigurationError("GROQ_MODEL must not be empty.")
    return selected.strip()


def create_groq_client(api_key: str | None = None) -> Any:
    """Create a Groq client without exposing credential values in errors."""
    load_dotenv(_PROJECT_ROOT / ".env", override=False)
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key or not key.strip():
        raise LLMConfigurationError(
            "GROQ_API_KEY is not configured. Add it to the private .env file."
        )
    try:
        from groq import Groq

        return Groq(api_key=key.strip())
    except Exception as error:
        raise LLMConfigurationError(
            "The Groq client could not be initialized. Check the installation and configuration."
        ) from error


def complete_chat(
    messages: list[dict[str, str]],
    *,
    client: Any | None = None,
    model: str | None = None,
) -> str:
    """Request a chat completion and return nonempty response text."""
    selected_model = configured_model(model)
    active_client = client if client is not None else create_groq_client()
    try:
        response = active_client.chat.completions.create(
            messages=messages,
            model=selected_model,
            temperature=0.2,
        )
        content = response.choices[0].message.content
    except Exception as error:
        raise LLMServiceError(
            "The Groq request failed. Check the service connection and model configuration, then try again."
        ) from error
    if not isinstance(content, str) or not content.strip():
        raise LLMServiceError("The Groq service returned an empty answer. Please try again.")
    return content.strip()
