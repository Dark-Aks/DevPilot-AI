from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def get_embedding_function() -> OpenAIEmbeddings:
    """Return an OpenAI embedding function configured from settings."""
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
