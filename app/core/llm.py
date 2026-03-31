from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from app.core.logging import get_logger

logger = get_logger(__name__)


def get_llm(
    provider: str = "openai",
    model: str = "gpt-4o",
    temperature: float = 0.1,
    api_key: str = "",
    **kwargs,
) -> BaseChatModel:
    """Factory that returns a LangChain chat model for the requested provider."""
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            **kwargs,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=api_key,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
