from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──
    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.1
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # ── Embeddings ──
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 100

    # ── GitHub ──
    github_token: str = ""
    github_webhook_secret: str = ""
    github_api_base: str = "https://api.github.com"

    # ── ChromaDB ──
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_prefix: str = "devpilot"

    # ── RAG ──
    rag_top_k: int = 15
    rag_rerank_top_k: int = 8
    rag_hybrid_alpha: float = 0.7  # Weight: 0=keyword-only, 1=vector-only

    # ── Caching ──
    cache_ttl_seconds: int = 300
    cache_max_size: int = 1000

    # ── Resilience ──
    llm_timeout_seconds: float = 120.0
    github_circuit_breaker_threshold: int = 5
    github_circuit_breaker_recovery: float = 60.0

    # ── App ──
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    app_version: str = "0.2.0"

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_persist_dir)


settings = Settings()
