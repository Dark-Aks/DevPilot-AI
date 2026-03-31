from __future__ import annotations

from functools import lru_cache
from typing import Annotated

import chromadb
from fastapi import Depends

from app.config import Settings, settings
from app.core.llm import get_llm


@lru_cache
def get_settings() -> Settings:
    return settings


def get_chroma_client(
    cfg: Annotated[Settings, Depends(get_settings)],
) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(cfg.chroma_path))


def get_llm_instance(cfg: Annotated[Settings, Depends(get_settings)]):
    return get_llm(
        provider=cfg.llm_provider,
        model=cfg.llm_model,
        temperature=cfg.llm_temperature,
        api_key=cfg.openai_api_key if cfg.llm_provider == "openai" else cfg.anthropic_api_key,
    )
