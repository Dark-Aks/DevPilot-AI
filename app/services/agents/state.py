"""
Agent State — DevPilot AI

Defines the shared state schema that flows through the LangGraph workflow.
Every node reads from and writes to this TypedDict, enabling:
  - Data sharing between agents without tight coupling
  - Conditional routing based on classified change types
  - Accumulated metrics across the pipeline
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated, TypedDict

from langchain_core.documents import Document


def _merge_list(a: list, b: list) -> list:
    """Reducer: concatenate lists (used for errors accumulation)."""
    return a + b


class ChangeType(str, Enum):
    """Classification of code changes for conditional agent routing.

    The classifier node inspects file paths and diff content to assign
    one or more change types, which determine which agents are invoked.
    """
    API = "api"               # Route/endpoint changes → test_generator + review
    LOGIC = "logic"           # Business logic changes → review + code_understanding
    UI = "ui"                 # Frontend/template changes → test_generator (selenium)
    CONFIG = "config"         # Config/infra changes → documentation
    DOCS = "docs"             # Documentation files → documentation agent only
    TEST = "test"             # Test file changes → review only
    SCHEMA = "schema"         # Data model changes → documentation + review
    UNKNOWN = "unknown"       # Fallback → all agents


class ChangedFile(TypedDict):
    filename: str
    status: str  # added | modified | removed


class AgentState(TypedDict, total=False):
    """Shared state flowing through the LangGraph agent workflow.

    Sections:
      Input     — provided by the webhook/API caller
      Routing   — set by the classifier node
      RAG       — set by the retrieval node
      Outputs   — set by individual agents
      Metrics   — accumulated across all nodes
    """

    # ── Input context ──
    repo: str
    branch: str
    changed_files: list[ChangedFile]
    diff: str
    commit_id: str

    # ── Routing decisions (set by classifier node) ──
    change_types: list[str]           # Detected change categories
    agents_to_run: list[str]          # Agent names selected by router
    routing_reasoning: str            # LLM explanation for routing choice

    # ── RAG retrieval output ──
    rag_context: list[Document]
    rag_context_text: str

    # ── Agent outputs ──
    code_understanding: dict | None
    test_suggestions: list[dict]
    documentation_updates: list[dict]
    review_findings: list[dict]

    # ── Metrics & error tracking ──
    errors: Annotated[list[str], _merge_list]
    metrics: dict                     # RequestMetrics.summary() snapshot

    # Error tracking
    errors: Annotated[list[str], _merge_list]
