"""
LangGraph Agent Workflow — DevPilot AI

Implements a conditional, non-sequential multi-agent workflow:

  ┌─────────┐    ┌────────────┐    ┌──────────────┐    ┌────────────────┐
  │  START   │───▶│  Classify  │───▶│   Retrieve   │───▶│  Route & Run   │
  └─────────┘    │  Changes   │    │   Context     │    │  Agents (||)   │
                 └────────────┘    └──────────────┘    └───────┬────────┘
                                                               │
                 ┌─────────────────────────────────────────────┘
                 │  Conditional fan-out based on change types:
                 │
                 │  API change    → test_generator + review
                 │  Logic change  → code_understanding + review
                 │  UI change     → test_generator (selenium) + review
                 │  Config change → documentation
                 │  Schema change → documentation + review
                 │  Docs change   → documentation only
                 │  Unknown       → all four agents
                 │
                 ▼
          ┌──────────────┐    ┌─────────────┐
          │  Aggregate   │───▶│   Emit      │───▶ END
          │  Results     │    │   Metrics   │
          └──────────────┘    └─────────────┘

Key design decisions:
  - Agents are NOT always sequential — they fan out in parallel
  - Routing is CONDITIONAL based on LLM-classified change types
  - Each agent is independently resilient (fallback on failure)
  - Metrics are collected at every node boundary
"""
from __future__ import annotations

import asyncio
import re
import uuid
from typing import Any

from langgraph.graph import StateGraph, START, END

from app.config import settings
from app.core.errors import agent_fallback
from app.core.llm import get_llm
from app.core.logging import get_logger
from app.core.metrics import RequestMetrics, track_latency, count_tokens
from app.services.agents.state import AgentState, ChangeType
from app.services.agents.code_understanding import run_code_understanding
from app.services.agents.test_generator import run_test_generator
from app.services.agents.documentation import run_documentation
from app.services.agents.review import run_review
from app.services.rag.retriever import retrieve_for_changes, format_context

logger = get_logger(__name__)


# ── Agent routing table ──
# Maps each ChangeType to the set of agents that should handle it

ROUTING_TABLE: dict[str, list[str]] = {
    ChangeType.API:     ["code_understanding", "test_generator", "review"],
    ChangeType.LOGIC:   ["code_understanding", "review"],
    ChangeType.UI:      ["test_generator", "review"],
    ChangeType.CONFIG:  ["documentation", "review"],
    ChangeType.DOCS:    ["documentation"],
    ChangeType.TEST:    ["review"],
    ChangeType.SCHEMA:  ["documentation", "review", "code_understanding"],
    ChangeType.UNKNOWN: ["code_understanding", "test_generator", "documentation", "review"],
}


# ── File-path heuristics for change classification ──

_CHANGE_PATTERNS: list[tuple[str, ChangeType]] = [
    (r"(route|endpoint|controller|api|handler|view)s?\.(py|js|ts)", ChangeType.API),
    (r"(test|spec|__test__|_test)\.(py|js|ts)", ChangeType.TEST),
    (r"\.(html|css|scss|jsx|tsx|vue|svelte)$", ChangeType.UI),
    (r"(README|CHANGELOG|docs/|\.md$)", ChangeType.DOCS),
    (r"(config|settings|\.env|\.yaml|\.yml|\.toml|Dockerfile|docker)", ChangeType.CONFIG),
    (r"(model|schema|migration|entity|types)\.(py|js|ts)", ChangeType.SCHEMA),
]


def _classify_by_file_paths(files: list[dict]) -> set[str]:
    """Heuristic classification based on file path patterns."""
    types: set[str] = set()
    for f in files:
        path = f.get("filename", "").lower()
        matched = False
        for pattern, change_type in _CHANGE_PATTERNS:
            if re.search(pattern, path):
                types.add(change_type.value)
                matched = True
                break
        if not matched:
            types.add(ChangeType.LOGIC.value)
    return types


# ── Graph node functions ──


async def classify_changes(state: AgentState) -> dict[str, Any]:
    """Node 1: Classify the type of code changes to determine agent routing.

    Uses file-path heuristics (fast, no LLM call) to classify changes.
    Falls back to UNKNOWN if no patterns match, which triggers all agents.
    """
    changed_files = state.get("changed_files", [])

    if not changed_files:
        return {
            "change_types": [ChangeType.UNKNOWN.value],
            "agents_to_run": list(ROUTING_TABLE[ChangeType.UNKNOWN]),
            "routing_reasoning": "No changed files detected — running all agents as fallback.",
        }

    # Heuristic classification
    change_types = _classify_by_file_paths(changed_files)
    if not change_types:
        change_types = {ChangeType.UNKNOWN.value}

    # Determine agents to run (union of all matched change types)
    agents: set[str] = set()
    for ct in change_types:
        agents.update(ROUTING_TABLE.get(ct, ROUTING_TABLE[ChangeType.UNKNOWN]))

    agents_list = sorted(agents)
    reasoning = (
        f"Detected change types: {sorted(change_types)}. "
        f"Routing to agents: {agents_list}."
    )

    logger.info(
        "change_classification",
        change_types=sorted(change_types),
        agents=agents_list,
        file_count=len(changed_files),
    )

    return {
        "change_types": sorted(change_types),
        "agents_to_run": agents_list,
        "routing_reasoning": reasoning,
    }


async def retrieve_context(state: AgentState) -> dict[str, Any]:
    """Node 2: RAG retrieval — fetch relevant code context for the changed files."""
    changed_files = state.get("changed_files", [])
    repo = state.get("repo", "")

    if not changed_files or not repo:
        return {"rag_context": [], "rag_context_text": "No context available."}

    with track_latency("rag_retrieve_for_changes", repo=repo):
        docs = retrieve_for_changes(changed_files, repo, top_k=settings.rag_top_k)
        context_text = format_context(docs)

    return {"rag_context": docs, "rag_context_text": context_text}


async def run_selected_agents(state: AgentState) -> dict[str, Any]:
    """Node 3: Run ONLY the agents selected by the classifier, in parallel.

    This is the core intelligence — not a blind pipeline.
    The routing decision from classify_changes determines which agents execute.
    """
    agents_to_run = set(state.get("agents_to_run", []))

    api_key = (
        settings.openai_api_key
        if settings.llm_provider == "openai"
        else settings.anthropic_api_key
    )
    llm = get_llm(
        provider=settings.llm_provider,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=api_key,
    )

    # Build task list based on routing decision
    agent_map = {
        "code_understanding": run_code_understanding,
        "test_generator": run_test_generator,
        "documentation": run_documentation,
        "review": run_review,
    }

    tasks = []
    task_names = []
    for name, func in agent_map.items():
        if name in agents_to_run:
            tasks.append(func(state, llm))
            task_names.append(name)

    if not tasks:
        logger.warning("no_agents_selected", routing=list(agents_to_run))
        return {"errors": ["No agents were selected for execution."]}

    logger.info("agents_dispatched", agents=task_names, total=len(tasks))

    # Run selected agents concurrently
    with track_latency("agents_parallel_execution", agent_count=len(tasks)):
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Merge results
    merged: dict[str, Any] = {"errors": []}
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error("agent_exception", agent=task_names[i], error=str(result))
            merged["errors"].append(f"{task_names[i]}: {type(result).__name__}: {str(result)}")
        elif isinstance(result, dict):
            for key, value in result.items():
                if key == "errors" and isinstance(value, list):
                    merged["errors"].extend(value)
                else:
                    merged[key] = value

    return merged


async def collect_metrics(state: AgentState) -> dict[str, Any]:
    """Node 4: Collect and emit metrics for the entire workflow run."""
    metrics = RequestMetrics(request_id=state.get("commit_id", str(uuid.uuid4())))

    # Count what we got
    metrics.record_retrieval(
        total=len(state.get("rag_context", [])),
        relevant=len(state.get("rag_context", [])),  # Heuristic: all returned are relevant
    )
    metrics.agents_invoked = state.get("agents_to_run", [])
    metrics.errors = state.get("errors", [])

    # Estimate token usage from context + diff
    context_text = state.get("rag_context_text", "")
    diff = state.get("diff", "")
    input_tokens = count_tokens(context_text) + count_tokens(diff)
    metrics.total_input_tokens = input_tokens
    metrics.total_output_tokens = input_tokens // 2  # Rough estimate
    metrics.total_cost_usd = metrics.total_cost_usd  # Will be set by agent-level tracking

    summary = metrics.summary()
    metrics.emit()

    return {"metrics": summary}


# ── Build the graph ──


def build_workflow() -> StateGraph:
    """Construct the LangGraph StateGraph with conditional routing.

    Graph topology:
      START → classify → retrieve → run_agents → metrics → END

    The classify node sets `agents_to_run` which controls which agents
    actually execute in the run_agents node. This is conditional routing
    without separate graph branches — simpler and more maintainable.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify_changes", classify_changes)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("run_selected_agents", run_selected_agents)
    graph.add_node("collect_metrics", collect_metrics)

    # Define edges
    graph.add_edge(START, "classify_changes")
    graph.add_edge("classify_changes", "retrieve_context")
    graph.add_edge("retrieve_context", "run_selected_agents")
    graph.add_edge("run_selected_agents", "collect_metrics")
    graph.add_edge("collect_metrics", END)

    return graph


# Compile the graph once at module level for reuse
_workflow = build_workflow().compile()


async def run_workflow(
    repo: str,
    changed_files: list[dict],
    diff: str = "",
    branch: str = "",
    commit_id: str = "",
) -> dict[str, Any]:
    """Execute the full agent workflow with conditional routing.

    This is the main entry point called by the webhook handler. It:
      1. Classifies changes to determine which agents to invoke
      2. Retrieves RAG context for the changed files
      3. Runs selected agents in parallel
      4. Collects metrics and returns structured results

    Args:
        repo: Repository identifier (owner/repo).
        changed_files: List of {filename, status} dicts.
        diff: Unified diff text.
        branch: Branch name.
        commit_id: Git commit SHA for tracking.

    Returns:
        Combined workflow result with all agent outputs and metrics.
    """
    initial_state: AgentState = {
        "repo": repo,
        "branch": branch,
        "changed_files": changed_files,
        "diff": diff,
        "commit_id": commit_id or str(uuid.uuid4()),
        "change_types": [],
        "agents_to_run": [],
        "routing_reasoning": "",
        "rag_context": [],
        "rag_context_text": "",
        "code_understanding": None,
        "test_suggestions": [],
        "documentation_updates": [],
        "review_findings": [],
        "errors": [],
        "metrics": {},
    }

    logger.info("workflow_start", repo=repo, changed_files=len(changed_files))

    with track_latency("workflow_total", repo=repo) as timing:
        result = await _workflow.ainvoke(initial_state)

    logger.info(
        "workflow_end",
        repo=repo,
        change_types=result.get("change_types", []),
        agents_run=result.get("agents_to_run", []),
        has_understanding=result.get("code_understanding") is not None,
        test_count=len(result.get("test_suggestions", [])),
        doc_count=len(result.get("documentation_updates", [])),
        review_count=len(result.get("review_findings", [])),
        error_count=len(result.get("errors", [])),
        total_ms=timing.get("duration_ms", 0),
    )

    return result

    logger.info("workflow_start", repo=repo, changed_files=len(changed_files))

    result = await _workflow.ainvoke(initial_state)

    logger.info(
        "workflow_end",
        repo=repo,
        has_understanding=result.get("code_understanding") is not None,
        test_count=len(result.get("test_suggestions", [])),
        doc_count=len(result.get("documentation_updates", [])),
        review_count=len(result.get("review_findings", [])),
        error_count=len(result.get("errors", [])),
    )

    return result
