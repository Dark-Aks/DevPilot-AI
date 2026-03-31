"""
Code Understanding Agent — DevPilot AI

Analyzes code changes and produces structured explanations:
  - What changed and why (intent detection)
  - Impact analysis (dependencies, breaking changes)
  - Risk assessment for downstream effects

Invoked when change_types include: LOGIC, API, SCHEMA, or UNKNOWN.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

from app.utils.errors import agent_fallback
from app.utils.logging import get_logger
from app.utils.metrics import track_latency, count_tokens
from app.agents.state import AgentState

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a senior software engineer specializing in code analysis and explanation.
Your task is to analyze code changes and provide clear, structured explanations.

You MUST respond with valid JSON in this exact format:
{
  "summary": "A concise high-level summary of ALL changes (1-3 sentences)",
  "details": [
    "Detailed explanation of change 1",
    "Detailed explanation of change 2"
  ],
  "impact": "Analysis of how these changes affect the codebase — dependencies, performance, breaking changes, etc."
}

Guidelines:
- Be precise and technical but accessible
- Identify the purpose/intent behind changes, not just what changed
- Note any potential side effects or dependencies
- If changes span multiple files, explain the relationship between them
- Never hallucinate file names or line numbers not present in the diff
"""


@agent_fallback("code_understanding", "code_understanding", {"summary": "Analysis failed", "details": [], "impact": ""})
async def run_code_understanding(state: AgentState, llm: BaseChatModel) -> dict[str, Any]:
    """Analyze code changes and produce a structured explanation."""
    diff = state.get("diff", "")
    context = state.get("rag_context_text", "")
    changed = state.get("changed_files", [])

    files_list = "\n".join(f"- {f['filename']} ({f['status']})" for f in changed)

    user_msg = f"""Analyze the following code changes:

## Changed Files
{files_list}

## Diff
```
{diff[:8000]}
```

## Related Codebase Context
{context[:6000]}

Provide your analysis as JSON.
"""

    with track_latency("agent_code_understanding"):
        response = await llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])

    content = response.content.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    result = json.loads(content)
    logger.info("code_understanding_complete", summary_len=len(result.get("summary", "")))
    return {"code_understanding": result}
