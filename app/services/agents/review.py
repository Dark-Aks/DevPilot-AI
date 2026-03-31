"""
Code Review Agent — DevPilot AI

Performs automated code review focused on:
  - Bug detection (null refs, off-by-one, race conditions)
  - Security vulnerabilities (injection, auth bypass, secrets exposure)
  - Performance issues (N+1 queries, resource leaks)
  - Actionable improvement suggestions

Invoked when change_types include: API, LOGIC, TEST, SCHEMA, CONFIG, or UNKNOWN.
This is the most commonly invoked agent — almost all changes get reviewed.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

from app.core.errors import agent_fallback
from app.core.logging import get_logger
from app.core.metrics import track_latency
from app.services.agents.state import AgentState

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a senior code reviewer focused on code quality, security, and best practices.
Review the code changes and identify bugs, security issues, and improvements.

You MUST respond with valid JSON in this exact format:
{
  "findings": [
    {
      "severity": "info | warning | error | critical",
      "category": "bug | security | performance | style | suggestion",
      "file_path": "path/to/file.py",
      "line": 42,
      "message": "Clear description of the issue",
      "suggestion": "How to fix or improve it"
    }
  ]
}

Review categories and priorities:
1. **critical/error** — Security vulnerabilities (injection, auth bypass, data exposure), data loss, crashes
2. **warning** — Bugs, race conditions, missing error handling, performance issues
3. **info** — Code style, naming, documentation, minor improvements

Guidelines:
- Be specific: reference file paths and line numbers when possible
- Provide actionable suggestions, not just complaints
- Consider the broader codebase context when evaluating changes
- Check for: SQL/command injection, hardcoded secrets, missing input validation,
  unhandled exceptions, resource leaks, race conditions, N+1 queries
- Don't flag minor style issues unless they impact readability significantly
- Never fabricate line numbers — only reference lines visible in the diff
"""


@agent_fallback("review", "review_findings", [])
async def run_review(state: AgentState, llm: BaseChatModel) -> dict[str, Any]:
    """Review code changes for bugs, security issues, and improvements."""
    diff = state.get("diff", "")
    context = state.get("rag_context_text", "")
    changed = state.get("changed_files", [])

    files_list = "\n".join(f"- {f['filename']} ({f['status']})" for f in changed)

    user_msg = f"""Review the following code changes for bugs, security issues, and improvements:

## Changed Files
{files_list}

## Diff
```
{diff[:8000]}
```

## Related Codebase Context
{context[:6000]}

Provide your review findings as JSON.
"""

    with track_latency("agent_review"):
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
    findings = result.get("findings", [])
    logger.info("review_complete", finding_count=len(findings))
    return {"review_findings": findings}
