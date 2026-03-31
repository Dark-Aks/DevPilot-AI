"""
Documentation Agent — DevPilot AI

Generates and updates project documentation based on code changes:
  - README sections for new features
  - API reference updates for endpoint changes
  - Configuration docs for config/schema changes

Invoked when change_types include: CONFIG, DOCS, SCHEMA, or UNKNOWN.
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

SYSTEM_PROMPT = """You are a technical writer who maintains project documentation.
Your job is to identify what documentation needs to be created or updated based on code changes.

You MUST respond with valid JSON in this exact format:
{
  "updates": [
    {
      "section": "Section name (e.g., API Reference, Installation, Usage)",
      "action": "create | update | delete",
      "content": "The documentation content in Markdown format"
    }
  ]
}

Guidelines:
- Focus on user-facing documentation: README sections, API docs, usage guides
- For new features, create documentation sections
- For changed APIs, update parameter descriptions, examples, and return values
- For removed features, mark sections for deletion
- Write clear, concise documentation with code examples
- Use Markdown formatting: headers, code blocks, tables where appropriate
- If changes are internal refactoring with no user-facing impact, return minimal updates
- Never fabricate API endpoints or parameters not present in the code
"""


@agent_fallback("documentation", "documentation_updates", [])
async def run_documentation(state: AgentState, llm: BaseChatModel) -> dict[str, Any]:
    """Generate documentation updates based on code changes."""
    diff = state.get("diff", "")
    context = state.get("rag_context_text", "")
    changed = state.get("changed_files", [])

    files_list = "\n".join(f"- {f['filename']} ({f['status']})" for f in changed)

    user_msg = f"""Based on the following code changes, suggest documentation updates:

## Changed Files
{files_list}

## Diff
```
{diff[:8000]}
```

## Existing Codebase Context
{context[:6000]}

Identify what documentation should be created, updated, or removed.
Respond as JSON.
"""

    with track_latency("agent_documentation"):
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
    updates = result.get("updates", [])
    logger.info("documentation_complete", update_count=len(updates))
    return {"documentation_updates": updates}
