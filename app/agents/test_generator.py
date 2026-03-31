"""
Test Generator Agent — DevPilot AI

Generates two types of tests based on code changes:
  1. k6 performance tests (JavaScript) — for API/backend changes
  2. Selenium UI tests (Python) — for frontend/UI changes

Invoked when change_types include: API, UI, or UNKNOWN.
The agent inspects the change context to decide which test type to generate.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

from app.utils.errors import agent_fallback
from app.utils.logging import get_logger
from app.utils.metrics import track_latency
from app.agents.state import AgentState

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a QA engineer specializing in test automation.
You generate two types of tests:
1. **k6 performance tests** — JavaScript-based load/stress tests
2. **Selenium UI tests** — Python-based browser automation tests

You MUST respond with valid JSON in this exact format:
{
  "tests": [
    {
      "test_type": "k6",
      "file_name": "tests/load/test_<feature>.js",
      "description": "What this test validates",
      "code": "// k6 test code here"
    },
    {
      "test_type": "selenium",
      "file_name": "tests/e2e/test_<feature>.py",
      "description": "What this test validates",
      "code": "# Selenium test code here"
    }
  ]
}

Guidelines:
- Generate k6 tests for API endpoints and backend logic changes
- Generate Selenium tests for UI/frontend changes
- Follow k6 best practices: use scenarios, thresholds, checks
- Follow Selenium best practices: use Page Object Model, explicit waits
- Make tests self-contained and runnable
- Include setup/teardown where needed
- Use realistic test data
- Never invent endpoints or URLs not evident from the code context
"""


@agent_fallback("test_generator", "test_suggestions", [])
async def run_test_generator(state: AgentState, llm: BaseChatModel) -> dict[str, Any]:
    """Generate k6 and Selenium tests for the changed code."""
    diff = state.get("diff", "")
    context = state.get("rag_context_text", "")
    changed = state.get("changed_files", [])
    change_types = state.get("change_types", [])

    files_list = "\n".join(f"- {f['filename']} ({f['status']})" for f in changed)

    # Tailor the instruction based on detected change types
    test_guidance = "Generate both k6 and Selenium tests as appropriate."
    if "api" in change_types and "ui" not in change_types:
        test_guidance = "Focus on k6 performance tests since changes are API/backend-only."
    elif "ui" in change_types and "api" not in change_types:
        test_guidance = "Focus on Selenium UI tests since changes are frontend-only."

    user_msg = f"""Generate appropriate tests for the following code changes:

## Changed Files
{files_list}

## Change Types Detected
{', '.join(change_types)}

## Diff
```
{diff[:8000]}
```

## Related Codebase Context (existing code patterns)
{context[:6000]}

{test_guidance}
Respond as JSON.
"""

    with track_latency("agent_test_generator"):
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
    tests = result.get("tests", [])
    logger.info("test_generator_complete", test_count=len(tests))
    return {"test_suggestions": tests}
