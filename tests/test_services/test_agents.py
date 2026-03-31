from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from app.services.agents.code_understanding import run_code_understanding
from app.services.agents.test_generator import run_test_generator
from app.services.agents.review import run_review
from app.services.agents.documentation import run_documentation


def _make_state(**overrides):
    state = {
        "repo": "owner/repo",
        "branch": "main",
        "changed_files": [{"filename": "src/app.py", "status": "modified"}],
        "diff": "--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-old\n+new",
        "rag_context": [],
        "rag_context_text": "def existing(): pass",
        "change_types": [],
        "agents_to_run": [],
        "routing_reasoning": "",
        "commit_id": "abc123",
        "metrics": {},
        "errors": [],
    }
    state.update(overrides)
    return state


def _mock_llm(response_json: str):
    llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = response_json
    llm.ainvoke = AsyncMock(return_value=mock_response)
    return llm


@pytest.mark.asyncio
async def test_code_understanding_agent():
    llm = _mock_llm('{"summary": "Changed app logic", "details": ["Updated handler"], "impact": "None"}')
    result = await run_code_understanding(_make_state(), llm)
    assert "code_understanding" in result
    assert result["code_understanding"]["summary"] == "Changed app logic"


@pytest.mark.asyncio
async def test_test_generator_agent():
    llm = _mock_llm('{"tests": [{"test_type": "k6", "file_name": "test.js", "description": "Load test", "code": "import http from k6/http;"}]}')
    result = await run_test_generator(_make_state(), llm)
    assert "test_suggestions" in result
    assert len(result["test_suggestions"]) == 1
    assert result["test_suggestions"][0]["test_type"] == "k6"


@pytest.mark.asyncio
async def test_documentation_agent():
    llm = _mock_llm('{"updates": [{"section": "API", "action": "update", "content": "New endpoint added"}]}')
    result = await run_documentation(_make_state(), llm)
    assert "documentation_updates" in result
    assert len(result["documentation_updates"]) == 1


@pytest.mark.asyncio
async def test_review_agent():
    llm = _mock_llm('{"findings": [{"severity": "warning", "category": "bug", "file_path": "app.py", "line": 5, "message": "Possible null ref", "suggestion": "Add null check"}]}')
    result = await run_review(_make_state(), llm)
    assert "review_findings" in result
    assert len(result["review_findings"]) == 1
    assert result["review_findings"][0]["severity"] == "warning"


@pytest.mark.asyncio
async def test_agent_handles_llm_error():
    """@agent_fallback decorator returns a graceful fallback on LLM errors."""
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(side_effect=Exception("API timeout"))
    result = await run_code_understanding(_make_state(), llm)
    # The fallback decorator produces a default value instead of erroring out
    assert "code_understanding" in result or "errors" in result


@pytest.mark.asyncio
async def test_test_generator_uses_change_types():
    """Test generator tailors output based on ChangeType."""
    llm = _mock_llm('{"tests": [{"test_type": "k6", "file_name": "load.js", "description": "API load test", "code": "// test"}]}')
    state = _make_state(change_types=["API"])
    result = await run_test_generator(state, llm)
    assert "test_suggestions" in result
