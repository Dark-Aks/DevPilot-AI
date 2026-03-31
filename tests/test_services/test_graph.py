from __future__ import annotations

import pytest

from app.agents.graph import (
    classify_changes,
    _classify_by_file_paths,
    ROUTING_TABLE,
)
from app.agents.state import ChangeType


# ── Heuristic classification ──


def test_classify_api_file():
    files = [{"filename": "src/routes/users.py", "status": "modified"}]
    types = _classify_by_file_paths(files)
    assert ChangeType.API.value in types


def test_classify_test_file():
    files = [{"filename": "tests/test_login.py", "status": "added"}]
    types = _classify_by_file_paths(files)
    assert ChangeType.TEST.value in types


def test_classify_ui_file():
    files = [{"filename": "src/App.tsx", "status": "modified"}]
    types = _classify_by_file_paths(files)
    assert ChangeType.UI.value in types


def test_classify_docs_file():
    files = [{"filename": "README.md", "status": "modified"}]
    types = _classify_by_file_paths(files)
    assert ChangeType.DOCS.value in types


def test_classify_config_file():
    files = [{"filename": "docker-compose.yml", "status": "modified"}]
    types = _classify_by_file_paths(files)
    assert ChangeType.CONFIG.value in types


def test_classify_schema_file():
    files = [{"filename": "app/models/schema.py", "status": "modified"}]
    types = _classify_by_file_paths(files)
    assert ChangeType.SCHEMA.value in types


def test_classify_unmatched_defaults_to_logic():
    files = [{"filename": "src/utils/math.py", "status": "modified"}]
    types = _classify_by_file_paths(files)
    assert ChangeType.LOGIC.value in types


def test_classify_mixed_files():
    files = [
        {"filename": "src/routes/api.py", "status": "modified"},
        {"filename": "README.md", "status": "modified"},
    ]
    types = _classify_by_file_paths(files)
    assert ChangeType.API.value in types
    assert ChangeType.DOCS.value in types


# ── classify_changes graph node ──


@pytest.mark.asyncio
async def test_classify_changes_empty_files():
    state = {"changed_files": [], "change_types": [], "agents_to_run": [], "routing_reasoning": ""}
    result = await classify_changes(state)
    assert ChangeType.UNKNOWN.value in result["change_types"]
    # UNKNOWN triggers all agents
    assert len(result["agents_to_run"]) == len(ROUTING_TABLE[ChangeType.UNKNOWN])


@pytest.mark.asyncio
async def test_classify_changes_routes_api():
    state = {
        "changed_files": [{"filename": "src/routes/users.py", "status": "modified"}],
        "change_types": [],
        "agents_to_run": [],
        "routing_reasoning": "",
    }
    result = await classify_changes(state)
    assert ChangeType.API.value in result["change_types"]
    # API routes should invoke test_gen + review + understanding
    for agent in ROUTING_TABLE[ChangeType.API]:
        assert agent in result["agents_to_run"]


@pytest.mark.asyncio
async def test_classify_changes_docs_only():
    state = {
        "changed_files": [{"filename": "docs/guide.md", "status": "modified"}],
        "change_types": [],
        "agents_to_run": [],
        "routing_reasoning": "",
    }
    result = await classify_changes(state)
    assert ChangeType.DOCS.value in result["change_types"]
    assert "documentation" in result["agents_to_run"]
    # Docs changes should NOT trigger code_understanding or test_generator
    assert "code_understanding" not in result["agents_to_run"]
    assert "test_generator" not in result["agents_to_run"]


# ── Routing table integrity ──


def test_routing_table_covers_all_change_types():
    for ct in ChangeType:
        assert ct.value in ROUTING_TABLE or ct in ROUTING_TABLE, f"Missing routing for {ct}"


def test_routing_table_values_are_valid_agent_names():
    valid_agents = {"code_understanding", "test_generator", "documentation", "review"}
    for ct, agents in ROUTING_TABLE.items():
        for agent in agents:
            assert agent in valid_agents, f"Invalid agent '{agent}' for {ct}"
