from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ── Enums ──

class AgentType(str, Enum):
    CODE_UNDERSTANDING = "code_understanding"
    TEST_GENERATOR = "test_generator"
    DOCUMENTATION = "documentation"
    REVIEW = "review"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ── Request Models ──

class IngestRequest(BaseModel):
    repo_url: str = Field(..., description="GitHub repo URL (e.g. https://github.com/owner/repo)")
    branch: str = Field(default="main", description="Branch to ingest")


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query about the codebase")
    repo: str = Field(..., description="Repository identifier (owner/repo)")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results")
    filter_language: str | None = Field(default=None, description="Filter by language")
    filter_file_path: str | None = Field(default=None, description="Filter by file path")


# ── Response Models ──

class CodeChunkResult(BaseModel):
    file_path: str
    function_name: str = ""
    class_name: str = ""
    chunk_type: str
    language: str
    start_line: int
    end_line: int
    content: str
    score: float = 0.0


class QueryResponse(BaseModel):
    query: str
    repo: str
    results: list[CodeChunkResult]
    total: int


class IngestResponse(BaseModel):
    repo: str
    branch: str
    files_processed: int
    total_files: int
    total_chunks: int
    errors: list[str] = Field(default_factory=list)


# ── Agent Output Models ──

class CodeExplanation(BaseModel):
    summary: str = Field(description="High-level summary of changes")
    details: list[str] = Field(default_factory=list, description="Detailed explanations per change")
    impact: str = Field(default="", description="Impact analysis")


class TestSuggestion(BaseModel):
    test_type: str = Field(description="k6 | selenium")
    file_name: str = Field(description="Suggested test file name")
    code: str = Field(description="Generated test code")
    description: str = Field(default="", description="What the test covers")


class DocUpdate(BaseModel):
    section: str = Field(description="Documentation section to update")
    content: str = Field(description="Updated documentation content")
    action: str = Field(default="update", description="create | update | delete")


class ReviewFinding(BaseModel):
    severity: Severity
    category: str = Field(description="bug | security | performance | style | suggestion")
    file_path: str = ""
    line: int | None = None
    message: str
    suggestion: str = ""


class MetricsInfo(BaseModel):
    total_latency_ms: float = 0.0
    agent_latencies: dict[str, float] = Field(default_factory=dict)
    retrieval_latency_ms: float = 0.0
    tokens_used: int = 0
    estimated_cost_usd: float = 0.0


class WorkflowResponse(BaseModel):
    repo: str
    branch: str = ""
    changed_files: list[str] = Field(default_factory=list)
    # ── Routing ──
    change_types: list[str] = Field(default_factory=list, description="Detected change categories")
    agents_used: list[str] = Field(default_factory=list, description="Agents invoked for this run")
    routing_reasoning: str = Field(default="", description="Why these agents were selected")
    # ── Agent outputs ──
    code_understanding: CodeExplanation | None = None
    test_suggestions: list[TestSuggestion] = Field(default_factory=list)
    documentation_updates: list[DocUpdate] = Field(default_factory=list)
    review_findings: list[ReviewFinding] = Field(default_factory=list)
    # ── Observability ──
    metrics: MetricsInfo | None = None
    errors: list[str] = Field(default_factory=list)


class WebhookResponse(BaseModel):
    status: str
    message: str
    task_id: str = ""
