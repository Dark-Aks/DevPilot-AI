from __future__ import annotations

import re

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from app.models.schemas import IngestRequest, IngestResponse
from app.services.webhook_handler import ingest_full_repo
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["ingest"])

_REPO_URL_PATTERN = re.compile(r"github\.com/([^/]+)/([^/\s.]+)")


def _parse_repo_url(url: str) -> tuple[str, str]:
    match = _REPO_URL_PATTERN.search(url)
    if not match:
        raise ValueError(f"Invalid GitHub repository URL: {url}")
    return match.group(1), match.group(2)


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_200_OK)
async def ingest_repository(request: IngestRequest) -> IngestResponse:
    """Ingest an entire GitHub repository into the vector store."""
    try:
        owner, repo_name = _parse_repo_url(request.repo_url)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    result = await ingest_full_repo(owner, repo_name, request.branch)

    return IngestResponse(
        repo=result["repo"],
        branch=result["branch"],
        files_processed=result["files_processed"],
        total_files=result["total_files"],
        total_chunks=result["total_chunks"],
        errors=result.get("errors", []),
    )
