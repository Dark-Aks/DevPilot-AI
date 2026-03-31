from __future__ import annotations

import hashlib
import hmac
from typing import Any

from app.config import settings
from app.utils.logging import get_logger
from app.services.client import GitHubClient
from app.rag.chunker import chunk_code
from app.rag.vectorstore import upsert_documents, delete_by_files

logger = get_logger(__name__)


def verify_webhook_signature(payload_body: bytes, signature: str) -> bool:
    """Verify GitHub webhook HMAC-SHA256 signature."""
    if not settings.github_webhook_secret:
        logger.warning("webhook_secret_not_configured")
        return False

    expected = hmac.new(
        settings.github_webhook_secret.encode("utf-8"),
        payload_body,
        hashlib.sha256,
    ).hexdigest()

    received = signature.removeprefix("sha256=")
    return hmac.compare_digest(expected, received)


def parse_push_event(payload: dict[str, Any]) -> dict:
    """Extract relevant information from a GitHub push webhook payload."""
    repo_full = payload.get("repository", {}).get("full_name", "")
    ref = payload.get("ref", "")
    before = payload.get("before", "")
    after = payload.get("after", "")

    added: list[str] = []
    modified: list[str] = []
    removed: list[str] = []

    for commit in payload.get("commits", []):
        added.extend(commit.get("added", []))
        modified.extend(commit.get("modified", []))
        removed.extend(commit.get("removed", []))

    # Deduplicate
    added = list(set(added))
    modified = list(set(modified))
    removed = list(set(removed))

    return {
        "repo": repo_full,
        "ref": ref,
        "before": before,
        "after": after,
        "added": added,
        "modified": modified,
        "removed": removed,
    }


async def process_push_event(payload: dict[str, Any]) -> dict:
    """Process a push event: update embeddings for changed files.

    Returns a summary of what was processed.
    """
    event = parse_push_event(payload)
    repo = event["repo"]
    ref = event["ref"]
    branch = ref.split("/")[-1] if "/" in ref else ref

    if not repo:
        logger.error("push_event_missing_repo")
        return {"error": "Missing repository information"}

    owner, repo_name = repo.split("/", 1)
    client = GitHubClient()

    # 1. Remove embeddings for deleted files
    if event["removed"]:
        delete_by_files(repo, event["removed"])

    # 2. Fetch and re-embed added + modified files
    changed_files = list(set(event["added"] + event["modified"]))
    documents_count = 0
    processed_files: list[str] = []
    errors: list[str] = []

    commit_id = event.get("after", "")

    for file_path in changed_files:
        try:
            content = await client.get_file_content(owner, repo_name, file_path, ref=branch)
            docs = chunk_code(file_path, content, repo, commit_id=commit_id)

            if docs:
                # Remove old chunks for this file first, then upsert new ones
                delete_by_files(repo, [file_path])
                count = upsert_documents(repo, docs)
                documents_count += count
                processed_files.append(file_path)
        except Exception as e:
            logger.error("push_process_file_error", file_path=file_path, error=str(e))
            errors.append(f"{file_path}: {str(e)}")

    result = {
        "repo": repo,
        "branch": branch,
        "files_processed": len(processed_files),
        "files_removed": len(event["removed"]),
        "chunks_upserted": documents_count,
        "changed_files": [
            {"filename": f, "status": "added" if f in event["added"] else "modified"}
            for f in processed_files
        ],
    }
    if errors:
        result["errors"] = errors

    logger.info("push_event_processed", **result)
    return result


async def ingest_full_repo(
    owner: str, repo_name: str, branch: str = "main"
) -> dict:
    """Ingest an entire repository: fetch all files, chunk, and embed.

    Returns ingestion statistics.
    """
    repo = f"{owner}/{repo_name}"
    client = GitHubClient()

    tree = await client.get_repo_tree(owner, repo_name, ref=branch)
    logger.info("ingest_start", repo=repo, branch=branch, total_files=len(tree))

    total_chunks = 0
    processed = 0
    errors: list[str] = []

    for item in tree:
        file_path = item["path"]
        try:
            content = await client.get_file_content(owner, repo_name, file_path, ref=branch)
            docs = chunk_code(file_path, content, repo)
            if docs:
                count = upsert_documents(repo, docs)
                total_chunks += count
            processed += 1
        except Exception as e:
            logger.error("ingest_file_error", file_path=file_path, error=str(e))
            errors.append(f"{file_path}: {str(e)}")

    result = {
        "repo": repo,
        "branch": branch,
        "files_processed": processed,
        "total_files": len(tree),
        "total_chunks": total_chunks,
    }
    if errors:
        result["errors"] = errors

    logger.info("ingest_complete", **result)
    return result
