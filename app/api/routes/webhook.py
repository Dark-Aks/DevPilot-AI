from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, Header, Request, status
from fastapi.responses import JSONResponse

from app.models.schemas import WebhookResponse
from app.services.webhook_handler import (
    process_push_event,
    verify_webhook_signature,
)
from app.agents.graph import run_workflow
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["webhook"])


async def _process_webhook(payload: dict) -> None:
    """Background task: update embeddings then run agent workflow."""
    try:
        push_result = await process_push_event(payload)

        if push_result.get("error"):
            logger.error("webhook_processing_error", error=push_result["error"])
            return

        # Run agent workflow on the changed files
        if push_result.get("changed_files"):
            repo = push_result["repo"]
            branch = push_result.get("branch", "main")
            changed_files = push_result["changed_files"]

            # Fetch diff summary for agents
            from app.services.client import GitHubClient

            client = GitHubClient()
            owner, repo_name = repo.split("/", 1)
            before = payload.get("before", "")
            after = payload.get("after", "")

            diff_text = ""
            if before and after and before != "0" * 40:
                try:
                    compare = await client.get_compare(owner, repo_name, before, after)
                    diff_text = "\n".join(
                        f.get("patch", "") for f in compare.get("files", []) if f.get("patch")
                    )
                except Exception as e:
                    logger.warning("diff_fetch_failed", error=str(e))

            workflow_result = await run_workflow(
                repo=repo,
                changed_files=changed_files,
                diff=diff_text,
            )

            # Optionally post as PR comment
            pr_number = await client.get_pr_for_commit(owner, repo_name, after)
            if pr_number:
                from app.utils.formatting import format_pr_comment

                comment = format_pr_comment(workflow_result)
                await client.post_pr_comment(owner, repo_name, pr_number, comment)
                logger.info("pr_comment_posted", repo=repo, pr=pr_number)

            logger.info("workflow_complete", repo=repo, agents_run=4)
    except Exception as e:
        logger.error("webhook_background_error", error=str(e), exc_info=True)


@router.post("/webhook/github", status_code=status.HTTP_202_ACCEPTED)
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: str = Header(default=""),
    x_github_event: str = Header(default=""),
) -> WebhookResponse:
    """Receive GitHub webhook events."""
    body = await request.body()

    # Verify signature
    if not verify_webhook_signature(body, x_hub_signature_256):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"status": "error", "message": "Invalid signature", "task_id": ""},
        )

    payload = await request.json()

    # Only process push events
    if x_github_event != "push":
        return WebhookResponse(
            status="ignored",
            message=f"Event '{x_github_event}' is not processed",
        )

    task_id = str(uuid.uuid4())
    background_tasks.add_task(_process_webhook, payload)

    logger.info(
        "webhook_received",
        github_event=x_github_event,
        repo=payload.get("repository", {}).get("full_name", ""),
        task_id=task_id,
    )

    return WebhookResponse(
        status="accepted",
        message="Push event processing started",
        task_id=task_id,
    )
