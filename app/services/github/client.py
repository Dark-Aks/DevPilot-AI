from __future__ import annotations

import base64
from typing import Any

import httpx

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_RETRY_STATUSES = {502, 503, 504, 429}
_MAX_RETRIES = 3


class GitHubClient:
    """Async GitHub REST API client with retry support."""

    def __init__(self, token: str | None = None, base_url: str | None = None):
        self._token = token or settings.github_token
        self._base_url = (base_url or settings.github_api_base).rstrip("/")
        self._headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._token:
            self._headers["Authorization"] = f"Bearer {self._token}"

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers,
            timeout=30.0,
        )

    async def _request(self, method: str, path: str, **kwargs) -> Any:
        """Make an API request with retry on transient errors."""
        async with self._client() as client:
            for attempt in range(1, _MAX_RETRIES + 1):
                resp = await client.request(method, path, **kwargs)
                if resp.status_code not in _RETRY_STATUSES:
                    resp.raise_for_status()
                    return resp.json()
                if attempt < _MAX_RETRIES:
                    wait = 2**attempt
                    logger.warning(
                        "github_retry",
                        status=resp.status_code,
                        attempt=attempt,
                        wait=wait,
                    )
                    import asyncio
                    await asyncio.sleep(wait)
            resp.raise_for_status()
            return resp.json()

    # ── Repository operations ──

    async def get_repo_tree(
        self, owner: str, repo: str, ref: str = "main"
    ) -> list[dict]:
        """Get recursive file tree for a repo at a given ref."""
        data = await self._request("GET", f"/repos/{owner}/{repo}/git/trees/{ref}?recursive=1")
        return [
            item for item in data.get("tree", []) if item.get("type") == "blob"
        ]

    async def get_file_content(
        self, owner: str, repo: str, path: str, ref: str = "main"
    ) -> str:
        """Fetch raw file content from a repository."""
        data = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/contents/{path}",
            params={"ref": ref},
        )
        content = data.get("content", "")
        encoding = data.get("encoding", "base64")
        if encoding == "base64":
            return base64.b64decode(content).decode("utf-8", errors="replace")
        return content

    async def get_compare(
        self, owner: str, repo: str, base: str, head: str
    ) -> dict:
        """Compare two commits and return diff information."""
        return await self._request(
            "GET",
            f"/repos/{owner}/{repo}/compare/{base}...{head}",
        )

    async def post_pr_comment(
        self, owner: str, repo: str, pr_number: int, body: str
    ) -> dict:
        """Post a comment on a pull request."""
        return await self._request(
            "POST",
            f"/repos/{owner}/{repo}/issues/{pr_number}/comments",
            json={"body": body},
        )

    async def get_pr_for_commit(
        self, owner: str, repo: str, commit_sha: str
    ) -> int | None:
        """Find a PR number associated with a commit, if any."""
        try:
            data = await self._request(
                "GET",
                f"/repos/{owner}/{repo}/commits/{commit_sha}/pulls",
            )
            if data and len(data) > 0:
                return data[0].get("number")
        except httpx.HTTPStatusError:
            pass
        return None
