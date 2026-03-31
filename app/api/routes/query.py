from __future__ import annotations

from fastapi import APIRouter, status

from app.models.schemas import QueryRequest, QueryResponse, CodeChunkResult
from app.services.rag.retriever import retrieve
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_codebase(request: QueryRequest) -> QueryResponse:
    """Semantic search over an ingested codebase."""
    filter_metadata: dict | None = None
    if request.filter_language or request.filter_file_path:
        filter_metadata = {}
        if request.filter_language:
            filter_metadata["language"] = request.filter_language
        if request.filter_file_path:
            filter_metadata["file_path"] = request.filter_file_path

    documents = retrieve(
        query=request.query,
        repo=request.repo,
        top_k=request.top_k,
        filter_metadata=filter_metadata,
        use_reranking=True,
        use_cache=True,
    )

    results = [
        CodeChunkResult(
            file_path=doc.metadata.get("file_path", ""),
            function_name=doc.metadata.get("function_name", ""),
            class_name=doc.metadata.get("class_name", ""),
            chunk_type=doc.metadata.get("chunk_type", ""),
            language=doc.metadata.get("language", ""),
            start_line=doc.metadata.get("start_line", 0),
            end_line=doc.metadata.get("end_line", 0),
            content=doc.page_content,
        )
        for doc in documents
    ]

    return QueryResponse(
        query=request.query,
        repo=request.repo,
        results=results,
        total=len(results),
    )
