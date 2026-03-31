"""
Vector Store — DevPilot AI

Manages ChromaDB persistence for code embeddings with:
  - Per-repository collection isolation
  - Batch upsert with configurable batch sizes (avoids OOM on large repos)
  - Deduplication by chunk_id
  - File-level deletion for incremental updates

Scaling notes:
  - ChromaDB file persistence works for single-process deployments
  - For multi-process, switch to ChromaDB client-server mode or Pinecone
  - Batch size is configurable via settings.embedding_batch_size
"""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import settings
from app.rag.embeddings import get_embedding_function
from app.utils.logging import get_logger
from app.utils.metrics import track_latency

logger = get_logger(__name__)


def _collection_name(repo: str) -> str:
    """Derive a ChromaDB collection name from a repo identifier."""
    safe = repo.replace("/", "_").replace("-", "_").lower()
    return f"{settings.chroma_collection_prefix}_{safe}"


def get_vectorstore(repo: str) -> Chroma:
    """Get or create a Chroma vectorstore for the given repository."""
    return Chroma(
        collection_name=_collection_name(repo),
        persist_directory=str(settings.chroma_path),
        embedding_function=get_embedding_function(),
    )


def upsert_documents(repo: str, documents: list[Document]) -> int:
    """Upsert documents into the vector store in batches.

    Batching prevents OOM errors when embedding large repositories and
    enables progress tracking. Documents are deduplicated by chunk_id.

    Returns the number of documents upserted.
    """
    if not documents:
        return 0

    store = get_vectorstore(repo)
    batch_size = settings.embedding_batch_size
    total_upserted = 0

    with track_latency("vectorstore_upsert", repo=repo, total_docs=len(documents)):
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            ids = [doc.metadata["chunk_id"] for doc in batch]
            texts = [doc.page_content for doc in batch]
            metadatas = [doc.metadata for doc in batch]

            store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            total_upserted += len(batch)

            if len(documents) > batch_size:
                logger.info(
                    "batch_upsert_progress",
                    repo=repo,
                    batch=i // batch_size + 1,
                    total_batches=(len(documents) + batch_size - 1) // batch_size,
                    docs_so_far=total_upserted,
                )

    logger.info("upserted_documents", repo=repo, count=total_upserted)
    return total_upserted


def delete_by_files(repo: str, file_paths: list[str]) -> None:
    """Remove all chunks associated with the given file paths.

    Called during incremental updates (webhook push events) to remove
    stale embeddings before upserting new ones for modified files.
    """
    if not file_paths:
        return

    store = get_vectorstore(repo)
    for file_path in file_paths:
        try:
            store._collection.delete(where={"file_path": file_path})
        except Exception as e:
            logger.warning("delete_chunks_error", file_path=file_path, error=str(e))

    logger.info("deleted_file_chunks", repo=repo, file_count=len(file_paths))
