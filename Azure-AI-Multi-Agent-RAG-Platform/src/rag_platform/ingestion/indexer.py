"""Idempotent synchronization of documents into the search index."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

from rag_platform.clients.protocols import Embedder, IndexStore
from rag_platform.domain import Chunk, SourceDocument, TokenUsage
from rag_platform.ingestion.chunker import ChunkingConfig, chunk_document, embedding_text
from rag_platform.ingestion.schema import to_search_document

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SyncReport:
    documents: int = 0
    chunks: int = 0
    embedded: int = 0
    unchanged: int = 0
    deleted: int = 0
    embedding_usage: TokenUsage = field(default_factory=TokenUsage)


class IndexSynchronizer:
    """Brings the index in line with the given documents, embedding only what changed.

    Chunk ids are content-addressed (see `chunker`), so for each document:
    new ids are embedded and uploaded, ids that disappeared are deleted, and surviving
    ids only get their position (`chunk_index`) refreshed. Running the same corpus
    twice performs zero embedding calls.
    """

    def __init__(
        self,
        store: IndexStore,
        embedder: Embedder,
        config: ChunkingConfig,
        embedding_batch_size: int = 16,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._config = config
        self._batch_size = embedding_batch_size

    async def sync(self, documents: Iterable[SourceDocument]) -> SyncReport:
        await self._store.ensure_index()
        report = SyncReport()
        for document in documents:
            await self._sync_document(document, report)
        return report

    async def _sync_document(self, document: SourceDocument, report: SyncReport) -> None:
        chunks = chunk_document(document, self._config)
        existing = await self._store.chunk_hashes(document.doc_id)
        fresh = [c for c in chunks if existing.get(c.chunk_id) != c.content_hash]
        surviving = [c for c in chunks if existing.get(c.chunk_id) == c.content_hash]
        current_ids = {c.chunk_id for c in chunks}
        stale = sorted(chunk_id for chunk_id in existing if chunk_id not in current_ids)

        for start in range(0, len(fresh), self._batch_size):
            batch = fresh[start : start + self._batch_size]
            report.embedding_usage += await self._embed_and_upload(batch)
        # Partial documents are merged by the index, so unchanged chunks keep their
        # vectors and only their ordinal is updated.
        await self._store.upsert(
            [{"id": c.chunk_id, "chunk_index": c.chunk_index} for c in surviving]
        )
        # Delete after uploading: briefly serving both versions is preferable to a
        # window in which the document has no chunks at all.
        await self._store.delete(stale)

        report.documents += 1
        report.chunks += len(chunks)
        report.embedded += len(fresh)
        report.unchanged += len(surviving)
        report.deleted += len(stale)
        logger.info(
            "document synchronized",
            extra={
                "doc_id": document.doc_id,
                "chunks": len(chunks),
                "embedded": len(fresh),
                "deleted": len(stale),
            },
        )

    async def _embed_and_upload(self, batch: list[Chunk]) -> TokenUsage:
        result = await self._embedder.embed(
            [embedding_text(c.title, c.heading_label, c.content) for c in batch]
        )
        if len(result.vectors) != len(batch):
            raise RuntimeError(
                f"Embedder returned {len(result.vectors)} vectors for {len(batch)} inputs"
            )
        await self._store.upsert(
            [to_search_document(c, v) for c, v in zip(batch, result.vectors, strict=True)]
        )
        return result.usage
