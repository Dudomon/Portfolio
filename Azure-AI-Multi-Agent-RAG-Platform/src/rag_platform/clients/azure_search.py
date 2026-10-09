"""Azure AI Search adapters implementing `SearchBackend` and `IndexStore`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.models import QueryType, VectorizedQuery

from rag_platform.clients.protocols import SearchQuery
from rag_platform.config import Settings
from rag_platform.domain import RetrievedChunk
from rag_platform.ingestion.schema import RETRIEVABLE_FIELDS, VECTOR_FIELD, build_index


def search_credential(
    settings: Settings, credential: AsyncTokenCredential | None
) -> AzureKeyCredential | AsyncTokenCredential:
    if settings.azure_search_api_key is not None:
        return AzureKeyCredential(settings.azure_search_api_key.get_secret_value())
    if credential is None:
        raise ValueError("A token credential is required when no Azure AI Search API key is set")
    return credential


def _odata_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class AzureSearchBackend:
    """Hybrid retrieval: BM25 and HNSW vector search fused with RRF, then semantic reranking."""

    def __init__(
        self, client: SearchClient, semantic_configuration: str, k_nearest_neighbors: int
    ) -> None:
        self._client = client
        self._semantic_configuration = semantic_configuration
        self._k_nearest_neighbors = k_nearest_neighbors

    async def hybrid_search(self, query: SearchQuery) -> list[RetrievedChunk]:
        # The vector leg over-fetches (k=50) so RRF has enough candidates to fuse
        # before the semantic ranker reorders the top results.
        vector_query = VectorizedQuery(
            vector=list(query.vector),
            k_nearest_neighbors=self._k_nearest_neighbors,
            fields=VECTOR_FIELD,
        )
        results = await self._client.search(
            search_text=query.keyword_text,
            vector_queries=[vector_query],
            query_type=QueryType.SEMANTIC,
            semantic_configuration_name=self._semantic_configuration,
            semantic_query=query.semantic_text,
            filter=query.odata_filter,
            select=list(RETRIEVABLE_FIELDS),
            top=query.top,
        )
        hits: list[RetrievedChunk] = []
        async for document in results:
            hits.append(
                RetrievedChunk(
                    chunk_id=document["id"],
                    doc_id=document["doc_id"],
                    title=document["title"],
                    heading_path=document.get("heading_path") or "",
                    content=document["content"],
                    source_path=document["source_path"],
                    score=float(document["@search.score"]),
                    reranker_score=document.get("@search.reranker_score"),
                )
            )
        return hits


class AzureIndexStore:
    def __init__(
        self, index_client: SearchIndexClient, search_client: SearchClient, settings: Settings
    ) -> None:
        self._index_client = index_client
        self._search_client = search_client
        self._settings = settings

    async def ensure_index(self) -> None:
        # create_or_update is idempotent; additive schema changes apply in place, while
        # breaking ones (vector dimensions, analyzers) require a new versioned index name.
        await self._index_client.create_or_update_index(
            build_index(
                self._settings.search_index_name,
                self._settings.embedding_dimensions,
                self._settings.semantic_configuration_name,
            )
        )

    async def chunk_hashes(self, doc_id: str) -> dict[str, str]:
        results = await self._search_client.search(
            search_text="*",
            filter=f"doc_id eq {_odata_literal(doc_id)}",
            select=["id", "content_hash"],
            top=1000,
        )
        return {document["id"]: document["content_hash"] async for document in results}

    async def upsert(self, documents: Sequence[Mapping[str, Any]]) -> None:
        if not documents:
            return
        results = await self._search_client.merge_or_upload_documents([dict(d) for d in documents])
        failed = [r.key for r in results if not r.succeeded]
        if failed:
            raise RuntimeError(f"Failed to index {len(failed)} chunks: {failed[:5]}")

    async def delete(self, chunk_ids: Sequence[str]) -> None:
        if chunk_ids:
            await self._search_client.delete_documents([{"id": c} for c in chunk_ids])
