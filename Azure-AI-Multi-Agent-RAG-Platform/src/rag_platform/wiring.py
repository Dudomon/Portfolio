"""Composition root: builds the Azure-backed collaborators and owns their lifetimes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from azure.core.credentials_async import AsyncTokenCredential
from azure.identity.aio import DefaultAzureCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient

from rag_platform.clients.azure_openai import AzureChatModel, AzureEmbedder, build_openai_client
from rag_platform.clients.azure_search import (
    AzureIndexStore,
    AzureSearchBackend,
    search_credential,
)
from rag_platform.clients.protocols import ChatModel, Embedder, IndexStore, SearchBackend
from rag_platform.config import Settings
from rag_platform.telemetry.genai import InstrumentedChatModel, InstrumentedEmbedder


@dataclass(frozen=True, slots=True)
class AzureRuntime:
    chat: ChatModel
    embedder: Embedder
    search: SearchBackend
    index_store: IndexStore


@asynccontextmanager
async def azure_runtime(settings: Settings) -> AsyncIterator[AzureRuntime]:
    async with AsyncExitStack() as stack:
        credential: AsyncTokenCredential | None = None
        if settings.azure_openai_api_key is None or settings.azure_search_api_key is None:
            # One credential instance is shared so its token cache is shared too.
            credential = await stack.enter_async_context(DefaultAzureCredential())
        openai_client = build_openai_client(settings, credential)
        stack.push_async_callback(openai_client.close)
        search_auth = search_credential(settings, credential)
        search_client = await stack.enter_async_context(
            SearchClient(settings.azure_search_endpoint, settings.search_index_name, search_auth)
        )
        index_client = await stack.enter_async_context(
            SearchIndexClient(settings.azure_search_endpoint, search_auth)
        )
        yield AzureRuntime(
            chat=InstrumentedChatModel(AzureChatModel(openai_client)),
            embedder=InstrumentedEmbedder(
                AzureEmbedder(
                    openai_client, settings.embedding_deployment, settings.embedding_dimensions
                ),
                settings.embedding_deployment,
            ),
            search=AzureSearchBackend(
                search_client,
                settings.semantic_configuration_name,
                settings.vector_k_nearest_neighbors,
            ),
            index_store=AzureIndexStore(index_client, search_client, settings),
        )
