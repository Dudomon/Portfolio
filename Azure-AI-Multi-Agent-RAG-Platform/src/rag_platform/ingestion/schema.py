"""Azure AI Search index definition and the chunk-to-document mapping."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    LexicalAnalyzerName,
    SearchableField,
    SearchField,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)

from rag_platform.domain import Chunk

# EDM type names are used directly: they are the stable wire contract, whereas the
# SDK's enum helpers changed shape between major versions.
_STRING = "Edm.String"
_INT32 = "Edm.Int32"
_VECTOR = "Collection(Edm.Single)"

VECTOR_FIELD = "content_vector"
VECTOR_PROFILE = "hnsw-cosine"
RETRIEVABLE_FIELDS = (
    "id",
    "doc_id",
    "title",
    "heading_path",
    "content",
    "source_path",
    "category",
)


def build_index(name: str, dimensions: int, semantic_configuration: str) -> SearchIndex:
    fields = [
        SimpleField(name="id", type=_STRING, key=True),
        SimpleField(name="doc_id", type=_STRING, filterable=True, facetable=True),
        SimpleField(name="chunk_index", type=_INT32, sortable=True),
        SearchableField(name="title", analyzer_name=LexicalAnalyzerName.EN_MICROSOFT),
        SearchableField(name="heading_path", analyzer_name=LexicalAnalyzerName.EN_MICROSOFT),
        # The Microsoft English analyzer lemmatizes ("invoices" -> "invoice"), which
        # matters for policy language more than the default standard analyzer does.
        SearchableField(name="content", analyzer_name=LexicalAnalyzerName.EN_MICROSOFT),
        SimpleField(name="source_path", type=_STRING),
        SimpleField(name="category", type=_STRING, filterable=True, facetable=True),
        SimpleField(name="content_hash", type=_STRING, filterable=True),
        SearchField(
            name=VECTOR_FIELD,
            type=_VECTOR,
            searchable=True,
            # Vectors are never returned to clients; `stored=False` also halves storage.
            retrievable=False,
            stored=False,
            vector_search_dimensions=dimensions,
            vector_search_profile_name=VECTOR_PROFILE,
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw",
                parameters=HnswParameters(
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                    metric=VectorSearchAlgorithmMetric.COSINE,
                ),
            )
        ],
        profiles=[VectorSearchProfile(name=VECTOR_PROFILE, algorithm_configuration_name="hnsw")],
    )
    semantic_search = SemanticSearch(
        default_configuration_name=semantic_configuration,
        configurations=[
            SemanticConfiguration(
                name=semantic_configuration,
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="title"),
                    content_fields=[SemanticField(field_name="content")],
                    keywords_fields=[SemanticField(field_name="heading_path")],
                ),
            )
        ],
    )
    return SearchIndex(
        name=name, fields=fields, vector_search=vector_search, semantic_search=semantic_search
    )


def to_search_document(chunk: Chunk, vector: Sequence[float]) -> dict[str, Any]:
    return {
        "id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "chunk_index": chunk.chunk_index,
        "title": chunk.title,
        "heading_path": chunk.heading_label,
        "content": chunk.content,
        "source_path": chunk.source_path,
        "category": chunk.category,
        "content_hash": chunk.content_hash,
        VECTOR_FIELD: list(vector),
    }
