from __future__ import annotations

from rag_platform.clients.memory_search import InMemorySearchIndex
from rag_platform.clients.protocols import SearchQuery
from rag_platform.domain import SourceDocument
from rag_platform.ingestion.chunker import ChunkingConfig
from rag_platform.ingestion.indexer import IndexSynchronizer
from rag_platform.ingestion.schema import VECTOR_FIELD, build_index

from conftest import FakeEmbedder

CONFIG = ChunkingConfig(max_tokens=80, overlap_tokens=10, embedding_signature="emb:3")


def policy(text: str, doc_id: str = "AP-9") -> SourceDocument:
    return SourceDocument(doc_id=doc_id, title="AP Rules", source_path="ap.md", text=text)


ORIGINAL = (
    "## Matching\n\nInvoices are matched to the PO and goods receipt.\n\n"
    "## Tolerances\n\nPrice variance tolerance is 2 percent or USD 100.\n\n"
    "## Holds\n\nBlocked invoices are resolved within 5 business days."
)


async def test_resync_of_unchanged_corpus_embeds_nothing() -> None:
    index, embedder = InMemorySearchIndex(), FakeEmbedder()
    synchronizer = IndexSynchronizer(index, embedder, CONFIG)

    first = await synchronizer.sync([policy(ORIGINAL)])
    second = await synchronizer.sync([policy(ORIGINAL)])

    assert (first.embedded, first.unchanged, first.deleted) == (3, 0, 0)
    assert (second.embedded, second.unchanged, second.deleted) == (0, 3, 0)
    assert len(embedder.calls) == 1
    assert len(index.documents) == 3


async def test_editing_one_section_reembeds_only_that_chunk_and_removes_stale() -> None:
    index, embedder = InMemorySearchIndex(), FakeEmbedder()
    synchronizer = IndexSynchronizer(index, embedder, CONFIG)
    await synchronizer.sync([policy(ORIGINAL)])
    before = set(index.documents)

    edited = ORIGINAL.replace("2 percent or USD 100", "1 percent or USD 50")
    report = await synchronizer.sync([policy(edited)])

    assert (report.embedded, report.unchanged, report.deleted) == (1, 2, 1)
    assert embedder.calls[-1] == [
        "AP Rules\nAP Rules > Tolerances\n\nPrice variance tolerance is 1 percent or USD 50."
    ]
    after = set(index.documents)
    assert len(before - after) == 1
    assert len(after - before) == 1
    assert all(doc[VECTOR_FIELD] for doc in index.documents.values())


async def test_reordering_sections_updates_positions_without_reembedding() -> None:
    index, embedder = InMemorySearchIndex(), FakeEmbedder()
    synchronizer = IndexSynchronizer(index, embedder, CONFIG)
    await synchronizer.sync([policy(ORIGINAL)])
    sections = ORIGINAL.split("\n\n## ")
    reordered = "\n\n## ".join([sections[0], sections[2], sections[1]])

    report = await synchronizer.sync([policy(reordered)])

    assert report.embedded == 0
    by_index = sorted(index.documents.values(), key=lambda d: int(d["chunk_index"]))
    assert "Blocked invoices" in str(by_index[1]["content"])


async def test_documents_are_isolated_by_doc_id() -> None:
    index = InMemorySearchIndex()
    synchronizer = IndexSynchronizer(index, FakeEmbedder(), CONFIG)
    await synchronizer.sync([policy(ORIGINAL, "AP-9"), policy("## Other\n\nText.", "AP-10")])

    report = await synchronizer.sync([policy(ORIGINAL, "AP-9")])

    assert report.deleted == 0
    assert {d["doc_id"] for d in index.documents.values()} == {"AP-9", "AP-10"}


async def test_in_memory_hybrid_search_ranks_keyword_matches_first() -> None:
    index = InMemorySearchIndex()
    await IndexSynchronizer(index, FakeEmbedder(), CONFIG).sync([policy(ORIGINAL)])

    hits = await index.hybrid_search(
        SearchQuery(keyword_text="price variance tolerance", semantic_text="", vector=(), top=2)
    )

    assert "Price variance" in hits[0].content
    assert hits[0].heading_path == "AP Rules > Tolerances"


def test_index_schema_is_hybrid_and_semantic() -> None:
    index = build_index("policies-v1", 1536, "policies-semantic")

    fields = {f.name: f for f in index.fields}
    assert fields["id"].key
    assert fields[VECTOR_FIELD].vector_search_dimensions == 1536
    assert fields[VECTOR_FIELD].retrievable is False
    assert fields["doc_id"].filterable
    assert fields["content"].searchable
    assert index.semantic_search is not None
    assert index.semantic_search.default_configuration_name == "policies-semantic"
