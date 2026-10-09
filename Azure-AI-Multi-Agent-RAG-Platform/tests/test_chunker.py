from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from rag_platform.domain import SourceDocument, estimate_tokens
from rag_platform.ingestion.chunker import ChunkingConfig, chunk_document
from rag_platform.ingestion.loaders import DocumentLoadError, load_documents, load_markdown

CONFIG = ChunkingConfig(max_tokens=60, overlap_tokens=20, embedding_signature="emb:1536")

SENTENCES = " ".join(
    f"Rule {i} requires approval from the controller before posting." for i in range(1, 13)
)


def document(text: str) -> SourceDocument:
    return SourceDocument(doc_id="FIN-9", title="Close Policy", source_path="fin.md", text=text)


def test_chunks_carry_heading_path_and_never_cross_sections() -> None:
    doc = document(
        "# Close Policy\n\nIntro paragraph about closing.\n\n"
        "## Accruals\n\nAccrue items of USD 10,000 or more.\n\n"
        "### Exceptions\n\nRecurring services are always accrued.\n\n"
        "## Journals\n\nManual journals need an approver."
    )

    chunks = chunk_document(doc, CONFIG)

    assert [c.heading_path for c in chunks] == [
        ("Close Policy",),
        ("Close Policy", "Accruals"),
        ("Close Policy", "Accruals", "Exceptions"),
        ("Close Policy", "Journals"),
    ]
    assert chunks[2].heading_label == "Close Policy > Accruals > Exceptions"
    assert [c.chunk_index for c in chunks] == [0, 1, 2, 3]


def test_long_sections_are_split_within_budget_with_sentence_overlap() -> None:
    chunks = chunk_document(document(f"## Approvals\n\n{SENTENCES}"), CONFIG)

    assert len(chunks) > 2
    assert all(c.token_estimate <= CONFIG.max_tokens for c in chunks)
    for previous, current in pairwise(chunks):
        last_unit = previous.content.split("\n\n")[-1]
        assert current.content.startswith(last_unit), "consecutive chunks must overlap"


def test_zero_overlap_produces_disjoint_chunks() -> None:
    config = ChunkingConfig(max_tokens=60, overlap_tokens=0)

    chunks = chunk_document(document(SENTENCES), config)

    combined = " ".join(c.content for c in chunks)
    assert combined.count("Rule 5 requires") == 1


def test_tables_and_fenced_code_are_kept_whole() -> None:
    table = "| Value | Approver |\n| --- | --- |\n| 5,000 | Manager |\n| 50,000 | CFO |"
    code = "```\n# not a heading\nPOST /journal\n```"
    chunks = chunk_document(document(f"## Limits\n\n{table}\n\n{code}"), CONFIG)

    assert any(table in c.content for c in chunks)
    assert any(code in c.content for c in chunks)
    assert all(c.heading_path == ("Close Policy", "Limits") for c in chunks)


def test_oversized_sentence_is_split_on_words() -> None:
    run_on = " ".join(["segregation"] * 200)

    chunks = chunk_document(document(run_on), CONFIG)

    assert len(chunks) > 1
    assert all(estimate_tokens(c.content) <= CONFIG.max_tokens for c in chunks)


def test_chunk_ids_are_content_addressed() -> None:
    original = "## A\n\nFirst section text.\n\n## B\n\nSecond section text."
    edited = "## A\n\nFirst section text, now edited.\n\n## B\n\nSecond section text."

    before = {c.heading_path: c.chunk_id for c in chunk_document(document(original), CONFIG)}
    after = {c.heading_path: c.chunk_id for c in chunk_document(document(edited), CONFIG)}

    assert before[("Close Policy", "B")] == after[("Close Policy", "B")]
    assert before[("Close Policy", "A")] != after[("Close Policy", "A")]


def test_embedding_signature_changes_every_hash() -> None:
    doc = document("## A\n\nSome text.")
    other = ChunkingConfig(max_tokens=60, overlap_tokens=20, embedding_signature="emb:3072")

    (first,) = chunk_document(doc, CONFIG)
    (second,) = chunk_document(doc, other)

    assert first.content_hash != second.content_hash


def test_duplicate_blocks_get_distinct_ids() -> None:
    chunks = chunk_document(document("## X\n\nNone.\n\n## X\n\nNone."), CONFIG)

    assert len({c.chunk_id for c in chunks}) == 2


@pytest.mark.parametrize(("max_tokens", "overlap"), [(0, 0), (50, 50), (50, -1)])
def test_invalid_config_is_rejected(max_tokens: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        ChunkingConfig(max_tokens=max_tokens, overlap_tokens=overlap)


def test_markdown_loader_reads_front_matter(tmp_path: Path) -> None:
    path = tmp_path / "policy.md"
    path.write_text(
        "---\ndoc_id: TAX-9\ntitle: Tax Rules\ncategory: tax\neffective_date: 2025-02-01\n---\n"
        "# Ignored Heading\n\nBody.",
        encoding="utf-8",
    )

    doc = load_markdown(path, tmp_path)

    assert (doc.doc_id, doc.title, doc.category) == ("TAX-9", "Tax Rules", "tax")
    assert doc.effective_date is not None
    assert doc.effective_date.isoformat() == "2025-02-01"
    assert doc.source_path == "policy.md"
    assert not doc.text.startswith("---")


def test_markdown_loader_rejects_bad_dates(tmp_path: Path) -> None:
    path = tmp_path / "bad.md"
    path.write_text("---\neffective_date: someday\n---\n# T\n", encoding="utf-8")

    with pytest.raises(DocumentLoadError):
        load_markdown(path, tmp_path)


def test_sample_corpus_loads_with_unique_ids(corpus_dir: Path) -> None:
    documents = list(load_documents(corpus_dir))

    assert len(documents) >= 8
    assert len({d.doc_id for d in documents}) == len(documents)
    for doc in documents:
        chunks = chunk_document(doc, ChunkingConfig())
        assert chunks, doc.doc_id
        assert all(c.token_estimate <= ChunkingConfig().max_tokens for c in chunks)
