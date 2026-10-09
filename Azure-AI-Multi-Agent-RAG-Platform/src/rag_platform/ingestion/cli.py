"""`rag-ingest`: load, chunk, embed and index a document corpus."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from rag_platform.config import Settings, get_settings
from rag_platform.ingestion.chunker import ChunkingConfig, chunk_document
from rag_platform.ingestion.indexer import IndexSynchronizer
from rag_platform.ingestion.loaders import load_documents
from rag_platform.telemetry.setup import configure_telemetry
from rag_platform.wiring import azure_runtime


def chunking_config(settings: Settings, max_tokens: int, overlap_tokens: int) -> ChunkingConfig:
    return ChunkingConfig(
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        embedding_signature=f"{settings.embedding_deployment}:{settings.embedding_dimensions}",
    )


def _dry_run(corpus: Path, config: ChunkingConfig) -> dict[str, object]:
    documents = list(load_documents(corpus))
    per_document = {d.doc_id: len(chunk_document(d, config)) for d in documents}
    return {
        "documents": len(documents),
        "chunks": sum(per_document.values()),
        "per_document": per_document,
    }


async def _ingest(corpus: Path, settings: Settings, config: ChunkingConfig) -> dict[str, object]:
    async with azure_runtime(settings) as runtime:
        synchronizer = IndexSynchronizer(runtime.index_store, runtime.embedder, config)
        report = await synchronizer.sync(load_documents(corpus))
    return {
        "documents": report.documents,
        "chunks": report.chunks,
        "embedded": report.embedded,
        "unchanged": report.unchanged,
        "deleted": report.deleted,
        "embedding_tokens": report.embedding_usage.input_tokens,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/sample_corpus"))
    parser.add_argument("--max-tokens", type=int, default=350)
    parser.add_argument("--overlap-tokens", type=int, default=60)
    parser.add_argument(
        "--dry-run", action="store_true", help="Load and chunk only; no Azure calls are made."
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    config = chunking_config(settings, args.max_tokens, args.overlap_tokens)
    if args.dry_run:
        summary = _dry_run(args.corpus, config)
    else:
        configure_telemetry(settings)
        summary = asyncio.run(_ingest(args.corpus, settings, config))
    print(json.dumps(summary, indent=2))
