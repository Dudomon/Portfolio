"""`rag-eval`: run the golden-set evaluation, write a Markdown report and gate on thresholds.

Exit codes: 0 when every threshold passes, 1 when any fails. CI uses the exit code.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from rag_platform.agents.orchestrator import Orchestrator
from rag_platform.agents.prompts import PROMPT_VERSION
from rag_platform.clients.memory_search import InMemorySearchIndex
from rag_platform.clients.protocols import EmbeddingResult
from rag_platform.config import Settings, get_settings
from rag_platform.domain import TokenUsage
from rag_platform.evaluation.dataset import GoldenItem, load_golden, load_thresholds
from rag_platform.evaluation.judge import LlmJudge, load_rubrics
from rag_platform.evaluation.metrics import threshold_failures
from rag_platform.evaluation.report import render_report
from rag_platform.evaluation.runner import SuiteResult, run_full_suite, run_retrieval_suite
from rag_platform.ingestion.chunker import ChunkingConfig
from rag_platform.ingestion.indexer import IndexSynchronizer
from rag_platform.ingestion.loaders import load_documents
from rag_platform.wiring import azure_runtime


class _NoVectors:
    """Embedder for the offline baseline: empty vectors make retrieval keyword-only."""

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        return EmbeddingResult(vectors=[[] for _ in texts], usage=TokenUsage())


async def _offline_retrieval(items: list[GoldenItem], corpus: Path, k: int) -> SuiteResult:
    index = InMemorySearchIndex()
    embedder = _NoVectors()
    await IndexSynchronizer(index, embedder, ChunkingConfig()).sync(load_documents(corpus))
    return await run_retrieval_suite(items, index, embedder, k)


async def _azure_suite(
    items: list[GoldenItem], settings: Settings, mode: str, k: int, rubrics_dir: Path
) -> tuple[SuiteResult, dict[str, str]]:
    async with azure_runtime(settings) as runtime:
        if mode == "retrieval":
            return await run_retrieval_suite(items, runtime.search, runtime.embedder, k), {}
        judge = LlmJudge(runtime.chat, settings.judge_deployment, load_rubrics(rubrics_dir))
        orchestrator = Orchestrator(runtime.chat, runtime.embedder, runtime.search, settings)
        suite = await run_full_suite(items, orchestrator.answer, judge, k)
        return suite, {
            "rubrics": ", ".join(r.label for r in judge.rubrics.values()),
            "judge deployment": settings.judge_deployment,
        }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    parser.add_argument("--golden", type=Path, default=Path("evals/golden.jsonl"))
    parser.add_argument("--thresholds", type=Path, default=Path("evals/thresholds.toml"))
    parser.add_argument("--rubrics", type=Path, default=Path("evals/rubrics"))
    parser.add_argument("--corpus", type=Path, default=Path("data/sample_corpus"))
    parser.add_argument("--out", type=Path, default=Path("evals/reports/latest.md"))
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Retrieval mode only: keyword (BM25) baseline over the local corpus, no Azure calls.",
    )
    args = parser.parse_args(argv)
    if args.offline and args.mode != "retrieval":
        parser.error("--offline is only available with --mode retrieval")

    settings = get_settings()
    items = load_golden(args.golden)
    thresholds = load_thresholds(args.thresholds)
    metadata = {
        "mode": args.mode,
        "backend": "offline BM25 baseline"
        if args.offline
        else "Azure AI Search (hybrid + semantic)",
        "golden set": f"{args.golden.as_posix()} ({len(items)} items)",
        "k": str(args.k),
        "prompt version": PROMPT_VERSION,
    }
    if args.offline:
        suite = asyncio.run(_offline_retrieval(items, args.corpus, args.k))
    else:
        suite, extra = asyncio.run(_azure_suite(items, settings, args.mode, args.k, args.rubrics))
        metadata |= extra
        metadata["chat deployment"] = settings.chat_deployment

    # Threshold keys use `recall_at_5`-style names; only gates for this mode apply, and
    # recall gates only apply when the run used the same k.
    active = dict(thresholds.retrieval)
    if args.mode == "full":
        active |= thresholds.generation
    active = {m: v for m, v in active.items() if m in suite.summary()}
    failures = threshold_failures(suite.summary(), active)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_report(suite, active, failures, metadata), encoding="utf-8")
    sys.stdout.write(f"Report written to {args.out}\n")
    if failures:
        sys.stdout.write("Quality gate failed:\n" + "".join(f"  {f}\n" for f in failures))
        sys.exit(1)
