"""Evaluation suite runner: retrieval-only and end-to-end modes."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from rag_platform.agents.citations import is_abstention
from rag_platform.agents.messages import AnswerResult
from rag_platform.clients.protocols import Embedder, SearchBackend, SearchQuery
from rag_platform.evaluation.dataset import GoldenItem
from rag_platform.evaluation.judge import LlmJudge
from rag_platform.evaluation.metrics import (
    citation_precision,
    citation_recall,
    mean,
    recall_at_k,
    reciprocal_rank,
)

Mode = Literal["retrieval", "full"]


class ItemResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    category: str
    answerable: bool
    retrieved_doc_ids: tuple[str, ...]
    recall_at_k: float | None = None
    reciprocal_rank: float | None = None
    status: str | None = None
    answer: str | None = None
    cited_doc_ids: tuple[str, ...] = ()
    citation_precision: float | None = None
    citation_recall: float | None = None
    groundedness: float | None = None
    answer_relevance: float | None = None
    abstained: bool | None = None
    revisions: int | None = None
    cost_usd: float | None = None
    latency_seconds: float | None = None
    judge_notes: tuple[str, ...] = ()


class SuiteResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: Mode
    k: int
    items: tuple[ItemResult, ...]

    def summary(self) -> dict[str, float | None]:
        answerable = [i for i in self.items if i.answerable]
        unanswerable = [i for i in self.items if not i.answerable]
        summary: dict[str, float | None] = {
            f"recall_at_{self.k}": mean(i.recall_at_k for i in answerable),
            "mrr": mean(i.reciprocal_rank for i in answerable),
        }
        if self.mode == "full":
            summary |= {
                "citation_precision": mean(i.citation_precision for i in answerable),
                "citation_recall": mean(i.citation_recall for i in answerable),
                "groundedness": mean(i.groundedness for i in answerable),
                "answer_relevance": mean(i.answer_relevance for i in answerable),
                "abstention_accuracy": mean(
                    float(i.abstained) if i.abstained is not None else None for i in unanswerable
                ),
                "mean_cost_usd": mean(i.cost_usd for i in self.items),
                "p95_latency_seconds": _percentile([i.latency_seconds for i in self.items], 0.95),
            }
        return summary


def _percentile(values: Sequence[float | None], fraction: float) -> float | None:
    present = sorted(v for v in values if v is not None)
    if not present:
        return None
    index = min(len(present) - 1, round(fraction * (len(present) - 1)))
    return present[index]


async def _bounded[T](
    items: Sequence[GoldenItem], concurrency: int, work: Callable[[GoldenItem], Awaitable[T]]
) -> list[T]:
    semaphore = asyncio.Semaphore(concurrency)

    async def run(item: GoldenItem) -> T:
        async with semaphore:
            return await work(item)

    return list(await asyncio.gather(*(run(item) for item in items)))


def _retrieval_scores(
    item: GoldenItem, ranked: Sequence[str], k: int
) -> tuple[float | None, float | None]:
    """Recall@k and reciprocal rank; undefined (None) for unanswerable items."""
    if not item.answerable:
        return None, None
    return (
        recall_at_k(ranked, item.expected_doc_ids, k),
        reciprocal_rank(ranked, item.expected_doc_ids),
    )


async def run_retrieval_suite(
    items: Sequence[GoldenItem],
    search: SearchBackend,
    embedder: Embedder,
    k: int,
    concurrency: int = 4,
) -> SuiteResult:
    """Score retrieval alone with the raw question as the query (no planner or rewriting).

    This isolates index and ranking quality from the agents: if recall drops here, the
    cause is ingestion, chunking or index configuration, not prompting.
    """

    async def evaluate(item: GoldenItem) -> ItemResult:
        embedding = await embedder.embed([item.question])
        hits = await search.hybrid_search(
            SearchQuery(
                keyword_text=item.question,
                semantic_text=item.question,
                vector=tuple(embedding.vectors[0]),
                top=k,
            )
        )
        ranked = tuple(hit.doc_id for hit in hits)
        recall, rank_score = _retrieval_scores(item, ranked, k)
        return ItemResult(
            id=item.id,
            category=item.category,
            answerable=item.answerable,
            retrieved_doc_ids=ranked,
            recall_at_k=recall,
            reciprocal_rank=rank_score,
        )

    return SuiteResult(
        mode="retrieval", k=k, items=tuple(await _bounded(items, concurrency, evaluate))
    )


async def run_full_suite(
    items: Sequence[GoldenItem],
    answer: Callable[[str], Awaitable[AnswerResult]],
    judge: LlmJudge,
    k: int,
    concurrency: int = 2,
) -> SuiteResult:
    async def evaluate(item: GoldenItem) -> ItemResult:
        started = time.perf_counter()
        result = await answer(item.question)
        latency = time.perf_counter() - started
        ranked = tuple(source.doc_id for source in result.sources)
        recall, rank_score = _retrieval_scores(item, ranked, k)
        cited = tuple(dict.fromkeys(c.doc_id for c in result.citations))
        abstained = result.status == "insufficient_evidence" or (
            not result.citations and is_abstention(result.answer)
        )
        groundedness = relevance = None
        notes: list[str] = []
        if item.answerable and result.answer and not abstained:
            sources_text = "\n\n".join(
                f"[{s.source_number}] {s.title} > {s.heading_path}\n{s.content}"
                for s in result.sources
            )
            grounded = await judge.score(
                "groundedness", question=item.question, answer=result.answer, sources=sources_text
            )
            relevant = await judge.score(
                "answer_relevance",
                question=item.question,
                answer=result.answer,
                reference_answer=item.reference_answer,
            )
            groundedness, relevance = grounded.normalized, relevant.normalized
            notes = [f"groundedness: {grounded.rationale}", f"relevance: {relevant.rationale}"]
        elif item.answerable:
            # Abstaining on an answerable question is a failure of both metrics.
            groundedness, relevance = None, 0.0
            notes = ["abstained on an answerable question"]

        return ItemResult(
            id=item.id,
            category=item.category,
            answerable=item.answerable,
            retrieved_doc_ids=ranked,
            status=result.status,
            answer=result.answer,
            cited_doc_ids=cited,
            citation_precision=citation_precision(cited, item.expected_doc_ids)
            if item.answerable
            else None,
            citation_recall=citation_recall(cited, item.expected_doc_ids)
            if item.answerable
            else None,
            groundedness=groundedness,
            answer_relevance=relevance,
            abstained=abstained if not item.answerable else None,
            revisions=result.revisions,
            cost_usd=result.cost_usd,
            latency_seconds=round(latency, 3),
            judge_notes=tuple(notes),
            recall_at_k=recall,
            reciprocal_rank=rank_score,
        )

    return SuiteResult(mode="full", k=k, items=tuple(await _bounded(items, concurrency, evaluate)))
