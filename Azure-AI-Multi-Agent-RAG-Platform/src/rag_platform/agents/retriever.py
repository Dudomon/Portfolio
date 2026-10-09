"""Retriever agent: query rewriting plus hybrid search for each sub-question."""

from __future__ import annotations

import asyncio
import itertools
import logging
from dataclasses import dataclass

from pydantic import BaseModel, Field

from rag_platform.agents import prompts
from rag_platform.agents.gateway import AgentOutputError, ModelGateway
from rag_platform.agents.messages import EvidenceSet, Plan, SubQuestion, SubQuestionEvidence
from rag_platform.clients.protocols import ChatMessage, ChatRequest, SearchBackend, SearchQuery
from rag_platform.domain import RetrievedChunk
from rag_platform.safety.injection import neutralize, screen

logger = logging.getLogger(__name__)


class _Rewrite(BaseModel):
    keyword_query: str = Field(min_length=1)
    semantic_query: str = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    deployment: str
    top_k: int
    min_reranker_score: float
    max_evidence_chunks: int


class RetrieverAgent:
    name = "retriever"

    def __init__(self, search: SearchBackend, settings: RetrievalSettings) -> None:
        self._search = search
        self._settings = settings

    async def retrieve(self, gateway: ModelGateway, plan: Plan) -> EvidenceSet:
        # Sub-questions are independent, so they are searched concurrently. The task
        # group cancels the siblings if one fails, and propagates cancellation.
        try:
            async with asyncio.TaskGroup() as group:
                tasks = [
                    group.create_task(self._retrieve_one(gateway, sub_question))
                    for sub_question in plan.sub_questions
                ]
        except ExceptionGroup as failures:
            # Callers handle domain errors (budget, content filter) by type; the first
            # failure is representative because its siblings were canceled by it.
            raise failures.exceptions[0] from failures
        results = [task.result() for task in tasks]

        merged = self._merge([hits for _, hits in results])
        screened = screen(merged)
        if screened.withheld:
            logger.warning(
                "withheld retrieved chunks that look like prompt injection",
                extra={"chunk_ids": ",".join(c.chunk_id for c in screened.withheld)},
            )
        return EvidenceSet(
            sources=screened.accepted,
            per_sub_question=tuple(evidence for evidence, _ in results),
            withheld_chunk_ids=tuple(c.chunk_id for c in screened.withheld),
        )

    async def _rewrite(self, gateway: ModelGateway, sub_question: SubQuestion) -> _Rewrite:
        request = ChatRequest(
            deployment=self._settings.deployment,
            operation="query_rewrite",
            max_output_tokens=150,
            json_output=True,
            messages=(
                ChatMessage(role="system", content=prompts.QUERY_REWRITER),
                ChatMessage(role="user", content=neutralize(sub_question.text)),
            ),
        )
        try:
            return await gateway.complete_json(request, _Rewrite)
        except AgentOutputError:
            return _Rewrite(keyword_query=sub_question.text, semantic_query=sub_question.text)

    async def _retrieve_one(
        self, gateway: ModelGateway, sub_question: SubQuestion
    ) -> tuple[SubQuestionEvidence, list[RetrievedChunk]]:
        rewrite = await self._rewrite(gateway, sub_question)
        embedding = await gateway.embed([rewrite.semantic_query], operation="query_embedding")
        hits = await self._search.hybrid_search(
            SearchQuery(
                keyword_text=rewrite.keyword_query,
                semantic_text=rewrite.semantic_query,
                vector=tuple(embedding.vectors[0]),
                top=self._settings.top_k,
            )
        )
        # The semantic reranker score (0-4) is calibrated across queries, unlike the
        # fused RRF score, so it is the right signal for an absolute relevance cutoff.
        relevant = [
            hit
            for hit in hits
            if hit.reranker_score is None or hit.reranker_score >= self._settings.min_reranker_score
        ]
        evidence = SubQuestionEvidence(
            sub_question_id=sub_question.id,
            keyword_query=rewrite.keyword_query,
            semantic_query=rewrite.semantic_query,
            chunk_ids=tuple(hit.chunk_id for hit in relevant),
        )
        return evidence, relevant

    def _merge(self, rankings: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
        """Interleave rankings round-robin so every sub-question contributes its best hits.

        Simple concatenation would let the first sub-question fill the evidence cap and
        starve the others, which shows up as half-answered comparison questions.
        """
        merged: dict[str, RetrievedChunk] = {}
        for hit in itertools.chain.from_iterable(itertools.zip_longest(*rankings)):
            if hit is not None and hit.chunk_id not in merged:
                merged[hit.chunk_id] = hit
            if len(merged) == self._settings.max_evidence_chunks:
                break
        return list(merged.values())
