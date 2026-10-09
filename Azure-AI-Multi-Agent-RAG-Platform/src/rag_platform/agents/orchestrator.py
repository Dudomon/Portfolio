"""Explicit multi-agent orchestration: plan -> retrieve -> answer -> verify (-> revise once).

The control flow is plain Python rather than a graph framework on purpose: the flow is
short, the loop is bounded, and every transition is visible in one function, which is
what reviewers, on-call engineers and auditors need to reason about the system.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import aclosing, suppress
from dataclasses import dataclass, field

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from rag_platform.agents import prompts
from rag_platform.agents.answerer import AnswerAgent
from rag_platform.agents.budget import BudgetExhaustedError, TokenBudget
from rag_platform.agents.citations import as_citation
from rag_platform.agents.critic import CriticAgent
from rag_platform.agents.gateway import AgentOutputError, ModelGateway
from rag_platform.agents.messages import (
    AgentEvent,
    AnswerDeltaEvent,
    AnswerResult,
    AnswerStatus,
    CompletedEvent,
    Draft,
    EvidenceSet,
    Plan,
    PlanEvent,
    RetrievalEvent,
    Verification,
    VerificationEvent,
)
from rag_platform.agents.planner import PlannerAgent
from rag_platform.agents.retriever import RetrievalSettings, RetrieverAgent
from rag_platform.clients.protocols import ChatModel, Embedder, SearchBackend
from rag_platform.config import Settings
from rag_platform.safety.content_filter import ContentFilteredError
from rag_platform.telemetry.cost import CostLedger

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer("rag_platform.agents")

Emit = Callable[[AgentEvent], None]


@dataclass(slots=True)
class _RunState:
    """Mutable bookkeeping for one request, used to build a result on any exit path."""

    plan: Plan | None = None
    evidence: EvidenceSet | None = None
    latest_draft: Draft | None = None
    verified: tuple[Draft, Verification] | None = None
    revisions: int = 0
    warnings: list[str] = field(default_factory=list)


class Orchestrator:
    def __init__(
        self,
        chat: ChatModel,
        embedder: Embedder,
        search: SearchBackend,
        settings: Settings,
    ) -> None:
        self._chat = chat
        self._embedder = embedder
        self._settings = settings
        self.planner = PlannerAgent(settings.utility_deployment, settings.max_sub_questions)
        self.retriever = RetrieverAgent(
            search,
            RetrievalSettings(
                deployment=settings.utility_deployment,
                top_k=settings.retrieval_top_k,
                min_reranker_score=settings.min_reranker_score,
                max_evidence_chunks=settings.max_evidence_chunks,
            ),
        )
        self.answerer = AnswerAgent(settings.chat_deployment)
        self.critic = CriticAgent(settings.utility_deployment)

    async def answer(self, question: str) -> AnswerResult:
        """Run the full pipeline and return only the final result."""
        return await self._execute(question, lambda _event: None)

    async def stream(self, question: str) -> AsyncGenerator[AgentEvent, None]:
        """Run the pipeline in a background task and yield its events as they happen.

        The pipeline runs in its own task so that the request deadline applies to the
        work itself, not to how fast the client reads. If the consumer goes away (client
        disconnect closes this generator), the task is canceled, which cancels any
        in-flight model or search call.
        """
        # Unbounded is acceptable: events per request are few and small.
        queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()

        async def produce() -> None:
            try:
                await self._execute(question, queue.put_nowait)
            finally:
                queue.put_nowait(None)

        task = asyncio.create_task(produce())
        try:
            while (event := await queue.get()) is not None:
                yield event
            await task  # surfaces unexpected errors raised by the pipeline
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    async def _execute(self, question: str, emit: Emit) -> AnswerResult:
        ledger = CostLedger(self._settings.model_prices)
        gateway = ModelGateway(
            self._chat,
            self._embedder,
            TokenBudget(self._settings.request_token_budget, ledger),
            self._settings.embedding_deployment,
        )
        state = _RunState()
        started = time.perf_counter()
        with _tracer.start_as_current_span("rag.request") as span:
            span.set_attribute("rag.prompt_version", prompts.PROMPT_VERSION)
            span.set_attribute("rag.token_budget", self._settings.request_token_budget)
            try:
                async with asyncio.timeout(self._settings.request_timeout_seconds):
                    status = await self._pipeline(gateway, question, state, emit)
            except BudgetExhaustedError as error:
                status = "budget_exhausted"
                state.warnings.append(str(error))
            except TimeoutError:
                status = "timeout"
                state.warnings.append("The request deadline was reached before completion.")
            except ContentFilteredError as error:
                status = "blocked"
                span.set_attribute("rag.content_filter.stage", error.stage)
                state.warnings.append(f"Content filter triggered on the {error.stage}.")

            result = self._result(question, status, state, ledger, span)
            span.set_attributes(
                {
                    "rag.status": result.status,
                    "rag.revisions": result.revisions,
                    "rag.tokens.total": result.usage.total,
                    "rag.cost_usd": result.cost_usd,
                }
            )
            logger.info(
                "request completed",
                extra={
                    "status": result.status,
                    "revisions": result.revisions,
                    "tokens": result.usage.total,
                    "cost_usd": result.cost_usd,
                    "duration_ms": round((time.perf_counter() - started) * 1000),
                },
            )
        emit(CompletedEvent(result=result))
        return result

    async def _step[T](
        self, name: str, work: Awaitable[T], span_hook: Callable[[Span, T], None]
    ) -> T:
        with _tracer.start_as_current_span(f"invoke_agent {name}") as span:
            span.set_attributes(
                {
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.agent.name": name,
                    "rag.prompt_version": prompts.PROMPT_VERSION,
                }
            )
            try:
                async with asyncio.timeout(self._settings.step_timeout_seconds):
                    value = await work
            except Exception as error:
                span.set_attribute("error.type", type(error).__name__)
                span.set_status(Status(StatusCode.ERROR, str(error)))
                raise
            span_hook(span, value)
            return value

    async def _pipeline(
        self, gateway: ModelGateway, question: str, state: _RunState, emit: Emit
    ) -> AnswerStatus:
        state.plan = await self._step(
            "planner",
            self.planner.plan(gateway, question),
            lambda span, plan: span.set_attributes(
                {"rag.sub_questions": len(plan.sub_questions), "rag.plan_fallback": plan.fallback}
            ),
        )
        emit(PlanEvent(plan=state.plan))

        state.evidence = await self._step(
            "retriever",
            self.retriever.retrieve(gateway, state.plan),
            lambda span, evidence: span.set_attributes(
                {
                    "rag.sources": len(evidence.sources),
                    "rag.withheld": len(evidence.withheld_chunk_ids),
                }
            ),
        )
        evidence = state.evidence
        emit(
            RetrievalEvent(
                source_count=len(evidence.sources),
                withheld_count=len(evidence.withheld_chunk_ids),
            )
        )
        if not evidence.sources:
            # Skipping generation entirely is both cheaper and the strongest guarantee
            # that nothing is made up when retrieval comes back empty.
            return "insufficient_evidence"

        attempt = 0
        while True:
            draft = await self._step(
                "answer",
                self._draft(gateway, question, evidence, attempt, state.verified, emit),
                lambda span, d: span.set_attribute("rag.citations", len(d.citations)),
            )
            state.latest_draft = draft
            try:
                verification = await self._step(
                    "critic",
                    self.critic.verify(gateway, question, evidence, draft),
                    lambda span, v: span.set_attributes(
                        {
                            "rag.claims": len(v.claims),
                            "rag.grounded_ratio": v.grounded_ratio,
                        }
                    ),
                )
            except AgentOutputError:
                logger.warning("critic output invalid; returning the draft unverified")
                return "answered"
            state.verified = (draft, verification)
            needs_revision = (
                verification.grounded_ratio < self._settings.groundedness_threshold
                or bool(verification.unsupported)
            )
            will_revise = needs_revision and state.revisions < self._settings.max_revisions
            emit(
                VerificationEvent(
                    attempt=attempt,
                    grounded_ratio=verification.grounded_ratio,
                    unsupported_claims=tuple(c.claim for c in verification.unsupported),
                    will_revise=will_revise,
                )
            )
            if not will_revise:
                return "answered"
            state.revisions += 1
            attempt += 1

    async def _draft(
        self,
        gateway: ModelGateway,
        question: str,
        evidence: EvidenceSet,
        attempt: int,
        revision_of: tuple[Draft, Verification] | None,
        emit: Emit,
    ) -> Draft:
        draft: Draft | None = None
        # The stream is consumed to the end (and closed deterministically on errors) so
        # the budget reservation is settled and the model span ends here, not at GC time.
        stream = self.answerer.draft(gateway, question, evidence, attempt, revision_of)
        async with aclosing(stream) as items:
            async for item in items:
                if isinstance(item, Draft):
                    draft = item
                else:
                    emit(AnswerDeltaEvent(attempt=attempt, text=item))
        if draft is None:
            raise AgentOutputError("answer stream ended without a completed draft")
        return draft

    def _result(
        self,
        question: str,
        status: AnswerStatus,
        state: _RunState,
        ledger: CostLedger,
        span: Span,
    ) -> AnswerResult:
        # Prefer the newest draft that passed through verification; fall back to an
        # unverified one (flagged by `verification=None`) only when none was verified.
        draft, verification = (
            state.verified if state.verified is not None else (state.latest_draft, None)
        )
        if status == "insufficient_evidence":
            answer = prompts.INSUFFICIENT_EVIDENCE_ANSWER
        elif status == "blocked":
            answer, draft, verification = prompts.BLOCKED_ANSWER, None, None
        elif draft is not None:
            answer = draft.text
        else:
            answer = ""
        if draft is not None and verification is None:
            state.warnings.append("The returned answer has not been verified.")
        if ledger.unpriced_deployments:
            state.warnings.append(
                f"No price configured for: {', '.join(sorted(ledger.unpriced_deployments))}."
            )
        sources = tuple(
            as_citation(number, chunk)
            for number, chunk in enumerate(
                state.evidence.sources if state.evidence is not None else (), start=1
            )
        )
        context = span.get_span_context()
        return AnswerResult(
            question=question,
            status=status,
            answer=answer,
            citations=draft.citations if draft is not None else (),
            sources=sources,
            verification=verification,
            revisions=state.revisions,
            plan=state.plan,
            usage=ledger.total_usage,
            cost_usd=ledger.total_cost_usd,
            trace_id=format(context.trace_id, "032x") if context.is_valid else None,
            warnings=tuple(state.warnings),
        )
