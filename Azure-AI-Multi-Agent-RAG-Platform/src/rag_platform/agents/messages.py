"""Typed messages exchanged between agents and the events streamed to clients.

Every hand-off between agents is one of these immutable models, never a free-form
dict or string. That makes each agent independently testable and lets the
orchestrator log, trace and replay the intermediate state of a request.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from rag_platform.domain import RetrievedChunk, TokenUsage


class _Message(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SubQuestion(_Message):
    id: str
    text: str


class Plan(_Message):
    sub_questions: tuple[SubQuestion, ...] = Field(min_length=1)
    # True when the planner output was unusable and the original question is used as-is.
    fallback: bool = False


class SubQuestionEvidence(_Message):
    sub_question_id: str
    keyword_query: str
    semantic_query: str
    chunk_ids: tuple[str, ...]


class EvidenceSet(_Message):
    """Deduplicated sources shown to the answer and critic agents, numbered from 1."""

    sources: tuple[RetrievedChunk, ...]
    per_sub_question: tuple[SubQuestionEvidence, ...]
    withheld_chunk_ids: tuple[str, ...] = ()


class Citation(_Message):
    source_number: int
    chunk_id: str
    doc_id: str
    title: str
    heading_path: str
    source_path: str
    # The excerpt itself, so clients can display it and judges can check against it.
    content: str


class Draft(_Message):
    text: str
    citations: tuple[Citation, ...]
    attempt: int


class ClaimVerdict(_Message):
    claim: str
    source_numbers: tuple[int, ...]
    supported: bool
    reason: str


class Verification(_Message):
    claims: tuple[ClaimVerdict, ...]

    @property
    def unsupported(self) -> tuple[ClaimVerdict, ...]:
        return tuple(c for c in self.claims if not c.supported)

    @property
    def grounded_ratio(self) -> float:
        if not self.claims:
            return 1.0
        return sum(c.supported for c in self.claims) / len(self.claims)


AnswerStatus = Literal[
    "answered",
    "insufficient_evidence",
    "blocked",
    "budget_exhausted",
    "timeout",
]


class AnswerResult(_Message):
    question: str
    status: AnswerStatus
    answer: str
    citations: tuple[Citation, ...] = ()
    # Every source shown to the answer agent, in retrieval order; `citations` is the
    # subset the answer actually cites. Also used to score retrieval in evaluations.
    sources: tuple[Citation, ...] = ()
    verification: Verification | None = None
    revisions: int = 0
    plan: Plan | None = None
    usage: TokenUsage = TokenUsage()
    cost_usd: float = 0.0
    trace_id: str | None = None
    warnings: tuple[str, ...] = ()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def grounded(self) -> bool:
        """True only when the returned answer was verified and every claim is supported."""
        return self.verification is not None and not self.verification.unsupported


class PlanEvent(_Message):
    type: Literal["plan"] = "plan"
    plan: Plan


class RetrievalEvent(_Message):
    type: Literal["retrieval"] = "retrieval"
    source_count: int
    withheld_count: int


class AnswerDeltaEvent(_Message):
    """Incremental answer text. A later attempt supersedes earlier ones (see `attempt`)."""

    type: Literal["answer_delta"] = "answer_delta"
    attempt: int
    text: str


class VerificationEvent(_Message):
    type: Literal["verification"] = "verification"
    attempt: int
    grounded_ratio: float
    unsupported_claims: tuple[str, ...]
    will_revise: bool


class CompletedEvent(_Message):
    type: Literal["completed"] = "completed"
    result: AnswerResult


AgentEvent = Annotated[
    PlanEvent | RetrievalEvent | AnswerDeltaEvent | VerificationEvent | CompletedEvent,
    Field(discriminator="type"),
]
