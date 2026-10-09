"""Answer agent: drafts (and revises) a cited answer from the evidence set."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from rag_platform.agents import prompts
from rag_platform.agents.citations import resolve_citations
from rag_platform.agents.gateway import ModelGateway
from rag_platform.agents.messages import Draft, EvidenceSet, Verification
from rag_platform.clients.protocols import ChatMessage, ChatRequest, StreamCompleted
from rag_platform.safety.injection import neutralize, render_sources


def _feedback(verification: Verification) -> str:
    lines = [f"- Claim: {c.claim}\n  Problem: {c.reason}" for c in verification.unsupported]
    return "<feedback>\n" + "\n".join(lines) + "\n</feedback>"


class AnswerAgent:
    name = "answer"

    def __init__(self, deployment: str, max_output_tokens: int = 800) -> None:
        self._deployment = deployment
        self._max_output_tokens = max_output_tokens

    def _request(
        self,
        question: str,
        evidence: EvidenceSet,
        revision_of: tuple[Draft, Verification] | None,
    ) -> ChatRequest:
        user = f"{render_sources(evidence.sources)}\n\nQuestion: {neutralize(question)}"
        messages = [
            ChatMessage(role="system", content=prompts.ANSWER),
            ChatMessage(role="user", content=user),
        ]
        if revision_of is not None:
            draft, verification = revision_of
            messages += [
                ChatMessage(role="assistant", content=draft.text),
                ChatMessage(
                    role="user",
                    content=f"{prompts.REVISION_INSTRUCTIONS}\n\n{_feedback(verification)}",
                ),
            ]
        return ChatRequest(
            deployment=self._deployment,
            operation=self.name if revision_of is None else f"{self.name}_revision",
            max_output_tokens=self._max_output_tokens,
            temperature=0.1,
            messages=tuple(messages),
        )

    async def draft(
        self,
        gateway: ModelGateway,
        question: str,
        evidence: EvidenceSet,
        attempt: int,
        revision_of: tuple[Draft, Verification] | None = None,
    ) -> AsyncGenerator[str | Draft, None]:
        """Yield text deltas as they arrive, then the complete `Draft` as the last item."""
        request = self._request(question, evidence, revision_of)
        async for event in gateway.stream(request):
            if isinstance(event, StreamCompleted):
                text = event.result.text.strip()
                yield Draft(
                    text=text,
                    citations=resolve_citations(text, evidence.sources),
                    attempt=attempt,
                )
            else:
                yield event.text
