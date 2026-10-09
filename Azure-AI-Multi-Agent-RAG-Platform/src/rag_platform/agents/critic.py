"""Critic agent: verifies that each claim in a draft is supported by its cited sources."""

from __future__ import annotations

import html

from pydantic import BaseModel, Field

from rag_platform.agents import prompts
from rag_platform.agents.citations import invalid_numbers, uncited_statements
from rag_platform.agents.gateway import ModelGateway
from rag_platform.agents.messages import ClaimVerdict, Draft, EvidenceSet, Verification
from rag_platform.clients.protocols import ChatMessage, ChatRequest
from rag_platform.safety.injection import neutralize, render_sources


class _Claim(BaseModel):
    claim: str
    sources: list[int] = Field(default_factory=list)
    supported: bool
    reason: str = ""


class _CriticOutput(BaseModel):
    claims: list[_Claim]


class CriticAgent:
    """Combines deterministic citation checks with an LLM groundedness judgment.

    The deterministic findings are added on top of the model's verdicts and can only
    make the verdict stricter, so a persuasive but uncited draft cannot pass.
    """

    name = "critic"

    def __init__(self, deployment: str, max_output_tokens: int = 900) -> None:
        self._deployment = deployment
        self._max_output_tokens = max_output_tokens

    async def verify(
        self, gateway: ModelGateway, question: str, evidence: EvidenceSet, draft: Draft
    ) -> Verification:
        request = ChatRequest(
            deployment=self._deployment,
            operation=self.name,
            max_output_tokens=self._max_output_tokens,
            json_output=True,
            messages=(
                ChatMessage(role="system", content=prompts.CRITIC),
                ChatMessage(
                    role="user",
                    content=(
                        f"{render_sources(evidence.sources)}\n\n"
                        f"Question: {neutralize(question)}\n\n"
                        f"<draft>\n{html.escape(draft.text, quote=False)}\n</draft>"
                    ),
                ),
            ),
        )
        # Invalid critic output propagates as AgentOutputError: unlike planning, a
        # verification step that silently passes would defeat its purpose.
        output = await gateway.complete_json(request, _CriticOutput)
        verdicts = [
            ClaimVerdict(
                claim=c.claim,
                source_numbers=tuple(c.sources),
                supported=c.supported,
                reason=c.reason,
            )
            for c in output.claims
        ]
        verdicts += self._deterministic_findings(draft, len(evidence.sources))
        return Verification(claims=tuple(verdicts))

    @staticmethod
    def _deterministic_findings(draft: Draft, source_count: int) -> list[ClaimVerdict]:
        findings = [
            ClaimVerdict(
                claim=f"Citation [{number}]",
                source_numbers=(number,),
                supported=False,
                reason=f"Source [{number}] does not exist; {source_count} sources were given.",
            )
            for number in invalid_numbers(draft.text, source_count)
        ]
        findings += [
            ClaimVerdict(
                claim=statement,
                source_numbers=(),
                supported=False,
                reason="Factual statement without a citation.",
            )
            for statement in uncited_statements(draft.text)
        ]
        return findings
