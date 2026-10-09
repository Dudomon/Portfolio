"""Planner agent: decomposes a question into self-contained sub-questions."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from rag_platform.agents import prompts
from rag_platform.agents.gateway import AgentOutputError, ModelGateway
from rag_platform.agents.messages import Plan, SubQuestion
from rag_platform.clients.protocols import ChatMessage, ChatRequest
from rag_platform.safety.injection import neutralize

logger = logging.getLogger(__name__)


class _PlannerOutput(BaseModel):
    sub_questions: list[str] = Field(min_length=1)


class PlannerAgent:
    name = "planner"

    def __init__(self, deployment: str, max_sub_questions: int) -> None:
        self._deployment = deployment
        self._max_sub_questions = max_sub_questions

    async def plan(self, gateway: ModelGateway, question: str) -> Plan:
        request = ChatRequest(
            deployment=self._deployment,
            operation=self.name,
            max_output_tokens=300,
            json_output=True,
            messages=(
                ChatMessage(
                    role="system",
                    content=prompts.PLANNER.format(max_sub_questions=self._max_sub_questions),
                ),
                ChatMessage(role="user", content=neutralize(question)),
            ),
        )
        try:
            output = await gateway.complete_json(request, _PlannerOutput)
        except AgentOutputError:
            # Planning is an optimization, not a requirement: searching with the
            # original question still yields a useful (if less complete) answer.
            logger.warning("planner output invalid; falling back to the original question")
            return Plan(sub_questions=(SubQuestion(id="q1", text=question),), fallback=True)

        unique = list(dict.fromkeys(q.strip() for q in output.sub_questions if q.strip()))
        selected = unique[: self._max_sub_questions] or [question]
        return Plan(
            sub_questions=tuple(
                SubQuestion(id=f"q{i}", text=text) for i, text in enumerate(selected, start=1)
            )
        )
