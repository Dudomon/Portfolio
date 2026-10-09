"""LLM-as-judge scoring with versioned rubric prompts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from string import Template

from pydantic import BaseModel, Field, ValidationError

from rag_platform.clients.protocols import ChatMessage, ChatModel, ChatRequest

_RUBRIC_FILE = re.compile(r"^(?P<name>[a-z_]+)\.v(?P<version>\d+)\.md$")


@dataclass(frozen=True, slots=True)
class Rubric:
    name: str
    version: int
    template: Template
    sha256: str

    @property
    def label(self) -> str:
        return f"{self.name}.v{self.version} ({self.sha256[:12]})"


def load_rubrics(directory: Path) -> dict[str, Rubric]:
    """Load the highest version of each rubric (`<name>.v<N>.md`).

    Reports record the version and content hash, so judge scores from different
    rubric revisions are never compared by accident.
    """
    rubrics: dict[str, Rubric] = {}
    for path in sorted(directory.glob("*.md")):
        match = _RUBRIC_FILE.match(path.name)
        if match is None:
            continue
        text = path.read_text(encoding="utf-8")
        rubric = Rubric(
            name=match.group("name"),
            version=int(match.group("version")),
            template=Template(text),
            sha256=hashlib.sha256(text.encode()).hexdigest(),
        )
        current = rubrics.get(rubric.name)
        if current is None or rubric.version > current.version:
            rubrics[rubric.name] = rubric
    return rubrics


class JudgeScore(BaseModel):
    score: int = Field(ge=1, le=5)
    rationale: str

    @property
    def normalized(self) -> float:
        """Map the 1-5 scale onto 0-1 so it can be averaged and compared to thresholds."""
        return (self.score - 1) / 4


class JudgeError(Exception):
    pass


class LlmJudge:
    def __init__(self, chat: ChatModel, deployment: str, rubrics: dict[str, Rubric]) -> None:
        self._chat = chat
        self._deployment = deployment
        self.rubrics = rubrics

    async def score(self, rubric_name: str, **fields: str) -> JudgeScore:
        rubric = self.rubrics[rubric_name]
        prompt = rubric.template.substitute(fields)
        result = await self._chat.complete(
            ChatRequest(
                deployment=self._deployment,
                operation=f"judge_{rubric_name}",
                # Temperature 0 and a fixed rubric keep run-to-run variance low; it is
                # not zero, which is why thresholds are set below observed baselines.
                temperature=0.0,
                max_output_tokens=400,
                json_output=True,
                messages=(ChatMessage(role="user", content=prompt),),
            )
        )
        try:
            return JudgeScore.model_validate(json.loads(result.text))
        except (json.JSONDecodeError, ValidationError) as error:
            raise JudgeError(f"{rubric_name}: judge returned invalid output") from error
