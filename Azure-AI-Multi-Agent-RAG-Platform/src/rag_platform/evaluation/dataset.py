"""Golden dataset and threshold loading."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GoldenItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    question: str
    expected_doc_ids: tuple[str, ...] = ()
    reference_answer: str
    category: str
    # Unanswerable items check that the assistant abstains instead of guessing.
    answerable: bool = True

    @model_validator(mode="after")
    def _expected_docs_match_answerability(self) -> GoldenItem:
        if self.answerable != bool(self.expected_doc_ids):
            raise ValueError(f"{self.id}: answerable items need expected_doc_ids, others none")
        return self


class Thresholds(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    retrieval: dict[str, float] = Field(default_factory=dict)
    generation: dict[str, float] = Field(default_factory=dict)


def load_golden(path: Path) -> list[GoldenItem]:
    items = [
        GoldenItem.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate item ids")
    return items


def load_thresholds(path: Path) -> Thresholds:
    with path.open("rb") as handle:
        return Thresholds.model_validate(tomllib.load(handle))
