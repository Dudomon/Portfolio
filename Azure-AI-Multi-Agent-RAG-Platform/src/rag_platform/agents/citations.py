"""Deterministic citation parsing and checks.

These checks run before (and independently of) the LLM critic. They are cheap,
cannot be talked out of a verdict, and catch the most common failure modes:
citing a source number that does not exist and stating facts without any citation.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rag_platform.agents.messages import Citation
from rag_platform.domain import RetrievedChunk

_MARKER = re.compile(r"\[(\d{1,3})\]")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_LEADING_MARKERS = re.compile(r"^((?:\[\d{1,3}\])+)\s*")
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_ABSTENTION = re.compile(
    r"\b(?:not (?:covered|specified|stated|mentioned|addressed)|could not find|"
    r"do(?:es)? not (?:say|specify|state|cover|address)|no information)\b",
    re.IGNORECASE,
)
_MIN_WORDS_FOR_CLAIM = 4


def cited_numbers(text: str) -> tuple[int, ...]:
    """Source numbers in order of first appearance."""
    return tuple(dict.fromkeys(int(n) for n in _MARKER.findall(text)))


def as_citation(number: int, chunk: RetrievedChunk) -> Citation:
    return Citation(
        source_number=number,
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        title=chunk.title,
        heading_path=chunk.heading_path,
        source_path=chunk.source_path,
        content=chunk.content,
    )


def resolve_citations(text: str, sources: Sequence[RetrievedChunk]) -> tuple[Citation, ...]:
    """Map the valid markers in `text` to their sources; invalid numbers are skipped."""
    return tuple(
        as_citation(number, sources[number - 1])
        for number in cited_numbers(text)
        if 1 <= number <= len(sources)
    )


def is_abstention(text: str) -> bool:
    return _ABSTENTION.search(text) is not None


def invalid_numbers(text: str, source_count: int) -> tuple[int, ...]:
    return tuple(n for n in cited_numbers(text) if not 1 <= n <= source_count)


def statements(text: str) -> list[str]:
    """Split an answer into sentence- or bullet-level statements.

    Models place markers both before and after the period ("... 30 days [2]." and
    "... 30 days. [2]"); markers that open a sentence are re-attached to the
    previous statement they belong to.
    """
    result: list[str] = []
    for raw_line in text.splitlines():
        line = _BULLET.sub("", raw_line).strip()
        for raw_sentence in _SENTENCE_BOUNDARY.split(line) if line else ():
            sentence = raw_sentence
            leading = _LEADING_MARKERS.match(sentence)
            if leading and result:
                result[-1] = f"{result[-1]} {leading.group(1)}"
                sentence = sentence[leading.end() :]
            if sentence.strip():
                result.append(sentence.strip())
    return result


def uncited_statements(text: str) -> list[str]:
    """Factual-looking statements that carry no citation marker."""
    return [
        s
        for s in statements(text)
        if not _MARKER.search(s)
        and len(s.split()) >= _MIN_WORDS_FOR_CLAIM
        and not s.endswith(":")
        and not _ABSTENTION.search(s)
    ]
