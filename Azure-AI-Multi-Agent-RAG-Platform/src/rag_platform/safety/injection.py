"""Defenses against indirect prompt injection carried by retrieved documents.

Retrieved text is untrusted: anyone who can edit a wiki page or upload a PDF can
plant instructions in it. Three layers are applied here; the system prompts add the
fourth (an explicit instruction hierarchy that treats `<source>` content as data):

1. Delimiting: every chunk is wrapped in a `<source>` element with escaped attributes.
2. Neutralization: tag-like sequences that could close or forge a delimiter are escaped,
   so a chunk cannot break out of its element.
3. Screening: chunks matching well-known injection phrasings are withheld from the
   prompt and reported, rather than being trusted to the model's judgment.
"""

from __future__ import annotations

import html
import re
from collections.abc import Sequence
from dataclasses import dataclass

from rag_platform.domain import RetrievedChunk

_DELIMITER_TAGS = re.compile(
    r"<\s*/?\s*(source|sources|system|instructions|user|assistant|draft|feedback)\b",
    re.IGNORECASE,
)

_INJECTION_PATTERNS = re.compile(
    r"|".join(
        (
            r"ignore\s+(?:all\s+|any\s+)?(?:the\s+)?(?:previous|prior|above)\s+instructions",
            r"disregard\s+(?:all\s+|the\s+)?(?:previous|prior|above|system)",
            r"you\s+are\s+now\s+(?:a|an|in)\b",
            r"reveal\s+(?:your|the)\s+(?:system\s+)?prompt",
            r"new\s+instructions\s*:",
            r"do\s+not\s+cite\s+(?:any\s+)?sources",
        )
    ),
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ScreenedEvidence:
    accepted: tuple[RetrievedChunk, ...]
    withheld: tuple[RetrievedChunk, ...]


def neutralize(text: str) -> str:
    """Escape anything that looks like an opening or closing delimiter tag."""
    return _DELIMITER_TAGS.sub(lambda m: m.group(0).replace("<", "&lt;"), text)


def looks_like_injection(text: str) -> bool:
    return _INJECTION_PATTERNS.search(text) is not None


def screen(chunks: Sequence[RetrievedChunk]) -> ScreenedEvidence:
    accepted: list[RetrievedChunk] = []
    withheld: list[RetrievedChunk] = []
    for chunk in chunks:
        (withheld if looks_like_injection(chunk.content) else accepted).append(chunk)
    return ScreenedEvidence(accepted=tuple(accepted), withheld=tuple(withheld))


def render_sources(chunks: Sequence[RetrievedChunk]) -> str:
    """Render chunks as numbered `<source>` elements; numbers are 1-based citation ids."""
    parts = []
    for number, chunk in enumerate(chunks, start=1):
        attributes = (
            f'id="{number}" doc="{html.escape(chunk.doc_id)}" '
            f'title="{html.escape(chunk.title)}" section="{html.escape(chunk.heading_path)}"'
        )
        parts.append(f"<source {attributes}>\n{neutralize(chunk.content)}\n</source>")
    return "<sources>\n" + "\n".join(parts) + "\n</sources>"
