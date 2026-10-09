"""Structure-aware chunking for policy documents.

Design choices:

* Chunks never cross a heading boundary. Policy documents are organized so that one
  section answers one kind of question ("Approval thresholds", "Exceptions"), and
  mixing sections dilutes both the embedding and the citation.
* Blocks (paragraphs, lists, tables, fenced code) are packed whole; only a block that
  is larger than a chunk on its own is split, first at sentence boundaries.
* Consecutive chunks of the same section overlap by whole sentences so a rule and its
  qualifier ("...unless the supplier is pre-approved") are not separated.
* The heading path travels with every chunk as metadata and is embedded together with
  the text ("contextual chunk headers"), so a chunk that says "the limit is $25,000"
  still knows it belongs to "Procurement > Approval thresholds > Capital expenditure".
* Chunk ids are content-addressed, so re-ingesting an edited document re-embeds only
  the chunks whose text actually changed.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass

from rag_platform.domain import Chunk, SourceDocument, estimate_tokens

_HEADING = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<text>.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")
_KEY_SAFE = re.compile(r"[^A-Za-z0-9_-]")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    max_tokens: int = 350
    overlap_tokens: int = 60
    # Part of the content hash: changing the embedding model or its dimensions must
    # invalidate every stored vector even when the text is unchanged.
    embedding_signature: str = "unspecified"

    def __post_init__(self) -> None:
        if self.max_tokens <= 0 or not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError("require max_tokens > 0 and 0 <= overlap_tokens < max_tokens")


@dataclass(frozen=True, slots=True)
class _Section:
    heading_path: tuple[str, ...]
    blocks: tuple[str, ...]


def _split_blocks(lines: list[str]) -> list[str]:
    """Group lines into blank-line separated blocks, keeping fenced code intact."""
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in lines:
        if _FENCE.match(line):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return [b for b in blocks if b]


def _sections(document: SourceDocument) -> Iterator[_Section]:
    stack: list[tuple[int, str]] = []
    pending: list[str] = []
    in_fence = False

    def flush() -> _Section | None:
        blocks = _split_blocks(pending)
        path = (document.title, *(text for _, text in stack if text != document.title))
        return _Section(heading_path=path, blocks=tuple(blocks)) if blocks else None

    for line in document.text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        heading = None if in_fence else _HEADING.match(line)
        if heading is None:
            pending.append(line)
            continue
        if (section := flush()) is not None:
            yield section
        pending = []
        level = len(heading.group("hashes"))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading.group("text").strip()))
    if (section := flush()) is not None:
        yield section


def _sentences(block: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(block) if s.strip()]


def _split_oversized(block: str, max_tokens: int) -> list[str]:
    """Split a block larger than `max_tokens` at sentence, then word, boundaries."""
    pieces: list[str] = []
    for sentence in _sentences(block):
        if estimate_tokens(sentence) <= max_tokens:
            pieces.append(sentence)
            continue
        words = sentence.split()
        current: list[str] = []
        for word in words:
            if current and estimate_tokens(" ".join([*current, word])) > max_tokens:
                pieces.append(" ".join(current))
                current = []
            current.append(word)
        if current:
            pieces.append(" ".join(current))
    return pieces


def _overlap_tail(units: list[str], overlap_tokens: int) -> list[str]:
    """Return the trailing sentences of `units` that fit in the overlap budget."""
    if overlap_tokens == 0:
        return []
    tail: list[str] = []
    budget = overlap_tokens
    for sentence in reversed(_sentences("\n\n".join(units))):
        cost = estimate_tokens(sentence)
        if cost > budget:
            break
        tail.insert(0, sentence)
        budget -= cost
    return [" ".join(tail)] if tail else []


def _pack(blocks: tuple[str, ...], config: ChunkingConfig) -> list[str]:
    units: list[str] = []
    for block in blocks:
        if estimate_tokens(block) > config.max_tokens:
            units.extend(_split_oversized(block, config.max_tokens))
        else:
            units.append(block)

    texts: list[str] = []
    current: list[str] = []
    carried = 0  # number of leading units in `current` that are overlap from the previous chunk
    for unit in units:
        candidate = "\n\n".join([*current, unit])
        if current and estimate_tokens(candidate) > config.max_tokens and len(current) > carried:
            texts.append("\n\n".join(current))
            current = _overlap_tail(current, config.overlap_tokens)
            # Drop the overlap if it would make the next unit overflow on its own.
            if estimate_tokens("\n\n".join([*current, unit])) > config.max_tokens:
                current = []
            carried = len(current)
        current.append(unit)
    if len(current) > carried:
        texts.append("\n\n".join(current))
    return texts


def embedding_text(title: str, heading_label: str, content: str) -> str:
    """The exact text that is embedded for a chunk (content plus its context header)."""
    return f"{title}\n{heading_label}\n\n{content}"


def chunk_document(document: SourceDocument, config: ChunkingConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen_ids: dict[str, int] = {}
    doc_key = _KEY_SAFE.sub("-", document.doc_id)
    for section in _sections(document):
        heading_label = " > ".join(section.heading_path)
        for text in _pack(section.blocks, config):
            digest = hashlib.sha256(
                f"{config.embedding_signature}\n"
                f"{embedding_text(document.title, heading_label, text)}".encode()
            ).hexdigest()
            chunk_id = f"{doc_key}-{digest[:20]}"
            # Identical text under an identical heading can legitimately repeat
            # ("Exceptions: none."); disambiguate instead of silently dropping it.
            occurrence = seen_ids.get(chunk_id, 0)
            seen_ids[chunk_id] = occurrence + 1
            if occurrence:
                chunk_id = f"{chunk_id}-{occurrence}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    chunk_index=len(chunks),
                    title=document.title,
                    heading_path=section.heading_path,
                    content=text,
                    source_path=document.source_path,
                    category=document.category,
                    content_hash=digest,
                    token_estimate=estimate_tokens(text),
                )
            )
    return chunks
