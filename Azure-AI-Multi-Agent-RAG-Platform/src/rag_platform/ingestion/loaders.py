"""Document loaders for Markdown (with front matter) and text-based PDF files."""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import date
from pathlib import Path

from pypdf import PdfReader

from rag_platform.domain import SourceDocument

SUPPORTED_SUFFIXES = frozenset({".md", ".pdf"})
_FRONT_MATTER = re.compile(r"\A---\s*\n(?P<body>.*?)\n---\s*\n", re.DOTALL)
_H1 = re.compile(r"^#\s+(?P<title>.+)$", re.MULTILINE)


class DocumentLoadError(ValueError):
    pass


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Parse flat `key: value` front matter.

    A full YAML parser is not needed for the flat metadata policy documents carry,
    and avoiding it keeps arbitrary YAML tags out of the ingestion path.
    """
    match = _FRONT_MATTER.match(text)
    if match is None:
        return {}, text
    metadata: dict[str, str] = {}
    for line in match.group("body").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip():
            metadata[key.strip().lower()] = value.strip().strip("\"'")
    return metadata, text[match.end() :]


def _doc_id_from_path(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "-", path.stem).upper()


def load_markdown(path: Path, root: Path) -> SourceDocument:
    metadata, body = _parse_front_matter(path.read_text(encoding="utf-8"))
    heading = _H1.search(body)
    title = metadata.get("title") or (heading.group("title").strip() if heading else path.stem)
    try:
        effective = (
            date.fromisoformat(metadata["effective_date"]) if "effective_date" in metadata else None
        )
    except ValueError as error:
        raise DocumentLoadError(f"{path}: invalid effective_date") from error
    return SourceDocument(
        doc_id=metadata.get("doc_id") or _doc_id_from_path(path),
        title=title,
        source_path=path.relative_to(root).as_posix(),
        text=body.strip(),
        category=metadata.get("category", "general"),
        owner=metadata.get("owner"),
        effective_date=effective,
    )


def load_pdf(path: Path, root: Path) -> SourceDocument:
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    if not text:
        # Scanned PDFs need OCR (e.g. Azure AI Document Intelligence) before ingestion.
        raise DocumentLoadError(f"{path}: no extractable text layer")
    metadata_title = reader.metadata.title if reader.metadata else None
    return SourceDocument(
        doc_id=_doc_id_from_path(path),
        title=metadata_title or path.stem.replace("_", " ").replace("-", " ").title(),
        source_path=path.relative_to(root).as_posix(),
        text=text,
    )


def load_documents(root: Path) -> Iterator[SourceDocument]:
    """Yield documents under `root` in a stable order so ingestion runs are reproducible."""
    for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES):
        if path.suffix.lower() == ".md":
            yield load_markdown(path, root)
        else:
            yield load_pdf(path, root)
