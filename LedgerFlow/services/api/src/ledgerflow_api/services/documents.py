"""Document use cases for authenticated users.

Each mutating method is one unit of work: lock the row, check the precondition, apply the
change through the lifecycle state machine, write the audit event (and outbox event where
relevant), commit. Nothing is committed piecemeal.
"""

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Select, select, tuple_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from ledgerflow_api.db.models import AuditEvent, Document, Extraction
from ledgerflow_api.domain.fields import (
    FieldValueError,
    JsonValue,
    normalize_field,
    overall_confidence,
)
from ledgerflow_api.domain.lifecycle import CORRECTABLE_STATUSES, DocumentStatus, ensure_transition
from ledgerflow_api.domain.money import DecimalFormatError, parse_money
from ledgerflow_api.errors import ConflictError, NotFoundError, UnprocessableError
from ledgerflow_api.etags import check_if_match
from ledgerflow_api.pagination import Cursor, SortOrder, encode_cursor
from ledgerflow_api.security import Principal
from ledgerflow_api.services.audit import Diff, diff_snapshots, record_audit
from ledgerflow_api.services.outbox import enqueue_document_uploaded
from ledgerflow_api.services.storage import ObjectStorage, storage_key
from ledgerflow_api.services.uploads import UploadedFile


@dataclass(frozen=True, slots=True)
class DocumentFilters:
    statuses: Sequence[DocumentStatus] = ()
    doc_type: str | None = None
    query: str | None = None
    min_confidence: float | None = None


def _escape_like(term: str) -> str:
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _with_uploader(stmt: Select[Document]) -> Select[Document]:
    return stmt.options(joinedload(Document.uploaded_by, innerjoin=True))


async def lock_document(
    session: AsyncSession, document_id: uuid.UUID, *, tenant_id: uuid.UUID | None = None
) -> Document:
    """Load a document with ``FOR UPDATE`` so concurrent writers serialize on the row."""
    stmt = _with_uploader(select(Document)).where(Document.id == document_id)
    if tenant_id is not None:
        stmt = stmt.where(Document.tenant_id == tenant_id)
    document = await session.scalar(stmt.with_for_update(of=Document))
    if document is None:
        raise NotFoundError(f"Document {document_id} does not exist.")
    return document


def transition(
    session: AsyncSession,
    document: Document,
    target: DocumentStatus,
    *,
    actor_id: uuid.UUID | None,
    action: str | None = None,
    extra_diff: Diff | None = None,
) -> None:
    previous = document.status
    document.status = ensure_transition(previous, target)
    record_audit(
        session,
        tenant_id=document.tenant_id,
        actor_id=actor_id,
        action=action or f"document.{target.value}",
        entity_id=document.id,
        diff={"status": {"before": previous.value, "after": target.value}, **(extra_diff or {})},
    )


def attach_extraction(document: Document, extraction: Extraction) -> None:
    """Point the document at a new extraction and refresh its denormalized summary columns."""
    values = {key: field["value"] for key, field in extraction.fields.items()}
    document.extraction_version = extraction.version
    document.overall_confidence = extraction.overall_confidence
    supplier_name, currency = values.get("supplier_name"), values.get("currency")
    document.supplier_name = supplier_name if isinstance(supplier_name, str) else None
    document.currency = currency if isinstance(currency, str) and len(currency) == 3 else None
    try:
        document.total_amount = parse_money(values.get("total_amount"))
    except DecimalFormatError:
        document.total_amount = None


async def current_extraction(session: AsyncSession, document: Document) -> Extraction | None:
    if document.extraction_version is None:
        return None
    return await session.scalar(
        select(Extraction).where(
            Extraction.document_id == document.id,
            Extraction.version == document.extraction_version,
        )
    )


class DocumentService:
    def __init__(self, session: AsyncSession, principal: Principal) -> None:
        self._session = session
        self._principal = principal

    async def get(self, document_id: uuid.UUID) -> Document:
        # Tenant isolation is enforced by RLS; a foreign document is simply invisible (404).
        document = await self._session.scalar(
            _with_uploader(select(Document)).where(Document.id == document_id)
        )
        if document is None:
            raise NotFoundError(f"Document {document_id} does not exist.")
        return document

    async def extraction_of(self, document: Document) -> Extraction | None:
        return await current_extraction(self._session, document)

    async def search(
        self,
        filters: DocumentFilters,
        *,
        limit: int,
        cursor: Cursor | None,
        sort: SortOrder,
    ) -> tuple[list[Document], str | None]:
        descending = sort == "-created_at"
        stmt = _with_uploader(select(Document))
        if filters.statuses:
            stmt = stmt.where(Document.status.in_(filters.statuses))
        if filters.doc_type:
            stmt = stmt.where(Document.doc_type == filters.doc_type)
        if filters.min_confidence is not None:
            stmt = stmt.where(Document.overall_confidence >= filters.min_confidence)
        if filters.query:
            pattern = f"%{_escape_like(filters.query)}%"
            stmt = stmt.where(
                Document.filename.ilike(pattern, escape="\\")
                | Document.supplier_name.ilike(pattern, escape="\\")
            )
        if cursor is not None:
            key = tuple_(Document.created_at, Document.id)
            boundary = tuple_(cursor.created_at, cursor.id)
            stmt = stmt.where(key < boundary if descending else key > boundary)
        order = (
            (Document.created_at.desc(), Document.id.desc())
            if descending
            else (Document.created_at.asc(), Document.id.asc())
        )
        # Fetch one extra row to learn whether another page exists without a COUNT.
        rows = list((await self._session.scalars(stmt.order_by(*order).limit(limit + 1))).all())
        page, has_more = rows[:limit], len(rows) > limit
        next_cursor = None
        if has_more:
            last = page[-1]
            next_cursor = encode_cursor(Cursor(created_at=last.created_at, id=last.id, sort=sort))
        return page, next_cursor

    async def upload(self, storage: ObjectStorage, upload: UploadedFile) -> tuple[Document, bool]:
        """Store and enqueue a file; returns ``(document, created)``. Duplicates are reused."""
        existing = await self._find_by_sha256(upload.sha256)
        if existing is not None:
            return existing, False

        document_id = uuid.uuid4()
        key = storage_key(self._principal.tenant_id, document_id, upload.sha256)
        # Object first, row second: a crash in between leaves an unreferenced object (cheap to
        # sweep), never a row pointing at a file that doesn't exist.
        await storage.put(key, upload.data, content_type=upload.content_type)

        document = Document(
            id=document_id,
            tenant_id=self._principal.tenant_id,
            uploaded_by_id=self._principal.user_id,
            filename=upload.filename,
            content_type=upload.content_type,
            size_bytes=upload.size_bytes,
            sha256=upload.sha256,
            storage_key=key,
            status=DocumentStatus.UPLOADED,
        )
        try:
            async with self._session.begin_nested():
                self._session.add(document)
        except IntegrityError:
            # A concurrent upload of the same bytes won the (tenant_id, sha256) unique race.
            winner = await self._find_by_sha256(upload.sha256)
            if winner is None:
                raise
            return winner, False

        record_audit(
            self._session,
            tenant_id=document.tenant_id,
            actor_id=self._principal.user_id,
            action="document.uploaded",
            entity_id=document.id,
            diff={
                "filename": {"before": None, "after": document.filename},
                "sha256": {"before": None, "after": document.sha256},
            },
        )
        # The outbox row commits atomically with this transition, so "queued" can never be
        # observed for a document whose event might still be lost.
        transition(self._session, document, DocumentStatus.QUEUED, actor_id=self._principal.user_id)
        enqueue_document_uploaded(self._session, document)
        await self._session.commit()
        return await self.get(document.id), True

    async def correct_fields(
        self, document_id: uuid.UUID, corrections: Mapping[str, JsonValue], if_match: str | None
    ) -> tuple[Document, Extraction]:
        # Human input is authoritative, so only per-field validity is enforced here; the
        # cross-field heuristics exist to triage model output, not to second-guess reviewers.
        normalized: dict[str, JsonValue] = {}
        errors: list[dict[str, str]] = []
        for key, value in corrections.items():
            try:
                normalized[key] = normalize_field(key, value)
            except FieldValueError as exc:
                errors.append({"field": exc.key, "msg": exc.message})
        if errors:
            raise UnprocessableError(
                "One or more field values are invalid.", extensions={"errors": errors}
            )

        document = await lock_document(self._session, document_id)
        check_if_match(if_match, document.version)
        if document.status not in CORRECTABLE_STATUSES:
            raise ConflictError(
                f"Fields cannot be corrected while the document is '{document.status}'."
            )
        latest = await current_extraction(self._session, document)
        if latest is None:
            raise ConflictError("The document has no extraction to correct.")

        fields: dict[str, Any] = dict(latest.fields)
        for key, value in normalized.items():
            fields[key] = {"value": value, "confidence": 1.0, "source": "human"}
        extraction = Extraction(
            tenant_id=document.tenant_id,
            document_id=document.id,
            version=latest.version + 1,
            model=None,
            prompt_version=None,
            fields=fields,
            overall_confidence=overall_confidence(f["confidence"] for f in fields.values()),
            cost_usd_micros=0,
            latency_ms=0,
            created_by_id=self._principal.user_id,
        )
        self._session.add(extraction)
        attach_extraction(document, extraction)
        record_audit(
            self._session,
            tenant_id=document.tenant_id,
            actor_id=self._principal.user_id,
            action="document.fields_corrected",
            entity_id=document.id,
            diff=diff_snapshots(
                {key: latest.fields.get(key, {}).get("value") for key in normalized},
                normalized,
            ),
        )
        await self._session.commit()
        return document, extraction

    async def approve(self, document_id: uuid.UUID, if_match: str | None) -> Document:
        document = await lock_document(self._session, document_id)
        check_if_match(if_match, document.version)
        transition(
            self._session, document, DocumentStatus.APPROVED, actor_id=self._principal.user_id
        )
        document.reviewed_by_id = self._principal.user_id
        document.reviewed_at = datetime.now(UTC)
        await self._session.commit()
        return document

    async def reject(self, document_id: uuid.UUID, reason: str, if_match: str | None) -> Document:
        document = await lock_document(self._session, document_id)
        check_if_match(if_match, document.version)
        transition(
            self._session,
            document,
            DocumentStatus.REJECTED,
            actor_id=self._principal.user_id,
            extra_diff={"rejection_reason": {"before": document.rejection_reason, "after": reason}},
        )
        document.rejection_reason = reason
        document.reviewed_by_id = self._principal.user_id
        document.reviewed_at = datetime.now(UTC)
        await self._session.commit()
        return document

    async def retry(self, document_id: uuid.UUID) -> None:
        document = await lock_document(self._session, document_id)
        transition(
            self._session,
            document,
            DocumentStatus.QUEUED,
            actor_id=self._principal.user_id,
            action="document.requeued",
            extra_diff={"last_error": {"before": document.last_error, "after": None}},
        )
        document.last_error = None
        enqueue_document_uploaded(self._session, document)
        await self._session.commit()

    async def audit_trail(self, document_id: uuid.UUID) -> list[AuditEvent]:
        await self.get(document_id)
        result = await self._session.scalars(
            select(AuditEvent)
            .options(joinedload(AuditEvent.actor))
            .where(AuditEvent.entity_type == "document", AuditEvent.entity_id == document_id)
            .order_by(AuditEvent.seq)
        )
        return list(result.all())

    async def _find_by_sha256(self, sha256: str) -> Document | None:
        return await self._session.scalar(
            _with_uploader(select(Document)).where(Document.sha256 == sha256)
        )
