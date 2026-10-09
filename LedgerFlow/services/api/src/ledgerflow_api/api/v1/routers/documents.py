import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Header, Query, Response, UploadFile, status
from fastapi.responses import RedirectResponse

from ledgerflow_api.api.deps import (
    AdminPrincipal,
    CurrentPrincipal,
    Documents,
    Resources,
    ReviewerPrincipal,
    Storage,
)
from ledgerflow_api.db.models import Document, Extraction
from ledgerflow_api.domain.lifecycle import DocumentStatus
from ledgerflow_api.errors import BadRequestError
from ledgerflow_api.etags import etag_for
from ledgerflow_api.pagination import InvalidCursorError, SortOrder, decode_cursor
from ledgerflow_api.schemas.audit import AuditEventOut
from ledgerflow_api.schemas.common import Page
from ledgerflow_api.schemas.documents import (
    DocumentDetail,
    DocumentSummary,
    FieldCorrectionRequest,
    RejectRequest,
)
from ledgerflow_api.services.documents import DocumentFilters
from ledgerflow_api.services.uploads import read_upload

router = APIRouter(prefix="/documents", tags=["documents"])

IfMatch = Annotated[str | None, Header(description='ETag from a previous read, e.g. W/"3"')]


def _detail(
    response: Response, document: Document, extraction: Extraction | None
) -> DocumentDetail:
    response.headers["ETag"] = etag_for(document.version)
    return DocumentDetail.from_parts(document, extraction)


@router.get("")
async def list_documents(
    _principal: CurrentPrincipal,
    documents: Documents,
    status_: Annotated[list[DocumentStatus] | None, Query(alias="status")] = None,
    doc_type: Literal["invoice", "nfe"] | None = None,
    q: Annotated[str | None, Query(max_length=200)] = None,
    min_confidence: Annotated[float | None, Query(ge=0, le=1)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    cursor: str | None = None,
    sort: SortOrder = "-created_at",
) -> Page[DocumentSummary]:
    try:
        decoded = decode_cursor(cursor, expected_sort=sort) if cursor else None
    except InvalidCursorError as exc:
        raise BadRequestError(str(exc)) from None
    filters = DocumentFilters(
        statuses=status_ or (),
        doc_type=doc_type,
        query=q.strip() if q and q.strip() else None,
        min_confidence=min_confidence,
    )
    page, next_cursor = await documents.search(filters, limit=limit, cursor=decoded, sort=sort)
    return Page(items=[DocumentSummary.from_document(d) for d in page], next_cursor=next_cursor)


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    responses={status.HTTP_200_OK: {"description": "Identical file already uploaded"}},
)
async def upload_document(
    _principal: ReviewerPrincipal,
    documents: Documents,
    storage: Storage,
    resources: Resources,
    file: UploadFile,
    response: Response,
) -> DocumentSummary:
    upload = await read_upload(
        file, filename=file.filename, max_bytes=resources.settings.max_upload_bytes
    )
    document, created = await documents.upload(storage, upload)
    if not created:
        response.status_code = status.HTTP_200_OK
    return DocumentSummary.from_document(document)


@router.get("/{document_id}")
async def get_document(
    document_id: uuid.UUID, _principal: CurrentPrincipal, documents: Documents, response: Response
) -> DocumentDetail:
    document = await documents.get(document_id)
    return _detail(response, document, await documents.extraction_of(document))


@router.get("/{document_id}/file", status_code=status.HTTP_307_TEMPORARY_REDIRECT)
async def download_document(
    document_id: uuid.UUID, _principal: CurrentPrincipal, documents: Documents, storage: Storage
) -> RedirectResponse:
    document = await documents.get(document_id)
    url = await storage.presigned_get_url(
        document.storage_key, filename=document.filename, content_type=document.content_type
    )
    return RedirectResponse(url, status_code=status.HTTP_307_TEMPORARY_REDIRECT)


@router.patch("/{document_id}/fields")
async def correct_fields(
    document_id: uuid.UUID,
    body: FieldCorrectionRequest,
    _principal: ReviewerPrincipal,
    documents: Documents,
    response: Response,
    if_match: IfMatch = None,
) -> DocumentDetail:
    document, extraction = await documents.correct_fields(document_id, body.fields, if_match)
    return _detail(response, document, extraction)


@router.post("/{document_id}/approve")
async def approve_document(
    document_id: uuid.UUID,
    _principal: ReviewerPrincipal,
    documents: Documents,
    response: Response,
    if_match: IfMatch = None,
) -> DocumentDetail:
    document = await documents.approve(document_id, if_match)
    return _detail(response, document, await documents.extraction_of(document))


@router.post("/{document_id}/reject")
async def reject_document(
    document_id: uuid.UUID,
    body: RejectRequest,
    _principal: ReviewerPrincipal,
    documents: Documents,
    response: Response,
    if_match: IfMatch = None,
) -> DocumentDetail:
    document = await documents.reject(document_id, body.reason, if_match)
    return _detail(response, document, await documents.extraction_of(document))


@router.post("/{document_id}/retry", status_code=status.HTTP_202_ACCEPTED)
async def retry_document(
    document_id: uuid.UUID, _principal: AdminPrincipal, documents: Documents
) -> Response:
    await documents.retry(document_id)
    return Response(status_code=status.HTTP_202_ACCEPTED)


@router.get("/{document_id}/audit")
async def get_audit_trail(
    document_id: uuid.UUID, _principal: CurrentPrincipal, documents: Documents
) -> list[AuditEventOut]:
    return [
        AuditEventOut.model_validate(event) for event in await documents.audit_trail(document_id)
    ]
