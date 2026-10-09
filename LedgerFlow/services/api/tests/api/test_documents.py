from httpx import AsyncClient
from sqlalchemy import func, select

from ledgerflow_api.api.deps import AppResources
from ledgerflow_api.db.models import OutboxEvent
from ledgerflow_api.db.session import system_session
from ledgerflow_api.domain.roles import Role
from tests.api.conftest import AuthHeaders, InMemoryStorage, SeededTenant
from tests.api.factories import extract, pdf_bytes, upload


async def _outbox_count(resources: AppResources) -> int:
    async with system_session(resources.session_factory) as session:
        return await session.scalar(select(func.count()).select_from(OutboxEvent)) or 0


async def test_upload_stores_sniffed_type_and_enqueues_event(
    client: AsyncClient,
    tenant: SeededTenant,
    auth: AuthHeaders,
    storage: InMemoryStorage,
    resources: AppResources,
) -> None:
    response = await client.post(
        "/documents",
        headers=auth(tenant, Role.REVIEWER),
        # The declared type is a lie; the bytes say PDF.
        files={"file": ("../scan.png", pdf_bytes(), "image/png")},
    )

    assert response.status_code == 202
    document = response.json()
    assert document["status"] == "queued"
    assert document["filename"] == "scan.png"
    detail = (
        await client.get(f"/documents/{document['id']}", headers=auth(tenant, Role.VIEWER))
    ).json()
    assert detail["content_type"] == "application/pdf"
    key = f"tenants/{tenant.id}/documents/{document['id']}/{detail['sha256']}"
    assert storage.objects[key][1] == "application/pdf"
    assert await _outbox_count(resources) == 1


async def test_duplicate_upload_returns_existing_document(
    client: AsyncClient, tenant: SeededTenant, auth: AuthHeaders, resources: AppResources
) -> None:
    headers = auth(tenant, Role.REVIEWER)
    first = await upload(client, headers, marker="same")

    again = await client.post(
        "/documents",
        headers=headers,
        files={"file": ("renamed.pdf", pdf_bytes("same"), "application/pdf")},
    )

    assert again.status_code == 200
    assert again.json()["id"] == first["id"]
    assert await _outbox_count(resources) == 1


async def test_upload_rejects_unsupported_and_oversized_files(
    client: AsyncClient, tenant: SeededTenant, auth: AuthHeaders
) -> None:
    headers = auth(tenant, Role.REVIEWER)

    html = await client.post(
        "/documents",
        headers=headers,
        files={"file": ("x.pdf", b"<html></html>", "application/pdf")},
    )
    huge = await client.post(
        "/documents",
        headers=headers,
        files={"file": ("x.pdf", b"%PDF-" + b"0" * (64 * 1024), "application/pdf")},
    )

    assert html.status_code == 415
    assert huge.status_code == 413


async def test_cursor_pagination_walks_all_pages_without_overlap(
    client: AsyncClient, tenant: SeededTenant, auth: AuthHeaders
) -> None:
    headers = auth(tenant, Role.REVIEWER)
    uploaded = [await upload(client, headers, marker=str(i)) for i in range(5)]

    seen: list[str] = []
    cursor = None
    while True:
        params = {"limit": 2} | ({"cursor": cursor} if cursor else {})
        page = (await client.get("/documents", headers=headers, params=params)).json()
        seen.extend(item["id"] for item in page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            break

    assert seen == [doc["id"] for doc in reversed(uploaded)]


async def test_invalid_cursor_is_a_bad_request(
    client: AsyncClient, tenant: SeededTenant, auth: AuthHeaders
) -> None:
    response = await client.get(
        "/documents", headers=auth(tenant, Role.VIEWER), params={"cursor": "garbage"}
    )
    assert response.status_code == 400


async def test_documents_are_invisible_across_tenants(
    client: AsyncClient,
    tenant: SeededTenant,
    other_tenant: SeededTenant,
    auth: AuthHeaders,
) -> None:
    document = await upload(client, auth(tenant, Role.REVIEWER))
    outsider = auth(other_tenant, Role.ADMIN)

    detail = await client.get(f"/documents/{document['id']}", headers=outsider)
    listing = await client.get("/documents", headers=outsider)

    assert detail.status_code == 404
    assert listing.json() == {"items": [], "next_cursor": None}


async def test_file_download_redirects_to_presigned_url(
    client: AsyncClient, tenant: SeededTenant, auth: AuthHeaders
) -> None:
    document = await upload(client, auth(tenant, Role.REVIEWER))

    response = await client.get(
        f"/documents/{document['id']}/file", headers=auth(tenant, Role.VIEWER)
    )

    assert response.status_code == 307
    assert response.headers["location"].startswith(f"https://storage.test/tenants/{tenant.id}/")


async def test_approval_requires_current_etag(
    client: AsyncClient, tenant: SeededTenant, auth: AuthHeaders
) -> None:
    reviewer = auth(tenant, Role.REVIEWER)
    document = await upload(client, reviewer)
    await extract(client, document, tenant.id, confidence=0.6)
    url = f"/documents/{document['id']}"
    read = await client.get(url, headers=reviewer)
    etag = read.headers["etag"]
    assert etag == f'W/"{read.json()["version"]}"'

    missing = await client.post(f"{url}/approve", headers=reviewer)
    stale = await client.post(f"{url}/approve", headers={**reviewer, "If-Match": 'W/"1"'})
    approved = await client.post(f"{url}/approve", headers={**reviewer, "If-Match": etag})
    replayed = await client.post(f"{url}/approve", headers={**reviewer, "If-Match": etag})

    assert missing.status_code == 428
    assert stale.status_code == 412
    assert stale.json()["current_etag"] == etag
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"
    assert approved.headers["etag"] != etag
    assert replayed.status_code == 412


async def test_correction_creates_human_extraction_version(
    client: AsyncClient, tenant: SeededTenant, auth: AuthHeaders
) -> None:
    reviewer = auth(tenant, Role.REVIEWER)
    document = await upload(client, reviewer)
    await extract(client, document, tenant.id, confidence=0.6)
    url = f"/documents/{document['id']}"
    etag = (await client.get(url, headers=reviewer)).headers["etag"]

    response = await client.patch(
        f"{url}/fields",
        json={"fields": {"total_amount": "1600.00", "supplier_name": "ACME S.A."}},
        headers={**reviewer, "If-Match": etag},
    )

    assert response.status_code == 200
    body = response.json()
    extraction = body["extraction"]
    assert extraction["version"] == 2
    assert extraction["model"] is None
    assert extraction["fields"]["total_amount"] == {
        "value": "1600.00",
        "confidence": 1.0,
        "source": "human",
    }
    assert extraction["fields"]["currency"]["source"] == "llm"
    assert body["total_amount"] == "1600.00"
    assert body["supplier_name"] == "ACME S.A."

    audit = (await client.get(f"{url}/audit", headers=reviewer)).json()
    correction = audit[-1]
    assert correction["action"] == "document.fields_corrected"
    assert correction["actor"]["id"] == str(tenant.accounts[Role.REVIEWER].id)
    assert correction["diff"] == {
        "supplier_name": {"before": "ACME Distribuidora Ltda", "after": "ACME S.A."},
        "total_amount": {"before": "1530.40", "after": "1600.00"},
    }


async def test_invalid_correction_is_rejected_field_by_field(
    client: AsyncClient, tenant: SeededTenant, auth: AuthHeaders
) -> None:
    reviewer = auth(tenant, Role.REVIEWER)
    document = await upload(client, reviewer)
    await extract(client, document, tenant.id, confidence=0.6)
    url = f"/documents/{document['id']}"
    etag = (await client.get(url, headers=reviewer)).headers["etag"]

    response = await client.patch(
        f"{url}/fields",
        json={"fields": {"supplier_tax_id": "11.222.333/0001-80", "total_amount": 10.5}},
        headers={**reviewer, "If-Match": etag},
    )

    assert response.status_code == 422
    assert {e["field"] for e in response.json()["errors"]} == {"supplier_tax_id", "total_amount"}


async def test_reject_from_auto_extracted_state_is_a_conflict(
    client: AsyncClient, tenant: SeededTenant, auth: AuthHeaders
) -> None:
    reviewer = auth(tenant, Role.REVIEWER)
    document = await upload(client, reviewer)
    await extract(client, document, tenant.id, confidence=0.99)
    url = f"/documents/{document['id']}"
    etag = (await client.get(url, headers=reviewer)).headers["etag"]

    response = await client.post(
        f"{url}/reject", json={"reason": "duplicate"}, headers={**reviewer, "If-Match": etag}
    )

    assert response.status_code == 409
    body = response.json()
    assert body["type"] == "urn:ledgerflow:problem:invalid-transition"
    assert (body["current_status"], body["target_status"]) == ("extracted", "rejected")


async def test_audit_trail_records_lifecycle_in_order(
    client: AsyncClient, tenant: SeededTenant, auth: AuthHeaders
) -> None:
    reviewer = auth(tenant, Role.REVIEWER)
    document = await upload(client, reviewer)
    await extract(client, document, tenant.id, confidence=0.6)

    audit = (await client.get(f"/documents/{document['id']}/audit", headers=reviewer)).json()

    assert [event["action"] for event in audit] == [
        "document.uploaded",
        "document.queued",
        "document.processing",
        "document.needs_review",
    ]
    assert audit[-1]["actor"] is None
    assert audit[-1]["diff"]["status"] == {"before": "processing", "after": "needs_review"}
    assert all(event["request_id"] for event in audit)


async def test_stats_reflect_processed_documents(
    client: AsyncClient, tenant: SeededTenant, auth: AuthHeaders
) -> None:
    reviewer = auth(tenant, Role.REVIEWER)
    await extract(client, await upload(client, reviewer, "a"), tenant.id, confidence=0.99)
    await extract(client, await upload(client, reviewer, "b"), tenant.id, confidence=0.5)
    await upload(client, reviewer, "c")

    stats = (await client.get("/stats", headers=auth(tenant, Role.VIEWER))).json()

    assert stats["by_status"]["extracted"] == 1
    assert stats["by_status"]["needs_review"] == 1
    assert stats["by_status"]["queued"] == 1
    assert stats["by_status"]["failed"] == 0
    assert stats["auto_approval_rate"] == 0.5
    assert stats["avg_confidence"] == 0.745
    assert stats["llm_cost_usd"] == "0.000824"
    assert stats["p95_processing_ms"] >= 0
