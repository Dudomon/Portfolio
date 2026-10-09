import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import text

from ledgerflow_api.api.deps import AppResources
from ledgerflow_api.db.session import tenant_session
from ledgerflow_api.domain.roles import Role
from tests.api.conftest import AuthHeaders, SeededTenant
from tests.api.factories import SERVICE_HEADERS, extract, extraction_payload, upload

VALID_NFE_KEY = "35261011222333000181550010000000011000000010"


async def test_extraction_is_idempotent_on_event_id(
    client: AsyncClient, tenant: SeededTenant, auth: AuthHeaders
) -> None:
    reviewer = auth(tenant, Role.REVIEWER)
    document = await upload(client, reviewer)
    payload = await extract(client, document, tenant.id)
    audit_before = (await client.get(f"/documents/{document['id']}/audit", headers=reviewer)).json()

    redelivered = await client.post("/internal/extractions", json=payload, headers=SERVICE_HEADERS)

    assert redelivered.status_code == 204
    detail = (await client.get(f"/documents/{document['id']}", headers=reviewer)).json()
    assert detail["extraction"]["version"] == 1
    audit_after = (await client.get(f"/documents/{document['id']}/audit", headers=reviewer)).json()
    assert audit_after == audit_before


@pytest.mark.parametrize(
    ("confidence", "overrides", "expected"),
    [
        (0.97, {}, "extracted"),
        (0.84, {}, "needs_review"),
        # Confident but wrong: a failed check digit forces review regardless.
        (0.99, {"supplier_tax_id": "11.222.333/0001-80"}, "needs_review"),
        (0.99, {"tax_amount": "9999.00"}, "needs_review"),
    ],
)
async def test_threshold_and_rules_decide_review(
    client: AsyncClient,
    tenant: SeededTenant,
    auth: AuthHeaders,
    confidence: float,
    overrides: dict[str, str],
    expected: str,
) -> None:
    reviewer = auth(tenant, Role.REVIEWER)
    document = await upload(client, reviewer)

    await extract(client, document, tenant.id, confidence=confidence, **overrides)

    detail = (await client.get(f"/documents/{document['id']}", headers=reviewer)).json()
    assert detail["status"] == expected
    assert detail["overall_confidence"] == confidence
    assert detail["total_amount"] == "1530.40"


async def test_nfe_extraction_with_consistent_key_is_auto_approvable(
    client: AsyncClient, tenant: SeededTenant, auth: AuthHeaders
) -> None:
    reviewer = auth(tenant, Role.REVIEWER)
    document = await upload(client, reviewer)
    payload = extraction_payload(document, tenant.id, nfe_access_key=VALID_NFE_KEY)
    payload["doc_type"] = "nfe"

    response = await client.post("/internal/extractions", json=payload, headers=SERVICE_HEADERS)

    assert response.status_code == 204
    detail = (await client.get(f"/documents/{document['id']}", headers=reviewer)).json()
    assert (detail["status"], detail["doc_type"]) == ("extracted", "nfe")
    assert detail["extraction"]["fields"]["supplier_tax_id"]["value"] == "11222333000181"


async def test_extraction_for_wrong_tenant_is_not_found(
    client: AsyncClient, tenant: SeededTenant, other_tenant: SeededTenant, auth: AuthHeaders
) -> None:
    document = await upload(client, auth(tenant, Role.REVIEWER))
    payload = extraction_payload(document, other_tenant.id)

    response = await client.post("/internal/extractions", json=payload, headers=SERVICE_HEADERS)

    assert response.status_code == 404


async def test_unknown_field_keys_are_rejected(
    client: AsyncClient, tenant: SeededTenant, auth: AuthHeaders
) -> None:
    document = await upload(client, auth(tenant, Role.REVIEWER))
    payload = extraction_payload(document, tenant.id)
    payload["fields"]["iban"] = {"value": "x", "confidence": 0.9, "source": "llm"}

    response = await client.post("/internal/extractions", json=payload, headers=SERVICE_HEADERS)

    assert response.status_code == 422
    assert response.json()["unknown_fields"] == ["iban"]


async def test_failure_and_admin_retry(
    client: AsyncClient, tenant: SeededTenant, auth: AuthHeaders
) -> None:
    document = await upload(client, auth(tenant, Role.REVIEWER))
    status_url = f"/internal/documents/{document['id']}/status"
    for _ in range(2):  # redelivered callbacks are harmless
        processing = await client.post(
            status_url, json={"status": "processing"}, headers=SERVICE_HEADERS
        )
        assert processing.status_code == 204
    failed = await client.post(
        status_url, json={"status": "failed", "error": "OCR timeout"}, headers=SERVICE_HEADERS
    )
    assert failed.status_code == 204

    retried = await client.post(
        f"/documents/{document['id']}/retry", headers=auth(tenant, Role.ADMIN)
    )

    assert retried.status_code == 202
    audit = (
        await client.get(f"/documents/{document['id']}/audit", headers=auth(tenant, Role.ADMIN))
    ).json()
    assert audit[-1]["action"] == "document.requeued"
    assert audit[-1]["diff"]["last_error"] == {"before": "OCR timeout", "after": None}


async def test_worker_cannot_report_review_decisions(
    client: AsyncClient, tenant: SeededTenant, auth: AuthHeaders
) -> None:
    document = await upload(client, auth(tenant, Role.REVIEWER))

    response = await client.post(
        f"/internal/documents/{document['id']}/status",
        json={"status": "approved"},
        headers=SERVICE_HEADERS,
    )

    assert response.status_code == 422


async def test_app_role_cannot_rewrite_audit_history(
    resources: AppResources, tenant: SeededTenant, client: AsyncClient, auth: AuthHeaders
) -> None:
    await upload(client, auth(tenant, Role.REVIEWER))

    async with tenant_session(resources.session_factory, tenant.id) as session:
        visible = await session.scalar(text("SELECT count(*) FROM audit_events"))
        assert visible == 2
        with pytest.raises(Exception, match="permission denied"):
            await session.execute(text("UPDATE audit_events SET action = 'tampered'"))


async def test_tenant_session_sees_only_its_own_rows(
    resources: AppResources,
    tenant: SeededTenant,
    other_tenant: SeededTenant,
    client: AsyncClient,
    auth: AuthHeaders,
) -> None:
    await upload(client, auth(tenant, Role.REVIEWER))

    async with tenant_session(resources.session_factory, other_tenant.id) as session:
        documents = await session.scalar(text("SELECT count(*) FROM documents"))
        users = await session.scalar(text("SELECT count(*) FROM users"))
    async with tenant_session(resources.session_factory, uuid.uuid4()) as session:
        nobody = await session.scalar(text("SELECT count(*) FROM users"))

    assert (documents, users, nobody) == (0, 3, 0)
