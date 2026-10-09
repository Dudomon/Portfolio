from httpx import AsyncClient

from ledgerflow_api.domain.roles import Role
from tests.api.conftest import AuthHeaders, SeededTenant
from tests.api.factories import extract, pdf_bytes, upload


async def test_viewer_can_read_but_not_upload(
    client: AsyncClient, tenant: SeededTenant, auth: AuthHeaders
) -> None:
    viewer = auth(tenant, Role.VIEWER)

    listed = await client.get("/documents", headers=viewer)
    uploaded = await client.post(
        "/documents", headers=viewer, files={"file": ("a.pdf", pdf_bytes(), "application/pdf")}
    )

    assert listed.status_code == 200
    assert uploaded.status_code == 403
    assert uploaded.json()["type"] == "urn:ledgerflow:problem:forbidden"


async def test_viewer_cannot_approve(
    client: AsyncClient, tenant: SeededTenant, auth: AuthHeaders
) -> None:
    document = await upload(client, auth(tenant, Role.REVIEWER))
    await extract(client, document, tenant.id, confidence=0.5)
    viewer = auth(tenant, Role.VIEWER)
    etag = (await client.get(f"/documents/{document['id']}", headers=viewer)).headers["etag"]

    response = await client.post(
        f"/documents/{document['id']}/approve", headers={**viewer, "If-Match": etag}
    )

    assert response.status_code == 403
    detail = await client.get(f"/documents/{document['id']}", headers=viewer)
    assert detail.json()["status"] == "needs_review"


async def test_only_admin_can_retry(
    client: AsyncClient, tenant: SeededTenant, auth: AuthHeaders
) -> None:
    document = await upload(client, auth(tenant, Role.REVIEWER))

    response = await client.post(
        f"/documents/{document['id']}/retry", headers=auth(tenant, Role.REVIEWER)
    )

    assert response.status_code == 403
