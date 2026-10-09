from httpx import AsyncClient

from ledgerflow_api.domain.roles import Role
from tests.api.conftest import PASSWORD, AuthHeaders, SeededTenant


async def test_login_issues_token_accepted_by_me(client: AsyncClient, tenant: SeededTenant) -> None:
    reviewer = tenant.accounts[Role.REVIEWER]

    login = await client.post(
        "/auth/token", json={"email": reviewer.email.upper(), "password": PASSWORD}
    )

    assert login.status_code == 200
    body = login.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 900
    me = await client.get("/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.status_code == 200
    assert me.json() == {
        "id": str(reviewer.id),
        "tenant_id": str(tenant.id),
        "email": reviewer.email,
        "name": "Reviewer of acme",
        "role": "reviewer",
    }


async def test_wrong_password_and_unknown_email_are_indistinguishable(
    client: AsyncClient, tenant: SeededTenant
) -> None:
    wrong_password = await client.post(
        "/auth/token", json={"email": tenant.accounts[Role.ADMIN].email, "password": "nope"}
    )
    unknown_email = await client.post(
        "/auth/token", json={"email": "ghost@acme.example.com", "password": PASSWORD}
    )

    for response in (wrong_password, unknown_email):
        assert response.status_code == 401
        assert response.headers["content-type"] == "application/problem+json"
    assert wrong_password.json()["detail"] == unknown_email.json()["detail"]


async def test_me_rejects_tampered_token(
    client: AsyncClient, tenant: SeededTenant, auth: AuthHeaders
) -> None:
    headers = auth(tenant, Role.VIEWER)
    headers["Authorization"] = headers["Authorization"][:-2] + "xx"

    response = await client.get("/me", headers=headers)

    assert response.status_code == 401
