"""Request builders shared by the API tests."""

import uuid
from typing import Any

from httpx import AsyncClient

from tests.conftest import TEST_SERVICE_TOKEN

SERVICE_HEADERS = {"Authorization": f"Bearer {TEST_SERVICE_TOKEN}"}


def pdf_bytes(marker: str = "") -> bytes:
    return f"%PDF-1.7\n% test document {marker}\n%%EOF\n".encode()


def extraction_payload(
    document: dict[str, Any], tenant_id: uuid.UUID, *, confidence: float = 0.97, **fields: Any
) -> dict[str, Any]:
    values = {
        "supplier_name": "ACME Distribuidora Ltda",
        "supplier_tax_id": "11.222.333/0001-81",
        "invoice_number": "000123",
        "issue_date": "2026-10-01",
        "due_date": "2026-10-31",
        "currency": "BRL",
        "total_amount": "1530.40",
        "tax_amount": "275.47",
        "nfe_access_key": None,
        "line_items": [
            {"description": "Widget", "quantity": "2", "unit_price": "765.20", "total": "1530.40"}
        ],
    } | fields
    return {
        "event_id": str(uuid.uuid4()),
        "document_id": document["id"],
        "tenant_id": str(tenant_id),
        "doc_type": "invoice",
        "model": "gpt-4o-mini-2024-07-18",
        "prompt_version": "invoice-v3",
        "fields": {
            key: {"value": value, "confidence": confidence, "source": "llm"}
            for key, value in values.items()
        },
        "cost_usd_micros": 412,
        "latency_ms": 1830,
    }


async def upload(client: AsyncClient, headers: dict[str, str], marker: str = "") -> dict[str, Any]:
    response = await client.post(
        "/documents",
        headers=headers,
        files={"file": ("invoice.pdf", pdf_bytes(marker), "application/pdf")},
    )
    assert response.status_code == 202, response.text
    body: dict[str, Any] = response.json()
    return body


async def extract(
    client: AsyncClient, document: dict[str, Any], tenant_id: uuid.UUID, **kwargs: Any
) -> dict[str, Any]:
    """Play the worker: report processing, then post an extraction result."""
    status = await client.post(
        f"/internal/documents/{document['id']}/status",
        json={"status": "processing"},
        headers=SERVICE_HEADERS,
    )
    assert status.status_code == 204, status.text
    payload = extraction_payload(document, tenant_id, **kwargs)
    response = await client.post("/internal/extractions", json=payload, headers=SERVICE_HEADERS)
    assert response.status_code == 204, response.text
    return payload
