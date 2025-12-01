import tempfile

from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings


def make_app(tmp_path: str):
    settings = Settings(
        api_keys="test-key",
        rate_limit_per_minute=100,
        chroma_path=tmp_path,
        embedding_model="hash",
        openai_api_key=None,
    )
    return create_app(settings)


def test_ingest_and_chat_returns_sources():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        app = make_app(tmpdir)
        client = TestClient(app)
        headers = {"X-API-Key": "test-key"}

        resp = client.post(
            "/ingest",
            json={
                "tenant_id": "t1",
                "documents": [{"id": "doc-1", "text": "Support hours are 9 to 6."}],
            },
            headers=headers,
        )
        assert resp.status_code == 200
        assert resp.json()["ingested"] == 1

        chat_resp = client.post(
            "/chat",
            json={"tenant_id": "t1", "question": "What are support hours?"},
            headers=headers,
        )
        assert chat_resp.status_code == 200
        data = chat_resp.json()
        assert data["sources"]
        assert "Support" in data["sources"][0]["text"]
        assert data["answer"]


def test_auth_rejects_missing_key():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        app = make_app(tmpdir)
        client = TestClient(app)
        resp = client.post(
            "/ingest",
            json={"tenant_id": "t1", "documents": [{"text": "hi"}]},
        )
        assert resp.status_code == 401
