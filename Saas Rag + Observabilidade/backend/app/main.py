from typing import List
from fastapi import Depends, FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from .llm import LLMClient
from .models import ChatRequest, ChatResponse, IngestRequest, Source
from .observability import instrument_app, setup_tracing
from .security import get_auth, get_rate_limiter
from .settings import Settings, get_settings
from .store import Embedder, Store


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title=settings.app_name)

    if settings.otlp_endpoint:
        setup_tracing(settings.app_name, settings.otlp_endpoint, insecure=settings.otlp_insecure)
    instrument_app(app)
    Instrumentator().instrument(app).expose(app)

    llm_client = LLMClient(settings.openai_api_key)
    embedder = Embedder(settings.embedding_model)
    store = Store(path=settings.chroma_path, embedder=embedder)
    auth = get_auth(settings)
    rate_limit = get_rate_limiter(settings)

    @app.get("/health")
    def health():
        return {"status": "ok", "environment": settings.environment}

    @app.post("/ingest")
    def ingest(payload: IngestRequest, _auth=Depends(auth), _rl=Depends(rate_limit)):
        if not payload.documents:
            raise HTTPException(status_code=400, detail="No documents provided")
        store.add(payload.tenant_id, payload.documents)
        return {"ingested": len(payload.documents)}

    @app.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest, _auth=Depends(auth), _rl=Depends(rate_limit)):
        results = store.search(payload.tenant_id, payload.question, k=settings.top_k)
        sources: List[Source] = [
            Source(
                id=doc.id,
                score=score,
                text=doc.text,
                metadata=doc.metadata,
            )
            for doc, score in results
        ]
        answer = llm_client.answer(payload.question, sources)
        return ChatResponse(answer=answer, sources=sources)

    return app


def get_app(settings: Settings = Depends(get_settings)) -> FastAPI:
    return create_app(settings)


app = create_app(get_settings())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
