from fastapi import APIRouter

from ledgerflow_api.api.v1.routers import auth, documents, internal, me, stats

api_router = APIRouter(prefix="/api/v1")
for module in (auth, me, documents, stats, internal):
    api_router.include_router(module.router)
