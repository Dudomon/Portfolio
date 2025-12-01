"""
Vector store backed by ChromaDB with pluggable embeddings.
Falls back to deterministic hashing if transformer model fails to load.
"""
from __future__ import annotations

import hashlib
import uuid
from typing import List, Tuple

import numpy as np
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from .models import Document


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        if model_name.lower() in {"hash", "hash-only"}:
            self.model = None
            self.dim = 256
            return
        try:
            self.model = SentenceTransformer(model_name)
            self.dim = len(self.model.encode(["dim-check"])[0])
        except Exception:
            self.model = None
            self.dim = 256

    def _hash_embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:2], "big") % self.dim
            vec[idx] += 1.0
        if np.linalg.norm(vec) > 0:
            vec /= np.linalg.norm(vec)
        return vec

    def encode(self, texts: List[str]) -> List[List[float]]:
        if self.model:
            try:
                embeddings = self.model.encode(texts, normalize_embeddings=True)
                return [emb.tolist() for emb in embeddings]
            except Exception:
                pass
        return [self._hash_embed(text).tolist() for text in texts]


class Store:
    def __init__(self, path: str, embedder: Embedder):
        settings = ChromaSettings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=path,
            anonymized_telemetry=False,
        )
        try:
            self.client = chromadb.PersistentClient(path=path, settings=settings)
        except Exception:
            # Fallback for chromadb builds without PersistentClient
            self.client = chromadb.Client(settings)
        self.embedder = embedder

    def _collection(self, tenant_id: str):
        return self.client.get_or_create_collection(
            name=f"tenant-{tenant_id}",
            metadata={"tenant": tenant_id},
        )

    def add(self, tenant_id: str, docs: List[Document]) -> None:
        col = self._collection(tenant_id)
        ids = [doc.id or str(uuid.uuid4()) for doc in docs]
        embeddings = self.embedder.encode([doc.text for doc in docs])
        col.add(ids=ids, documents=[doc.text for doc in docs], metadatas=[doc.metadata for doc in docs], embeddings=embeddings)

    def search(self, tenant_id: str, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        col = self._collection(tenant_id)
        embeddings = self.embedder.encode([query])
        results = col.query(query_embeddings=embeddings, n_results=k)
        docs: List[Tuple[Document, float]] = []
        for ids, texts, metas, distances in zip(
            results.get("ids", [[]]),
            results.get("documents", [[]]),
            results.get("metadatas", [[]]),
            results.get("distances", [[]]),
        ):
            for doc_id, text, meta, dist in zip(ids, texts, metas, distances):
                score = 1 / (1 + dist) if dist is not None else 0.0
                docs.append((Document(id=doc_id, text=text, metadata=meta or {}), float(score)))
        return docs
