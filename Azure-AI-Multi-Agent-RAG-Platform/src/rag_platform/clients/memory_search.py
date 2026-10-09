"""In-process search backend used by tests and the offline retrieval baseline.

It mirrors the shape of the Azure hybrid query: BM25 over the keyword text, cosine
similarity over vectors when both sides have them, fused with Reciprocal Rank Fusion
(the same fusion Azure AI Search uses). There is no semantic reranker, so
`reranker_score` is always `None`, and OData filters are not supported.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from rag_platform.clients.protocols import SearchQuery
from rag_platform.domain import RetrievedChunk

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORD_TEXT = (
    "a an and are as at be by can do does for from how i in is it must of on or our "
    "the to we what when which who with within"
)
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())
_RRF_K = 60


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


class InMemorySearchIndex:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self._documents: dict[str, dict[str, Any]] = {}
        self._k1 = k1
        self._b = b

    @property
    def documents(self) -> Mapping[str, Mapping[str, Any]]:
        return self._documents

    async def ensure_index(self) -> None:
        return None

    async def chunk_hashes(self, doc_id: str) -> dict[str, str]:
        return {
            key: str(doc["content_hash"])
            for key, doc in self._documents.items()
            if doc["doc_id"] == doc_id
        }

    async def upsert(self, documents: Sequence[Mapping[str, Any]]) -> None:
        # Merge semantics, matching Azure AI Search `mergeOrUpload`.
        for document in documents:
            self._documents.setdefault(str(document["id"]), {}).update(document)

    async def delete(self, chunk_ids: Sequence[str]) -> None:
        for chunk_id in chunk_ids:
            self._documents.pop(chunk_id, None)

    def _bm25_ranking(self, query: str) -> list[str]:
        terms = tokenize(query)
        corpus = {
            key: tokenize(f"{d['title']} {d['heading_path']} {d['content']}")
            for key, d in self._documents.items()
        }
        if not terms or not corpus:
            return []
        average_length = sum(len(tokens) for tokens in corpus.values()) / len(corpus)
        document_frequency = Counter(t for tokens in corpus.values() for t in set(tokens))
        scores: dict[str, float] = {}
        for key, tokens in corpus.items():
            frequencies = Counter(tokens)
            score = 0.0
            for term in terms:
                if term not in frequencies:
                    continue
                df = document_frequency[term]
                idf = math.log(1 + (len(corpus) - df + 0.5) / (df + 0.5))
                tf = frequencies[term]
                length_norm = 1 - self._b + self._b * len(tokens) / average_length
                score += idf * tf * (self._k1 + 1) / (tf + self._k1 * length_norm)
            if score > 0:
                scores[key] = score
        return sorted(scores, key=lambda k: (-scores[k], k))

    def _vector_ranking(self, vector: Sequence[float]) -> list[str]:
        if not vector:
            return []
        scored = {
            key: _cosine(vector, doc["content_vector"])
            for key, doc in self._documents.items()
            if doc.get("content_vector")
        }
        return sorted(scored, key=lambda k: (-scored[k], k))

    async def hybrid_search(self, query: SearchQuery) -> list[RetrievedChunk]:
        fused: dict[str, float] = {}
        for ranking in (
            self._bm25_ranking(query.keyword_text),
            self._vector_ranking(query.vector),
        ):
            for rank, key in enumerate(ranking, start=1):
                fused[key] = fused.get(key, 0.0) + 1.0 / (_RRF_K + rank)
        ordered = sorted(fused, key=lambda k: (-fused[k], k))[: query.top]
        return [
            RetrievedChunk(
                chunk_id=key,
                doc_id=self._documents[key]["doc_id"],
                title=self._documents[key]["title"],
                heading_path=self._documents[key]["heading_path"],
                content=self._documents[key]["content"],
                source_path=self._documents[key]["source_path"],
                score=fused[key],
            )
            for key in ordered
        ]
