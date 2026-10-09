"""Deterministic evaluation metrics.

Retrieval is scored at document level: the golden set names the documents that
contain the answer, not individual chunks, because chunk ids change whenever the
chunking strategy is tuned and the golden set must survive that.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence


def unique_in_order(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def recall_at_k(ranked_doc_ids: Sequence[str], expected: Iterable[str], k: int) -> float:
    """Fraction of expected documents that appear in the top `k` distinct retrieved documents."""
    expected_set = set(expected)
    if not expected_set:
        raise ValueError("recall is undefined without expected documents")
    top = set(unique_in_order(ranked_doc_ids)[:k])
    return len(top & expected_set) / len(expected_set)


def reciprocal_rank(ranked_doc_ids: Sequence[str], expected: Iterable[str]) -> float:
    """1/rank of the first relevant distinct document; 0 when none is retrieved."""
    expected_set = set(expected)
    for rank, doc_id in enumerate(unique_in_order(ranked_doc_ids), start=1):
        if doc_id in expected_set:
            return 1.0 / rank
    return 0.0


def citation_precision(cited_doc_ids: Iterable[str], expected: Iterable[str]) -> float | None:
    """Share of cited documents that are expected sources; `None` when nothing is cited."""
    cited = set(cited_doc_ids)
    if not cited:
        return None
    return len(cited & set(expected)) / len(cited)


def citation_recall(cited_doc_ids: Iterable[str], expected: Iterable[str]) -> float:
    expected_set = set(expected)
    if not expected_set:
        raise ValueError("citation recall is undefined without expected documents")
    return len(set(cited_doc_ids) & expected_set) / len(expected_set)


def mean(values: Iterable[float | None]) -> float | None:
    """Mean over defined values; `None` when no value is defined."""
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def threshold_failures(
    summary: Mapping[str, float | None], thresholds: Mapping[str, float]
) -> list[str]:
    """Describe every metric that is below its threshold or was not measured."""
    failures = []
    for metric, minimum in sorted(thresholds.items()):
        value = summary.get(metric)
        if value is None:
            failures.append(f"{metric}: not measured (threshold {minimum:.2f})")
        elif value < minimum:
            failures.append(f"{metric}: {value:.3f} < {minimum:.2f}")
    return failures
