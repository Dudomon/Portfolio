"""Interpretation of Azure OpenAI content filter results.

Azure OpenAI reports filtering in two places:

* the *prompt* was blocked: HTTP 400 with `code == "content_filter"` and the per-category
  results under `innererror.content_filter_result`;
* the *completion* was cut: `finish_reason == "content_filter"` with per-category results
  under `choices[].content_filter_results`.

Both are normalized into `ContentFilteredError` so the orchestrator can return a policy
refusal instead of surfacing a raw 400 or a silently truncated answer.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

FilterStage = Literal["prompt", "completion"]


class ContentFilteredError(Exception):
    def __init__(self, stage: FilterStage, categories: tuple[str, ...]) -> None:
        self.stage = stage
        self.categories = categories
        label = ", ".join(categories) or "unspecified"
        super().__init__(f"Azure OpenAI content filter triggered on {stage}: {label}")


def filtered_categories(results: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Return the sorted category names whose result is marked as filtered."""
    if not results:
        return ()
    return tuple(
        sorted(
            name
            for name, outcome in results.items()
            if isinstance(outcome, Mapping) and outcome.get("filtered") is True
        )
    )


def prompt_filter_error(error_body: object) -> ContentFilteredError | None:
    """Build a `ContentFilteredError` from a 400 error body, if it is a content filter rejection.

    The SDK exposes either the full payload (`{"error": {...}}`) or just the inner error
    object depending on the version, so both shapes are accepted.
    """
    if not isinstance(error_body, Mapping):
        return None
    error = error_body.get("error", error_body)
    if not isinstance(error, Mapping) or error.get("code") != "content_filter":
        return None
    inner = error.get("innererror")
    results = inner.get("content_filter_result") if isinstance(inner, Mapping) else None
    return ContentFilteredError(
        "prompt", filtered_categories(results if isinstance(results, Mapping) else None)
    )
