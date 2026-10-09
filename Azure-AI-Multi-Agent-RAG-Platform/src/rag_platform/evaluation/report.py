"""Markdown report rendering for evaluation runs."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime

from rag_platform.evaluation.metrics import unique_in_order
from rag_platform.evaluation.runner import ItemResult, SuiteResult


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _item_row(item: ItemResult, k: int, full: bool) -> str:
    cells = [
        item.id,
        item.category,
        "answerable" if item.answerable else "unanswerable",
        _fmt(item.recall_at_k, 2),
        _fmt(item.reciprocal_rank, 2),
    ]
    if full:
        cells += [
            item.status or "-",
            _fmt(item.citation_precision, 2),
            _fmt(item.groundedness, 2),
            _fmt(item.answer_relevance, 2),
            "-" if item.abstained is None else ("yes" if item.abstained else "no"),
        ]
    cells.append(", ".join(unique_in_order(item.retrieved_doc_ids)[:k]) or "-")
    return "| " + " | ".join(cells) + " |"


def render_report(
    suite: SuiteResult,
    thresholds: Mapping[str, float],
    failures: list[str],
    run_metadata: Mapping[str, str],
) -> str:
    summary = suite.summary()
    full = suite.mode == "full"
    headers = [
        "Id",
        "Category",
        "Type",
        f"Recall@{suite.k}",
        "RR",
        *(
            ["Status", "Citation precision", "Groundedness", "Relevance", "Abstained"]
            if full
            else []
        ),
        "Top documents",
    ]
    lines = [
        f"# Evaluation report ({suite.mode})",
        "",
        f"Generated {datetime.now(tz=UTC).strftime('%Y-%m-%d %H:%M UTC')}.",
        "",
        "## Run configuration",
        "",
        *(f"- **{key}**: {value}" for key, value in run_metadata.items()),
        "",
        "## Summary",
        "",
        "| Metric | Value | Threshold | Result |",
        "| --- | --- | --- | --- |",
    ]
    for metric, value in summary.items():
        minimum = thresholds.get(metric)
        if minimum is None:
            verdict, shown = "info", "-"
        else:
            verdict = "pass" if value is not None and value >= minimum else "FAIL"
            shown = f"{minimum:.2f}"
        lines.append(f"| {metric} | {_fmt(value)} | {shown} | {verdict} |")
    lines += [
        "",
        f"**Gate: {'FAILED' if failures else 'PASSED'}**",
        "",
        *(f"- {failure}" for failure in failures),
        "",
        "## Items",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + " --- |" * len(headers),
        *(_item_row(item, suite.k, full) for item in suite.items),
    ]
    noted = [item for item in suite.items if item.judge_notes]
    if noted:
        lines += ["", "## Judge rationales", ""]
        for item in noted:
            lines.append(f"### {item.id}")
            lines += [f"- {note}" for note in item.judge_notes]
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
