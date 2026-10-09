from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_platform.agents.citations import as_citation
from rag_platform.agents.messages import AnswerResult
from rag_platform.evaluation.dataset import GoldenItem, load_golden, load_thresholds
from rag_platform.evaluation.judge import JudgeError, LlmJudge, load_rubrics
from rag_platform.evaluation.metrics import (
    citation_precision,
    citation_recall,
    mean,
    recall_at_k,
    reciprocal_rank,
    threshold_failures,
)
from rag_platform.evaluation.report import render_report
from rag_platform.evaluation.runner import run_full_suite, run_retrieval_suite
from rag_platform.ingestion.loaders import load_documents

from conftest import PROJECT_ROOT, FakeChatModel, FakeEmbedder, StaticSearch, chunk

EVALS = PROJECT_ROOT / "evals"


def test_recall_at_k_uses_distinct_documents() -> None:
    ranked = ["A", "A", "B", "C", "D"]

    assert recall_at_k(ranked, ["B", "D"], k=2) == 0.5
    assert recall_at_k(ranked, ["B", "D"], k=4) == 1.0
    with pytest.raises(ValueError, match="expected"):
        recall_at_k(ranked, [], k=3)


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["A", "A", "B"], ["B"]) == 0.5
    assert reciprocal_rank(["A"], ["Z"]) == 0.0


def test_citation_metrics() -> None:
    assert citation_precision(["A", "B"], ["A"]) == 0.5
    assert citation_precision([], ["A"]) is None
    assert citation_recall(["A"], ["A", "B"]) == 0.5


def test_mean_ignores_undefined_values() -> None:
    assert mean([1.0, None, 0.0]) == 0.5
    assert mean([None]) is None


def test_threshold_failures_flag_low_and_missing_metrics() -> None:
    failures = threshold_failures(
        {"recall_at_5": 0.8, "mrr": None, "groundedness": 0.95},
        {"recall_at_5": 0.9, "mrr": 0.7, "groundedness": 0.85},
    )

    assert failures == ["mrr: not measured (threshold 0.70)", "recall_at_5: 0.800 < 0.90"]


def test_golden_set_references_existing_documents(corpus_dir: Path) -> None:
    items = load_golden(EVALS / "golden.jsonl")
    corpus_ids = {d.doc_id for d in load_documents(corpus_dir)}

    assert len(items) >= 20
    assert any(not item.answerable for item in items)
    for item in items:
        assert set(item.expected_doc_ids) <= corpus_ids, item.id


def test_thresholds_and_rubrics_load() -> None:
    thresholds = load_thresholds(EVALS / "thresholds.toml")
    rubrics = load_rubrics(EVALS / "rubrics")

    assert "recall_at_5" in thresholds.retrieval
    assert "groundedness" in thresholds.generation
    assert set(rubrics) == {"groundedness", "answer_relevance"}
    assert rubrics["groundedness"].version == 1
    # Rubrics use $placeholders; every one must be fillable by the runner.
    rubrics["groundedness"].template.substitute(question="q", answer="a", sources="s")
    rubrics["answer_relevance"].template.substitute(question="q", answer="a", reference_answer="r")


def test_golden_item_requires_consistent_answerability() -> None:
    with pytest.raises(ValueError, match="answerable"):
        GoldenItem(id="x", question="q", reference_answer="r", category="c")


ITEMS = [
    GoldenItem(
        id="a1",
        question="How many quotes?",
        expected_doc_ids=("PROC-001",),
        reference_answer="Three quotes above USD 25,000.",
        category="procurement",
    ),
    GoldenItem(
        id="n1",
        question="Parental leave?",
        reference_answer="Not covered.",
        category="out-of-scope",
        answerable=False,
    ),
]

HITS = [chunk("P-1", "PROC-001", "Three quotes above USD 25,000."), chunk("T-1", "TAX-002", "x")]


async def test_retrieval_suite_scores_answerable_items_only() -> None:
    suite = await run_retrieval_suite(ITEMS, StaticSearch(HITS), FakeEmbedder(), k=5)

    summary = suite.summary()
    assert summary == {"recall_at_5": 1.0, "mrr": 1.0}
    assert suite.items[1].recall_at_k is None


async def test_full_suite_scores_answers_and_abstention() -> None:
    async def answer(question: str) -> AnswerResult:
        if question.startswith("Parental"):
            return AnswerResult(question=question, status="insufficient_evidence", answer="n/a")
        cited = as_citation(1, HITS[0])
        return AnswerResult(
            question=question,
            status="answered",
            answer="Three quotes are required [1].",
            citations=(cited,),
            sources=(cited, as_citation(2, HITS[1])),
            cost_usd=0.002,
        )

    judge_chat = FakeChatModel(
        {
            "judge_groundedness": [json.dumps({"score": 5, "rationale": "all grounded"})],
            "judge_answer_relevance": [json.dumps({"score": 4, "rationale": "minor gap"})],
        }
    )
    judge = LlmJudge(judge_chat, "gpt-4o", load_rubrics(EVALS / "rubrics"))

    suite = await run_full_suite(ITEMS, answer, judge, k=5)

    summary = suite.summary()
    assert summary["groundedness"] == 1.0
    assert summary["answer_relevance"] == 0.75
    assert summary["citation_precision"] == 1.0
    assert summary["abstention_accuracy"] == 1.0
    assert summary["mrr"] == 1.0
    judge_prompt = judge_chat.calls("judge_groundedness")[0].messages[0].content
    assert "Three quotes above USD 25,000." in judge_prompt

    report = render_report(suite, {"groundedness": 0.85, "answer_relevance": 0.8}, ["x"], {})
    assert "| answer_relevance | 0.750 | 0.80 | FAIL |" in report
    assert "**Gate: FAILED**" in report
    assert "### a1" in report


async def test_judge_rejects_out_of_range_scores() -> None:
    chat = FakeChatModel({"judge_groundedness": [json.dumps({"score": 9, "rationale": "?"})]})
    judge = LlmJudge(chat, "gpt-4o", load_rubrics(EVALS / "rubrics"))

    with pytest.raises(JudgeError):
        await judge.score("groundedness", question="q", answer="a", sources="s")
