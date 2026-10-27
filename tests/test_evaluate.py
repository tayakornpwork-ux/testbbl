"""Tests for the evaluation metrics, plus a retrieval-quality regression guard."""

from __future__ import annotations

import pytest

from evaluate import (
    DEFAULT_CASES_PATH,
    HOLDOUT_CASES_PATH,
    EndToEndResult,
    EvalCase,
    RetrievalScore,
    load_cases,
    score_retrieval,
)
from grounding import GroundingReport
from retriever_tool import BM25Retriever


def test_retrieval_score_metrics() -> None:
    case = EvalCase("q", ("kb-02", "kb-05"))
    score = RetrievalScore(case, ("kb-01", "kb-02", "kb-03"))
    assert not score.hit_at_1
    assert score.recall == 0.5
    assert score.reciprocal_rank == 0.5
    assert RetrievalScore(case, ()).reciprocal_rank == 0.0


def test_end_to_end_pass_rules() -> None:
    answered = GroundingReport(False, ("kb-04",), (), ())
    abstained = GroundingReport(True, (), (), ())
    assert EndToEndResult(EvalCase("q", ("kb-04",)), ("kb-04",), answered).passed
    assert not EndToEndResult(EvalCase("q", ("kb-01",)), ("kb-04",), answered).passed
    assert EndToEndResult(EvalCase("q", ()), (), abstained).passed
    assert not EndToEndResult(EvalCase("q", ()), ("kb-04",), answered).passed


@pytest.mark.parametrize(
    ("path", "min_hit_at_1"), [(DEFAULT_CASES_PATH, 1.0), (HOLDOUT_CASES_PATH, 0.9)]
)
def test_bundled_cases_meet_retrieval_quality_bar(path, min_hit_at_1: float) -> None:
    report = score_retrieval(BM25Retriever.from_file(), load_cases(path))
    assert report.hit_at_1 >= min_hit_at_1
    assert report.recall_at_k == 1.0
