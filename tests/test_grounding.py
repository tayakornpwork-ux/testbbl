"""Tests for the post-hoc grounding check. No LLM is needed."""

from __future__ import annotations

from grounding import check_grounding
from prompts import NOT_FOUND_MESSAGE
from retriever_tool import Snippet

SICK_LEAVE = Snippet(
    id="kb-05",
    title="Sick Leave",
    text=(
        "Sick Leave: Notify your manager before 10:00. Self-certify up to 3 days; from the "
        "fourth day a doctor's note is required. 15 days on full pay, then 30 days at 50 "
        "percent pay. Claims over 1,000 USD are reviewed."
    ),
    score=3.2,
)

GOOD_ANSWER = """**A doctor's note is needed from the 4th day.**

### Notifying
1. Tell your manager before 10:00 [kb-05]
- Self-certify up to 3 days [kb-05]
- 15 days full pay, then 30 days at 50% [kb-05]
- Claims over 1000 USD are reviewed [kb-05]

Sources: kb-05"""


def test_faithful_answer_passes() -> None:
    report = check_grounding(GOOD_ANSWER, [SICK_LEAVE])
    assert report.ok
    assert report.cited_ids == ("kb-05",)
    assert report.warnings() == []


def test_unretrieved_citation_is_flagged() -> None:
    report = check_grounding(GOOD_ANSWER + "\n- See also [kb-04]", [SICK_LEAVE])
    assert not report.ok
    assert report.unknown_citations == ("kb-04",)


def test_invented_number_is_flagged() -> None:
    report = check_grounding(GOOD_ANSWER + "\n- You get 20 sick days [kb-05]", [SICK_LEAVE])
    assert report.unsupported_numbers == ("20",)
    assert "20" in report.warnings()[0]


def test_answer_without_citations_is_flagged() -> None:
    report = check_grounding("**You get 15 days.**", [SICK_LEAVE])
    assert not report.ok
    assert report.warnings() == ["The answer cites no chunk ids."]


def test_not_found_answer_counts_as_abstaining() -> None:
    report = check_grounding(NOT_FOUND_MESSAGE, [])
    assert report.abstained
    assert report.ok


def test_numbers_from_the_question_are_allowed() -> None:
    answer = "**A claim of 1,500 USD is paid within 10 working days.** [kb-05]"
    snippet = Snippet(
        id="kb-05", title="t", text="Claims over 1,000 USD: 10 working days.", score=1
    )
    assert check_grounding(answer, [snippet]).unsupported_numbers == ("1500",)
    assert check_grounding(answer, [snippet], query="A 1,500 USD claim?").ok


def test_number_words_in_the_answer_are_not_treated_as_facts() -> None:
    answer = "**The snippets give two different deadlines.** [kb-05]"
    assert check_grounding(answer, [SICK_LEAVE]).ok
