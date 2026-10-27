"""Cheap post-hoc checks that a generated answer is grounded in its snippets.

These are heuristics, not proof: they catch the most common faithfulness failures
(citing a chunk that was never retrieved, or quoting a number that appears in no
snippet) without a second LLM call.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from prompts import NOT_FOUND_MESSAGE
from retriever_tool import Snippet

_CITATION_RE = re.compile(r"\bkb-\d{2}\b")
_LIST_MARKER_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_NUMBER_WORDS = {
    word: str(value)
    for value, words in enumerate(
        [
            (),
            ("one", "first"),
            ("two", "second"),
            ("three", "third"),
            ("four", "fourth"),
            ("five", "fifth"),
            ("six", "sixth"),
            ("seven", "seventh"),
            ("eight", "eighth"),
            ("nine", "ninth"),
            ("ten", "tenth"),
            ("eleven", "eleventh"),
            ("twelve", "twelfth"),
        ]
    )
    for word in words
}
_NUMBER_WORD_RE = re.compile(rf"\b({'|'.join(_NUMBER_WORDS)})\b", re.IGNORECASE)


@dataclass(frozen=True)
class GroundingReport:
    """Result of checking one answer against the snippets it was written from."""

    abstained: bool
    cited_ids: tuple[str, ...]
    unknown_citations: tuple[str, ...]
    unsupported_numbers: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True if the answer abstained, or cites only real chunks and real numbers."""
        if self.abstained:
            return True
        return bool(self.cited_ids) and not self.unknown_citations and not self.unsupported_numbers

    def warnings(self) -> list[str]:
        """Human-readable descriptions of every problem found."""
        if self.abstained:
            return []
        problems = []
        if not self.cited_ids:
            problems.append("The answer cites no chunk ids.")
        if self.unknown_citations:
            problems.append(
                f"Cites chunks that were not retrieved: {', '.join(self.unknown_citations)}"
            )
        if self.unsupported_numbers:
            problems.append(
                f"Numbers not found in any snippet: {', '.join(self.unsupported_numbers)}"
            )
        return problems


def _numbers(text: str, spelled_out: bool = False) -> set[str]:
    """Extract numbers, normalising '1,000' to '1000'.

    With ``spelled_out``, number words also count ('fourth' -> '4'). That is used for
    the sources only: in an answer, words like "two options" are prose, not facts.
    """
    if spelled_out:
        text = _NUMBER_WORD_RE.sub(lambda m: _NUMBER_WORDS[m.group(1).lower()], text)
    return {n.replace(",", "") for n in _NUMBER_RE.findall(text)}


def check_grounding(answer: str, snippets: Sequence[Snippet], query: str = "") -> GroundingReport:
    """Check that ``answer`` only cites retrieved chunks and only uses known numbers.

    A number is known if it appears in a snippet or in the user's ``query``:
    repeating the user's own figure ("a claim of 1,500 USD") is not a hallucination.
    """
    if answer.strip() == NOT_FOUND_MESSAGE:
        return GroundingReport(True, (), (), ())

    retrieved_ids = {s["id"] for s in snippets}
    cited = tuple(dict.fromkeys(_CITATION_RE.findall(answer)))
    unknown = tuple(c for c in cited if c not in retrieved_ids)

    prose = _LIST_MARKER_RE.sub("", _CITATION_RE.sub("", answer))
    sources = [s["text"] for s in snippets] + [query]
    known_numbers = set().union(*(_numbers(text, spelled_out=True) for text in sources))
    unsupported = tuple(sorted(_numbers(prose) - known_numbers, key=float))
    return GroundingReport(False, cited, unknown, unsupported)
