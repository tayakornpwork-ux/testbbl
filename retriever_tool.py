"""Custom RAG tool: paragraph chunking and BM25 ranking over a local knowledge base.

The retriever is deliberately dependency-free: BM25 is about thirty lines of code,
and owning it lets us use the Lucene-style IDF, which never goes negative. The
classic Okapi IDF used by ``rank_bm25.BM25Okapi`` turns negative for any term that
appears in more than half of the documents, which is common in a corpus of only a
few paragraphs and makes a minimum-score threshold unreliable.

Two refinements on top of plain BM25, both measured with ``evaluate.py``:

* Title boost: words in a chunk's topic title ("Sick Leave") count extra, so the
  chunk about a topic outranks chunks that merely mention it.
* Query synonyms: everyday words the policies never use ("vacation", "WFH",
  "per diem") are expanded to the policy vocabulary at a reduced weight.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from langchain_core.tools import BaseTool, tool

DEFAULT_KB_PATH = Path(__file__).with_name("knowledge_base.txt")
DEFAULT_TOP_K = 3
DEFAULT_MIN_SCORE = 1.0
DEFAULT_TITLE_BOOST = 2
SYNONYM_WEIGHT = 0.5
NO_RESULTS_MESSAGE = "No relevant chunks found. Try again with different keywords or synonyms."

_STOPWORD_TEXT = """
    a about after all also am an and any are as at be been before being but by can
    could did do does doing each every for from get got had has have how i if in into
    is it its just many me more most much my of on only or other our own per same
    should so some such than that the their them then there these they this to too
    up us very was we were what when where which who why will with would you your
"""
STOPWORDS = frozenset(_STOPWORD_TEXT.split())

# Query-side expansions, keyed by tokenised query word. Values are tokens that
# occur in the knowledge base.
QUERY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "vacation": ("annual", "leave"),
    "holiday": ("annual", "leave"),
    "pto": ("annual", "leave"),
    "off": ("leave",),
    "abroad": ("international",),
    "overseas": ("international",),
    "foreign": ("international",),
    "trip": ("travel",),
    "diem": ("daily", "allowance"),
    "mileage": ("kilometre", "car"),
    "reimbursement": ("reimbursed", "expense", "claim"),
    "refund": ("reimbursed", "expense", "claim"),
    "wfh": ("remote", "home"),
    "telework": ("remote",),
    "illness": ("sick", "ill"),
    "mfa": ("multi", "factor", "authentication"),
    "2fa": ("multi", "factor", "authentication"),
    "computer": ("laptop", "device"),
    "education": ("training", "course"),
}

_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n")
_TITLE_RE = re.compile(r"^(?P<title>[A-Z][A-Za-z &/-]{1,40}):\s")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Chunk:
    """One paragraph of the knowledge base."""

    id: str
    title: str
    text: str


class Snippet(TypedDict):
    """A retrieved chunk with its relevance score, as passed between agents."""

    id: str
    title: str
    text: str
    score: float


def split_into_chunks(text: str) -> list[Chunk]:
    """Split raw knowledge-base text into one chunk per paragraph.

    Paragraphs are separated by blank lines. Line breaks inside a paragraph are
    collapsed to single spaces. A leading ``Title:`` prefix becomes the chunk title.
    Chunk ids are stable and sequential: ``kb-01``, ``kb-02``, ...
    """
    paragraphs = (" ".join(p.split()) for p in _PARAGRAPH_BREAK_RE.split(text))
    chunks: list[Chunk] = []
    for paragraph in filter(None, paragraphs):
        match = _TITLE_RE.match(paragraph)
        title = match.group("title") if match else paragraph[:40]
        chunks.append(Chunk(id=f"kb-{len(chunks) + 1:02d}", title=title, text=paragraph))
    return chunks


def load_chunks(path: Path = DEFAULT_KB_PATH) -> list[Chunk]:
    """Read a knowledge-base file and split it into paragraph chunks."""
    if not path.is_file():
        raise FileNotFoundError(f"Knowledge base file not found: {path}")
    return split_into_chunks(path.read_text(encoding="utf-8"))


def _stem(token: str) -> str:
    """Apply a very light plural stemmer so 'days' matches 'day'."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lower-case, strip punctuation, drop stopwords and stem plurals."""
    tokens = _NON_ALNUM_RE.sub(" ", text.lower()).split()
    return [_stem(t) for t in tokens if len(t) > 1 and t not in STOPWORDS]


def expand_query(
    tokens: Sequence[str],
    synonyms: Mapping[str, Sequence[str]] = QUERY_SYNONYMS,
    weight: float = SYNONYM_WEIGHT,
) -> dict[str, float]:
    """Map each query term to its weight: 1.0 for the user's own words, ``weight``
    for synonym expansions. Repeated words count once."""
    weights = dict.fromkeys(tokens, 1.0)
    for token in tokens:
        for synonym in synonyms.get(token, ()):
            weights.setdefault(synonym, weight)
    return weights


class BM25Retriever:
    """Rank knowledge-base chunks against a query with Okapi BM25."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        k1: float = 1.5,
        b: float = 0.75,
        title_boost: int = DEFAULT_TITLE_BOOST,
        synonyms: Mapping[str, Sequence[str]] = QUERY_SYNONYMS,
    ) -> None:
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.synonyms = synonyms
        self._term_freqs = [self._index_chunk(c, title_boost) for c in self.chunks]
        self._doc_lens = [sum(tf.values()) for tf in self._term_freqs]
        self._avg_doc_len = sum(self._doc_lens) / len(self._doc_lens) if self.chunks else 0.0
        doc_freqs = Counter(term for tf in self._term_freqs for term in tf)
        n_docs = len(self.chunks)
        self._idf = {
            term: math.log(1 + (n_docs - df + 0.5) / (df + 0.5)) for term, df in doc_freqs.items()
        }

    @classmethod
    def from_file(cls, path: Path = DEFAULT_KB_PATH) -> BM25Retriever:
        """Build a retriever from a knowledge-base text file."""
        return cls(load_chunks(path))

    @staticmethod
    def _index_chunk(chunk: Chunk, title_boost: int) -> Counter[str]:
        """Term frequencies for a chunk, counting title words ``title_boost`` extra times."""
        term_freqs = Counter(tokenize(chunk.text))
        for term in tokenize(chunk.title):
            term_freqs[term] += title_boost
        return term_freqs

    def scores(self, query: str) -> list[float]:
        """Return the BM25 score of every chunk for ``query``, in chunk order."""
        terms = expand_query(tokenize(query), self.synonyms)
        return [
            sum(weight * self._term_score(term, i) for term, weight in terms.items())
            for i in range(len(self.chunks))
        ]

    def _term_score(self, term: str, doc_index: int) -> float:
        tf = self._term_freqs[doc_index].get(term, 0)
        if not tf:
            return 0.0
        length_norm = 1 - self.b + self.b * self._doc_lens[doc_index] / self._avg_doc_len
        return self._idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)

    def search(
        self, query: str, k: int = DEFAULT_TOP_K, min_score: float = DEFAULT_MIN_SCORE
    ) -> list[Snippet]:
        """Return up to ``k`` chunks scoring at least ``min_score``, best first.

        Returns an empty list (never raises) when the query has no usable terms or
        nothing in the knowledge base scores above the threshold.
        """
        # Round before filtering so the threshold applies to the score callers see.
        scores = self.scores(query)
        scored = [
            (chunk, round(score, 3)) for chunk, score in zip(self.chunks, scores, strict=True)
        ]
        ranked = sorted(scored, key=lambda pair: pair[1], reverse=True)
        return [
            Snippet(id=chunk.id, title=chunk.title, text=chunk.text, score=score)
            for chunk, score in ranked[:k]
            if score >= min_score and score > 0
        ]


def format_snippets(snippets: Sequence[Snippet]) -> str:
    """Render snippets as plain text blocks headed by their chunk id."""
    return "\n\n".join(f"[{s['id']}] (score {s['score']:.2f})\n{s['text']}" for s in snippets)


def make_search_tool(
    retriever: BM25Retriever,
    k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
) -> BaseTool:
    """Wrap a retriever as a LangChain tool for the Data Retriever agent.

    The tool returns text for the model and, as its artifact, the structured
    ``Snippet`` list so the workflow can pass the exact source text downstream.
    """

    @tool(response_format="content_and_artifact", parse_docstring=True)
    def search_knowledge_base(query: str) -> tuple[str, list[Snippet]]:
        """Search the company policy knowledge base and return the most relevant paragraphs.

        Use this for every question about company policy: travel, expenses, leave,
        sick leave, remote work, IT security, equipment and training. Matching is by
        keyword (BM25), so pass short keyword-rich queries such as
        "annual leave days carry over" rather than full sentences. Call it again
        with different keywords or synonyms if results are empty or incomplete.

        Args:
            query: Keywords describing the information needed.
        """
        hits = retriever.search(query, k=k, min_score=min_score)
        if not hits:
            return NO_RESULTS_MESSAGE, []
        return format_snippets(hits), hits

    return search_knowledge_base
