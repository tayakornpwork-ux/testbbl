"""Unit tests for chunking, tokenisation and BM25 ranking. No LLM is needed."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents import RetrievalResult, collect_snippets, select_snippets
from retriever_tool import (
    DEFAULT_KB_PATH,
    NO_RESULTS_MESSAGE,
    SYNONYM_WEIGHT,
    BM25Retriever,
    Chunk,
    expand_query,
    load_chunks,
    make_search_tool,
    split_into_chunks,
    tokenize,
)

SAMPLE_KB = """\
Annual Leave: Employees get 25 days of paid leave.
Requests go through the HR portal.

Sick Leave: Notify your manager before 10:00 on the first day.


IT Security: Passwords must be at least 14 characters long.
"""


@pytest.fixture
def retriever() -> BM25Retriever:
    return BM25Retriever(split_into_chunks(SAMPLE_KB))


@pytest.fixture(scope="module")
def kb_retriever() -> BM25Retriever:
    return BM25Retriever.from_file(DEFAULT_KB_PATH)


# --- chunking -------------------------------------------------------------------


def test_split_into_chunks_one_chunk_per_paragraph() -> None:
    chunks = split_into_chunks(SAMPLE_KB)
    assert [c.id for c in chunks] == ["kb-01", "kb-02", "kb-03"]
    assert [c.title for c in chunks] == ["Annual Leave", "Sick Leave", "IT Security"]


def test_split_into_chunks_joins_wrapped_lines() -> None:
    first = split_into_chunks(SAMPLE_KB)[0]
    assert first.text == (
        "Annual Leave: Employees get 25 days of paid leave. Requests go through the HR portal."
    )


def test_split_into_chunks_handles_empty_and_untitled_text() -> None:
    assert split_into_chunks("\n\n  \n") == []
    (chunk,) = split_into_chunks("no title here, just text")
    assert chunk.title == "no title here, just text"


def test_load_chunks_reads_bundled_knowledge_base() -> None:
    chunks = load_chunks(DEFAULT_KB_PATH)
    assert 6 <= len(chunks) <= 10
    assert len({c.id for c in chunks}) == len(chunks)


def test_load_chunks_missing_file_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Knowledge base file not found"):
        load_chunks(tmp_path / "missing.txt")


# --- tokenisation ---------------------------------------------------------------


def test_tokenize_normalises_case_punctuation_and_stopwords() -> None:
    assert tokenize("What is the Company's REMOTE-work policy?") == [
        "company",
        "remote",
        "work",
        "policy",
    ]


def test_tokenize_stems_simple_plurals() -> None:
    assert tokenize("days claims policies business") == ["day", "claim", "policy", "business"]


# --- ranking --------------------------------------------------------------------


def test_search_ranks_matching_chunk_first(retriever: BM25Retriever) -> None:
    hits = retriever.search("password length", min_score=0)
    assert hits[0]["id"] == "kb-03"
    assert hits[0]["score"] > 0


def test_search_is_case_and_punctuation_insensitive(retriever: BM25Retriever) -> None:
    assert retriever.search("PASSWORDS!!!") == retriever.search("passwords")


def test_search_respects_k_and_orders_by_score(kb_retriever: BM25Retriever) -> None:
    hits = kb_retriever.search("leave days manager approval", k=2, min_score=0)
    assert len(hits) == 2
    assert hits[0]["score"] >= hits[1]["score"]


def test_search_returns_empty_list_when_nothing_matches(retriever: BM25Retriever) -> None:
    assert retriever.search("pets in the office") == []
    assert retriever.search("") == []
    assert retriever.search("what is the") == []


def test_search_applies_min_score(kb_retriever: BM25Retriever) -> None:
    all_hits = kb_retriever.search("travel", k=10, min_score=0)
    strict = kb_retriever.search("travel", k=10, min_score=all_hits[0]["score"])
    assert [h["id"] for h in strict] == [all_hits[0]["id"]]


def test_idf_is_never_negative_for_common_terms() -> None:
    chunks = [Chunk(id=f"kb-{i}", title="t", text=f"leave policy item{i}") for i in range(5)]
    hits = BM25Retriever(chunks).search("leave", k=5, min_score=0)
    assert len(hits) == 5
    assert all(h["score"] > 0 for h in hits)


def test_empty_corpus_returns_no_results() -> None:
    assert BM25Retriever([]).search("anything") == []


@pytest.mark.parametrize(
    ("query", "expected_title"),
    [
        ("international travel approval", "International Travel"),
        ("annual leave days", "Annual Leave"),
        ("expense claim receipt", "Expense Reimbursement"),
        ("doctor note sick", "Sick Leave"),
        ("work from home remote", "Remote Work"),
        ("password VPN phishing", "IT Security"),
    ],
)
def test_knowledge_base_top_hit(
    kb_retriever: BM25Retriever, query: str, expected_title: str
) -> None:
    assert kb_retriever.search(query)[0]["title"] == expected_title


# --- LangChain tool -------------------------------------------------------------


def test_search_tool_returns_text_and_snippet_artifact(retriever: BM25Retriever) -> None:
    search = make_search_tool(retriever)
    message = search.invoke(
        {"type": "tool_call", "id": "1", "name": search.name, "args": {"query": "password"}}
    )
    assert "[kb-03]" in message.content
    assert [s["id"] for s in message.artifact] == ["kb-03"]


def test_search_tool_reports_no_results_without_error(retriever: BM25Retriever) -> None:
    search = make_search_tool(retriever)
    message = search.invoke(
        {"type": "tool_call", "id": "1", "name": search.name, "args": {"query": "pets"}}
    )
    assert message.content == NO_RESULTS_MESSAGE
    assert message.artifact == []


def test_search_tool_schema_exposes_only_query(retriever: BM25Retriever) -> None:
    search = make_search_tool(retriever)
    assert list(search.args) == ["query"]
    assert "knowledge base" in search.description


# --- snippet selection (agent output handling) ----------------------------------


def _tool_messages(retriever: BM25Retriever, *queries: str) -> list[object]:
    search = make_search_tool(retriever, min_score=0)
    return [
        search.invoke(
            {"type": "tool_call", "id": str(i), "name": search.name, "args": {"query": q}}
        )
        for i, q in enumerate(queries)
    ]


def test_select_snippets_keeps_agent_order_and_drops_unknown_ids(
    retriever: BM25Retriever,
) -> None:
    found = collect_snippets(_tool_messages(retriever, "password", "sick manager"))
    result = RetrievalResult(relevant_chunk_ids=["kb-02", "kb-99", "kb-03", "kb-02"])
    assert [s["id"] for s in select_snippets(found, result)] == ["kb-02", "kb-03"]


def test_select_snippets_empty_selection_means_no_snippets(retriever: BM25Retriever) -> None:
    found = collect_snippets(_tool_messages(retriever, "password"))
    assert select_snippets(found, RetrievalResult(relevant_chunk_ids=[])) == []


def test_select_snippets_falls_back_to_all_results_by_score(retriever: BM25Retriever) -> None:
    found = collect_snippets(_tool_messages(retriever, "leave"))
    selected = select_snippets(found, None)
    assert {s["id"] for s in selected} == set(found)
    assert selected == sorted(selected, key=lambda s: s["score"], reverse=True)


# --- query expansion and title boost --------------------------------------------


def test_tokenize_drops_function_words_like_each_and_per() -> None:
    assert tokenize("each employee per year") == ["employee", "year"]


def test_expand_query_adds_synonyms_at_reduced_weight() -> None:
    weights = expand_query(["vacation", "day", "day"])
    assert weights == {
        "vacation": 1.0,
        "day": 1.0,
        "annual": SYNONYM_WEIGHT,
        "leave": SYNONYM_WEIGHT,
    }


def test_expand_query_never_downweights_the_users_own_words() -> None:
    assert expand_query(["leave", "vacation"])["leave"] == 1.0


def test_synonyms_bridge_vocabulary_gaps(kb_retriever: BM25Retriever) -> None:
    assert kb_retriever.search("vacation days")[0]["title"] == "Annual Leave"
    assert kb_retriever.search("WFH")[0]["title"] == "Remote Work"
    plain = BM25Retriever(load_chunks(DEFAULT_KB_PATH), synonyms={})
    assert plain.search("WFH") == []


def test_title_boost_favours_the_chunk_about_the_topic() -> None:
    chunks = [
        Chunk("kb-01", "Sick Leave", "Sick Leave: notify your manager."),
        Chunk("kb-02", "Remote Work", "Remote Work: sick colleagues sick days sick note."),
    ]
    unboosted = BM25Retriever(chunks, title_boost=0).search("sick", min_score=0)
    boosted = BM25Retriever(chunks, title_boost=3).search("sick", min_score=0)
    assert unboosted[0]["id"] == "kb-02"
    assert boosted[0]["id"] == "kb-01"
