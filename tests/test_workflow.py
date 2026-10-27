"""Offline tests of the agent workflow, using a scripted fake chat model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from pydantic import Field
from rich.console import Console

from agents import build_data_retriever, build_report_generator
from main import build_workflow, run_query, stream_workflow
from prompts import NOT_FOUND_MESSAGE
from retriever_tool import DEFAULT_KB_PATH, BM25Retriever, make_search_tool


class ScriptedChatModel(GenericFakeChatModel):
    """Fake model that replays scripted replies and records each tool binding."""

    bindings: list[tuple[Any, list[str]]] = Field(default_factory=list)

    def bind_tools(self, tools: Sequence[Any], *, tool_choice: Any = None, **_: Any):
        names = [getattr(t, "name", None) or t.get("name", "") for t in tools]
        self.bindings.append((tool_choice, names))
        return self


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _app(retriever_replies: list[AIMessage], generator_replies: list[AIMessage]):
    # As in create_app, only the generator streams.
    retriever_model = ScriptedChatModel(messages=iter(retriever_replies), disable_streaming=True)
    generator_model = ScriptedChatModel(messages=iter(generator_replies))
    search_tool = make_search_tool(BM25Retriever.from_file(DEFAULT_KB_PATH))
    app = build_workflow(
        build_data_retriever(retriever_model, search_tool),
        build_report_generator(generator_model),
    )
    return app, retriever_model, generator_model


def test_workflow_runs_retriever_then_generator() -> None:
    answer = "**You get 25 days.**\n\n- 25 days per year [kb-04]\n\nSources: kb-04"
    app, retriever_model, _ = _app(
        [
            _tool_call("search_knowledge_base", {"query": "annual leave days"}, "c1"),
            _tool_call("RetrievalResult", {"relevant_chunk_ids": ["kb-04"]}, "c2"),
        ],
        [AIMessage(content=answer)],
    )

    steps = [next(iter(update)) for update in app.stream({"query": "Annual leave?"})]
    assert steps == ["retriever", "generator"]

    first_choice, first_tools = retriever_model.bindings[0]
    assert first_choice == "search_knowledge_base"
    assert first_tools == ["search_knowledge_base"]
    later_choice, later_tools = retriever_model.bindings[1]
    assert later_choice == "any"
    assert "RetrievalResult" in later_tools


def test_workflow_returns_raw_chunks_and_answer() -> None:
    answer = "**You get 25 days.**\n\nSources: kb-04"
    app, _, _ = _app(
        [
            _tool_call("search_knowledge_base", {"query": "annual leave days"}, "c1"),
            _tool_call("RetrievalResult", {"relevant_chunk_ids": ["kb-04"]}, "c2"),
        ],
        [AIMessage(content=answer)],
    )
    state = app.invoke({"query": "How many days of annual leave do I get?"})
    kb_text = DEFAULT_KB_PATH.read_text(encoding="utf-8")

    assert [s["id"] for s in state["snippets"]] == ["kb-04"]
    assert state["snippets"][0]["text"] in " ".join(kb_text.split())
    assert state["answer"] == answer


def test_workflow_skips_generator_model_when_nothing_relevant() -> None:
    # The generator has no scripted replies, so calling its model would raise.
    app, _, _ = _app(
        [
            _tool_call("search_knowledge_base", {"query": "pets office"}, "c1"),
            _tool_call("RetrievalResult", {"relevant_chunk_ids": []}, "c2"),
        ],
        [],
    )
    state = app.invoke({"query": "What is the policy on pets in the office?"})

    assert state["snippets"] == []
    assert state["answer"] == NOT_FOUND_MESSAGE


def test_retriever_may_search_several_times() -> None:
    app, _, _ = _app(
        [
            _tool_call("search_knowledge_base", {"query": "travel"}, "c1"),
            _tool_call("search_knowledge_base", {"query": "expense claim receipt"}, "c2"),
            _tool_call("RetrievalResult", {"relevant_chunk_ids": ["kb-03", "kb-01"]}, "c3"),
        ],
        [AIMessage(content="**Claim in ClaimIt.**\n\nSources: kb-03, kb-01")],
    )
    state = app.invoke({"query": "How do I claim travel expenses?"})
    assert [s["id"] for s in state["snippets"]] == ["kb-03", "kb-01"]


def test_generator_empty_reply_falls_back_to_not_found() -> None:
    app, _, _ = _app(
        [
            _tool_call("search_knowledge_base", {"query": "annual leave"}, "c1"),
            _tool_call("RetrievalResult", {"relevant_chunk_ids": ["kb-04"]}, "c2"),
        ],
        [AIMessage(content="   ")],
    )
    assert app.invoke({"query": "Annual leave?"})["answer"] == NOT_FOUND_MESSAGE


def _leave_app():
    return _app(
        [
            _tool_call("search_knowledge_base", {"query": "annual leave"}, "c1"),
            _tool_call("RetrievalResult", {"relevant_chunk_ids": ["kb-04"]}, "c2"),
        ],
        [AIMessage(content=LEAVE_ANSWER)],
    )[0]


LEAVE_ANSWER = (
    "**You get 25 days of annual leave.**\n\n- 25 days per year [kb-04]\n\nSources: kb-04"
)


def test_stream_workflow_yields_snippets_then_generator_tokens_then_answer() -> None:
    events = list(stream_workflow(_leave_app(), "Annual leave?"))
    kinds = [kind for kind, _ in events]

    assert kinds[0] == "snippets"
    assert [s["id"] for s in events[0][1]] == ["kb-04"]
    assert kinds[-1] == "answer"
    assert set(kinds[1:-1]) == {"token"}
    assert "".join(value for kind, value in events if kind == "token") == LEAVE_ANSWER
    assert events[-1][1] == LEAVE_ANSWER


def test_run_query_prints_chunks_answer_and_grounding_check() -> None:
    console = Console(record=True, width=100)
    state = run_query(console, _leave_app(), 1, "Annual leave?")
    output = console.export_text()

    assert state["answer"] == LEAVE_ANSWER
    for expected in ("Annual leave?", "kb-04", "You get 25 days", "Grounding check PASS"):
        assert expected in output
