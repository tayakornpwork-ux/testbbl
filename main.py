"""Agentic RAG pipeline: the LangGraph workflow and its command-line interface.

Usage:
    python main.py "How many days of annual leave do I get?"
    python main.py                 # run the demo queries
    python main.py --search-only   # demo the BM25 search tool alone, no LLM needed
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal, TypedDict

import anthropic
import openai
from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from agents import (
    REPORT_GENERATOR_NAME,
    build_data_retriever,
    build_report_generator,
    run_data_retriever,
    run_report_generator,
)
from config import ConfigError, build_chat_model, load_settings
from grounding import GroundingReport, check_grounding
from retriever_tool import DEFAULT_KB_PATH, BM25Retriever, Snippet, make_search_tool

# ("snippets", list[Snippet]) | ("token", str) | ("answer", str)
StreamEvent = tuple[Literal["snippets", "token", "answer"], Any]
# Failures reported to the user with describe_llm_error instead of a traceback.
LLM_ERRORS = (openai.OpenAIError, anthropic.AnthropicError, GraphRecursionError)

DEMO_QUERIES = (
    "What is the policy on international travel?",
    "How many days of annual leave do I get?",
    "How do I claim travel expenses?",
    "What is the company's policy on pets in the office?",
)


class GraphState(TypedDict):
    """State shared by the workflow nodes."""

    query: str
    snippets: list[Snippet]
    answer: str


def build_workflow(
    retriever_agent: CompiledStateGraph, generator_agent: CompiledStateGraph
) -> CompiledStateGraph:
    """Wire the two agents into a strict retriever -> generator -> END graph."""

    def retriever_node(state: GraphState) -> dict[str, list[Snippet]]:
        return {"snippets": run_data_retriever(retriever_agent, state["query"])}

    def generator_node(state: GraphState) -> dict[str, str]:
        answer = run_report_generator(generator_agent, state["query"], state["snippets"])
        return {"answer": answer}

    graph = StateGraph(GraphState)
    graph.add_node("retriever", retriever_node)
    graph.add_node("generator", generator_node)
    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", END)
    return graph.compile()


def create_app(kb_path: Path = DEFAULT_KB_PATH) -> CompiledStateGraph:
    """Build the full pipeline from environment settings and the knowledge base."""
    settings = load_settings()
    search_tool = make_search_tool(BM25Retriever.from_file(kb_path))
    retriever_model = build_chat_model(
        settings, settings.retriever_reasoning_effort, disable_streaming=True
    )
    generator_model = build_chat_model(settings, settings.generator_reasoning_effort)
    return build_workflow(
        build_data_retriever(retriever_model, search_tool),
        build_report_generator(generator_model),
    )


def describe_llm_error(exc: Exception) -> str:
    """Turn an LLM or workflow exception into a short, actionable message."""
    if isinstance(exc, (openai.AuthenticationError, anthropic.AuthenticationError)):
        return "Authentication failed (HTTP 401). Check the API key in your .env file."
    if isinstance(exc, (openai.NotFoundError, anthropic.NotFoundError)):
        return (
            "Model or deployment not found (HTTP 404). Check the model name in .env. "
            "For Azure, check AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT and "
            "AZURE_OPENAI_API_VERSION (the Responses API needs 2025-03-01-preview or later)."
        )
    if isinstance(exc, (openai.RateLimitError, anthropic.RateLimitError)):
        return "Rate limit or quota exceeded (HTTP 429). Wait a moment and retry."
    if isinstance(exc, (openai.APIConnectionError, anthropic.APIConnectionError)):
        return "Could not reach the LLM endpoint. Check the endpoint URL and your network."
    if isinstance(exc, (openai.APIStatusError, anthropic.APIStatusError)):
        return f"The LLM request failed (HTTP {exc.status_code}): {exc.message}"
    if isinstance(exc, GraphRecursionError):
        return "The Data Retriever exceeded its step limit without finishing."
    return f"Unexpected LLM error: {exc}"


def render_snippets(console: Console, snippets: Sequence[Snippet]) -> None:
    """Print the retrieved chunk ids, topics and scores as a table."""
    if not snippets:
        console.print("[yellow]No relevant chunks retrieved.[/yellow]")
        return
    table = Table(title="Retrieved chunks", title_justify="left", header_style="bold")
    table.add_column("Chunk id", style="cyan", no_wrap=True)
    table.add_column("Topic")
    table.add_column("BM25 score", justify="right")
    for snippet in snippets:
        table.add_row(snippet["id"], snippet["title"], f"{snippet['score']:.2f}")
    console.print(table)


def stream_workflow(app: CompiledStateGraph, query: str) -> Iterator[StreamEvent]:
    """Run the workflow, yielding the snippets, then answer tokens, then the answer.

    ``subgraphs=True`` surfaces token events from the agents running inside the
    nodes; only the Report Generator's text tokens are passed on.
    """
    for namespace, mode, payload in app.stream(
        {"query": query}, stream_mode=["updates", "messages"], subgraphs=True
    ):
        if mode == "updates" and not namespace:
            for node, values in payload.items():
                if node == "retriever":
                    yield "snippets", values["snippets"]
                elif node == "generator":
                    yield "answer", values["answer"]
        elif mode == "messages":
            chunk, metadata = payload
            is_generator = metadata.get("lc_agent_name") == REPORT_GENERATOR_NAME
            if is_generator and isinstance(chunk, AIMessageChunk) and chunk.text:
                yield "token", chunk.text


def _answer_panel(markdown: str) -> Panel:
    return Panel(Markdown(markdown), title="Final answer", border_style="green")


def render_grounding(console: Console, report: GroundingReport) -> None:
    """Print the outcome of the post-hoc grounding check."""
    if report.abstained:
        console.print("[dim]Grounding check: not applicable (no supporting chunks).[/dim]")
    elif report.ok:
        console.print(
            f"[green]Grounding check PASS:[/green] cites {', '.join(report.cited_ids)}; "
            "every number in the answer appears in the sources."
        )
    else:
        for warning in report.warnings():
            console.print(f"[yellow]Grounding check WARN:[/yellow] {warning}")


def run_query(console: Console, app: CompiledStateGraph, index: int, query: str) -> GraphState:
    """Run one query, printing chunks as soon as they are found and streaming the answer."""
    console.rule(f"[bold]Query {index}")
    console.print(Panel(query, title="Question", border_style="cyan"))
    state = GraphState(query=query, snippets=[], answer="")
    partial = ""
    live: Live | None = None
    status = console.status("Data Retriever: searching the knowledge base ...")
    status.start()
    try:
        for kind, value in stream_workflow(app, query):
            if kind == "snippets":
                status.stop()
                state["snippets"] = value
                render_snippets(console, value)
                # Live redraws in place, which only works on a real terminal.
                if console.is_terminal:
                    live = Live(_answer_panel("_Report Generator is writing..._"), console=console)
                    live.start()
            elif kind == "token" and live is not None:
                partial += value
                live.update(_answer_panel(partial))
            elif kind == "answer":
                state["answer"] = value
                if live is not None:
                    live.update(_answer_panel(value))
                else:
                    console.print(_answer_panel(value))
    finally:
        status.stop()
        if live is not None:
            live.stop()
    render_grounding(console, check_grounding(state["answer"], state["snippets"], query))
    console.print()
    return state


def run_pipeline(console: Console, queries: Sequence[str]) -> int:
    """Run each query through the agent workflow. Returns a process exit code."""
    try:
        app = create_app()
    except (ConfigError, FileNotFoundError) as exc:
        console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        return 2

    for index, query in enumerate(queries, start=1):
        try:
            run_query(console, app, index, query)
        except LLM_ERRORS as exc:
            console.print(f"[bold red]Error on query {index}:[/bold red] {describe_llm_error(exc)}")
            return 1
    return 0


def run_search_only(console: Console, queries: Sequence[str]) -> int:
    """Run only the BM25 search tool for each query; no LLM or API key required."""
    try:
        retriever = BM25Retriever.from_file(DEFAULT_KB_PATH)
    except FileNotFoundError as exc:
        console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        return 2
    for index, query in enumerate(queries, start=1):
        console.rule(f"[bold]Search {index}")
        console.print(Panel(query, title="Query", border_style="cyan"))
        render_snippets(console, retriever.search(query))
        console.print()
    return 0


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Two-agent RAG over a local company-policy knowledge base."
    )
    parser.add_argument(
        "query", nargs="?", help="Question to answer. Omit it to run the demo queries."
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Run only the BM25 search tool (no LLM call), to inspect retrieval.",
    )
    return parser.parse_args(argv)


def use_utf8_when_redirected() -> None:
    """Write UTF-8 when output goes to a file or pipe.

    Windows otherwise uses the legacy code page there, which turns bullets into "?".
    """
    if not sys.stdout.isatty() and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    load_dotenv()
    use_utf8_when_redirected()
    console = Console()
    queries = [args.query] if args.query else list(DEMO_QUERIES)
    if args.search_only:
        return run_search_only(console, queries)
    return run_pipeline(console, queries)


if __name__ == "__main__":
    sys.exit(main())
