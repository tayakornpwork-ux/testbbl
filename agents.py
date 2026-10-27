"""The two agents of the pipeline, built with LangChain's ``create_agent``.

* Data Retriever: a tool-calling agent that must search the knowledge base, may
  re-search with reworded queries, and returns only the ids of relevant chunks.
  The chunk text itself is taken verbatim from the tool results, so the snippets
  handed downstream are the raw source text, never a model paraphrase.
* Report Generator: a tool-less agent that writes the final answer from the
  snippets alone.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from prompts import (
    DATA_RETRIEVER_PROMPT,
    NOT_FOUND_MESSAGE,
    REPORT_GENERATOR_INPUT_TEMPLATE,
    REPORT_GENERATOR_PROMPT,
)
from retriever_tool import Snippet

# Each search costs two graph steps (model call + tool call); this allows the
# three searches the prompt permits plus the final structured response.
RETRIEVER_RECURSION_LIMIT = 12
DATA_RETRIEVER_NAME = "data_retriever"
REPORT_GENERATOR_NAME = "report_generator"


class RetrievalResult(BaseModel):
    """Final output of the Data Retriever: which retrieved chunks are relevant."""

    relevant_chunk_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Ids of the retrieved chunks that contain facts answering the question, "
            "most relevant first. Empty if no chunk is relevant."
        ),
    )


def _force_first_search(tool_name: str) -> Callable[..., ModelResponse]:
    """Middleware that forces the agent's first model call to use ``tool_name``.

    This turns "the retriever MUST call the search tool" from a prompt request into
    a guarantee. ``ToolStrategy`` always binds ``tool_choice="any"``, so on the first
    call the structured-output tool is withheld and the search tool is named
    explicitly. Later calls keep ``"any"``: the agent must either search again or
    submit its ``RetrievalResult``; it can never reply with free text.
    """

    @wrap_model_call
    def force_first_search(
        request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        if not any(isinstance(m, ToolMessage) for m in request.messages):
            request = request.override(tool_choice=tool_name, response_format=None)
        return handler(request)

    return force_first_search


def build_data_retriever(model: BaseChatModel, search_tool: BaseTool) -> CompiledStateGraph:
    """Create the Data Retriever agent around the knowledge-base search tool."""
    return create_agent(
        model,
        tools=[search_tool],
        system_prompt=DATA_RETRIEVER_PROMPT,
        middleware=[_force_first_search(search_tool.name)],
        response_format=ToolStrategy(RetrievalResult),
        name=DATA_RETRIEVER_NAME,
    )


def build_report_generator(model: BaseChatModel) -> CompiledStateGraph:
    """Create the Report Generator agent. It has no tools by design."""
    return create_agent(
        model,
        tools=[],
        system_prompt=REPORT_GENERATOR_PROMPT,
        name=REPORT_GENERATOR_NAME,
    )


def collect_snippets(messages: Sequence[object]) -> dict[str, Snippet]:
    """Gather every snippet returned by search tool calls, keeping the best score."""
    found: dict[str, Snippet] = {}
    for message in messages:
        if not isinstance(message, ToolMessage) or not message.artifact:
            continue
        for snippet in message.artifact:
            best = found.get(snippet["id"])
            if best is None or snippet["score"] > best["score"]:
                found[snippet["id"]] = snippet
    return found


def select_snippets(found: dict[str, Snippet], result: RetrievalResult | None) -> list[Snippet]:
    """Keep the snippets the agent judged relevant, in its order.

    Ids the agent invented (not actually retrieved) are ignored. If the agent gave
    no structured result, fall back to everything retrieved, best score first.
    """
    if result is None:
        return sorted(found.values(), key=lambda s: s["score"], reverse=True)
    ordered_ids = dict.fromkeys(result.relevant_chunk_ids)
    return [found[chunk_id] for chunk_id in ordered_ids if chunk_id in found]


def run_data_retriever(agent: CompiledStateGraph, query: str) -> list[Snippet]:
    """Run the Data Retriever and return the raw relevant chunks for ``query``."""
    state = agent.invoke(
        {"messages": [HumanMessage(query)]},
        config={"recursion_limit": RETRIEVER_RECURSION_LIMIT},
    )
    found = collect_snippets(state["messages"])
    return select_snippets(found, state.get("structured_response"))


def format_generator_input(query: str, snippets: Sequence[Snippet]) -> str:
    """Build the Report Generator's user message from the query and snippets."""
    blocks = "\n\n".join(f"[{s['id']}] {s['text']}" for s in snippets)
    return REPORT_GENERATOR_INPUT_TEMPLATE.format(query=query, snippets=blocks)


def run_report_generator(agent: CompiledStateGraph, query: str, snippets: Sequence[Snippet]) -> str:
    """Run the Report Generator and return the final Markdown answer.

    With no snippets there is nothing to ground an answer in, so the not-found
    message is returned without calling the model; this makes hallucination on
    an empty retrieval impossible rather than merely unlikely.
    """
    if not snippets:
        return NOT_FOUND_MESSAGE
    state = agent.invoke({"messages": [HumanMessage(format_generator_input(query, snippets))]})
    reply = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
    return reply.text.strip() or NOT_FOUND_MESSAGE
