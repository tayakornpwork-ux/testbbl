"""Measure retrieval quality on a labelled query set, and optionally the full pipeline.

Usage:
    python evaluate.py          # BM25 retrieval only: Hit@1, Recall@3, MRR (no API key)
    python evaluate.py --e2e    # the full two-agent pipeline (needs .env)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from langgraph.graph.state import CompiledStateGraph
from rich.console import Console
from rich.table import Table

from config import ConfigError
from grounding import GroundingReport, check_grounding
from main import LLM_ERRORS, create_app, describe_llm_error, use_utf8_when_redirected
from retriever_tool import DEFAULT_KB_PATH, DEFAULT_TOP_K, BM25Retriever

EVALUATION_DIR = Path(__file__).parent / "evaluation"
DEFAULT_CASES_PATH = EVALUATION_DIR / "cases.json"
# Written after the retriever was tuned on cases.json and scored once, untuned.
HOLDOUT_CASES_PATH = EVALUATION_DIR / "holdout.json"


@dataclass(frozen=True)
class EvalCase:
    """A question and the ids of the chunks that answer it (empty: not in the KB)."""

    query: str
    relevant: tuple[str, ...]

    @property
    def answerable(self) -> bool:
        return bool(self.relevant)


def load_cases(path: Path = DEFAULT_CASES_PATH) -> list[EvalCase]:
    """Load evaluation cases from a JSON list of ``{query, relevant}`` objects."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [EvalCase(query=c["query"], relevant=tuple(c["relevant"])) for c in raw]


@dataclass(frozen=True)
class RetrievalScore:
    """How well the retriever did on one case."""

    case: EvalCase
    retrieved: tuple[str, ...]

    @property
    def hit_at_1(self) -> bool:
        return bool(self.retrieved) and self.retrieved[0] in self.case.relevant

    @property
    def recall(self) -> float:
        found = set(self.retrieved) & set(self.case.relevant)
        return len(found) / len(self.case.relevant)

    @property
    def reciprocal_rank(self) -> float:
        for rank, chunk_id in enumerate(self.retrieved, start=1):
            if chunk_id in self.case.relevant:
                return 1 / rank
        return 0.0


@dataclass(frozen=True)
class RetrievalReport:
    """Aggregate retrieval metrics over a case set."""

    answerable: list[RetrievalScore]
    unanswerable: list[RetrievalScore]

    @property
    def hit_at_1(self) -> float:
        return mean(s.hit_at_1 for s in self.answerable)

    @property
    def recall_at_k(self) -> float:
        return mean(s.recall for s in self.answerable)

    @property
    def mrr(self) -> float:
        return mean(s.reciprocal_rank for s in self.answerable)

    @property
    def abstain_rate(self) -> float:
        """Share of out-of-KB questions where BM25 alone returned nothing."""
        return mean(not s.retrieved for s in self.unanswerable) if self.unanswerable else 0.0


def score_retrieval(
    retriever: BM25Retriever, cases: Sequence[EvalCase], k: int = DEFAULT_TOP_K
) -> RetrievalReport:
    """Run the retriever on every case and score its top-``k`` results."""
    scores = [
        RetrievalScore(case, tuple(s["id"] for s in retriever.search(case.query, k=k)))
        for case in cases
    ]
    return RetrievalReport(
        answerable=[s for s in scores if s.case.answerable],
        unanswerable=[s for s in scores if not s.case.answerable],
    )


@dataclass(frozen=True)
class EndToEndResult:
    """The full pipeline's output on one case, and whether it passed."""

    case: EvalCase
    snippet_ids: tuple[str, ...]
    grounding: GroundingReport

    @property
    def passed(self) -> bool:
        if not self.case.answerable:
            return self.grounding.abstained
        cites_relevant = bool(set(self.grounding.cited_ids) & set(self.case.relevant))
        return not self.grounding.abstained and self.grounding.ok and cites_relevant


def run_end_to_end(app: CompiledStateGraph, cases: Sequence[EvalCase]) -> list[EndToEndResult]:
    """Run every case through the two-agent workflow and check each answer."""
    results = []
    for case in cases:
        state = app.invoke({"query": case.query})
        results.append(
            EndToEndResult(
                case=case,
                snippet_ids=tuple(s["id"] for s in state["snippets"]),
                grounding=check_grounding(state["answer"], state["snippets"], case.query),
            )
        )
    return results


def _mark(ok: bool) -> str:
    return "[green]yes[/green]" if ok else "[red]no[/red]"


def print_retrieval_report(console: Console, report: RetrievalReport, name: str) -> None:
    """Print per-case retrieval results and the summary metrics."""
    table = Table(title=f"Retrieval evaluation, BM25 only: {name}", title_justify="left")
    table.add_column("Query")
    table.add_column("Expected", style="cyan")
    table.add_column("Retrieved (top 3)")
    table.add_column("Hit@1", justify="center")
    for score in report.answerable + report.unanswerable:
        expected = ", ".join(score.case.relevant) or "(none)"
        hit = _mark(score.hit_at_1) if score.case.answerable else "-"
        table.add_row(score.case.query, expected, ", ".join(score.retrieved) or "-", hit)
    console.print(table)
    console.print(
        f"[bold]Answerable questions: {len(report.answerable)}[/bold]  "
        f"Hit@1 {report.hit_at_1:.0%}  Recall@3 {report.recall_at_k:.0%}  MRR {report.mrr:.2f}"
    )
    console.print(
        f"[bold]Out-of-KB questions: {len(report.unanswerable)}[/bold]  "
        f"BM25 returned nothing for {report.abstain_rate:.0%}; "
        "the Data Retriever agent must filter the rest."
    )


def print_end_to_end_report(console: Console, results: Sequence[EndToEndResult]) -> None:
    """Print per-case pipeline results and the pass rate."""
    table = Table(title="End-to-end evaluation", title_justify="left")
    table.add_column("Query")
    table.add_column("Expected", style="cyan")
    table.add_column("Snippets")
    table.add_column("Cited")
    table.add_column("Pass", justify="center")
    table.add_column("Notes")
    for result in results:
        g = result.grounding
        table.add_row(
            result.case.query,
            ", ".join(result.case.relevant) or "(not in KB)",
            ", ".join(result.snippet_ids) or "-",
            "abstained" if g.abstained else ", ".join(g.cited_ids) or "-",
            _mark(result.passed),
            "; ".join(g.warnings()),
        )
    console.print(table)
    passed = sum(r.passed for r in results)
    console.print(f"[bold]Passed {passed}/{len(results)} ({passed / len(results):.0%})[/bold]")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answer quality.")
    parser.add_argument(
        "--cases", type=Path, nargs="+", default=[DEFAULT_CASES_PATH, HOLDOUT_CASES_PATH]
    )
    parser.add_argument(
        "--e2e", action="store_true", help="Also run the full LLM pipeline (needs .env)."
    )
    args = parser.parse_args(argv)
    use_utf8_when_redirected()
    console = Console()
    retriever = BM25Retriever.from_file(DEFAULT_KB_PATH)

    cases: list[EvalCase] = []
    for path in args.cases:
        file_cases = load_cases(path)
        cases.extend(file_cases)
        print_retrieval_report(console, score_retrieval(retriever, file_cases), path.name)
        console.print()
    if not args.e2e:
        return 0

    load_dotenv()
    try:
        app = create_app()
        with console.status(f"Running {len(cases)} questions through both agents ..."):
            results = run_end_to_end(app, cases)
    except ConfigError as exc:
        console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        return 2
    except LLM_ERRORS as exc:
        console.print(f"[bold red]Error:[/bold red] {describe_llm_error(exc)}")
        return 1
    console.print()
    print_end_to_end_report(console, results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
