"""Tests that provider errors become short, actionable messages."""

from __future__ import annotations

import anthropic
import httpx
import openai
import pytest

from main import LLM_ERRORS, describe_llm_error


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", "https://example.test"))


@pytest.mark.parametrize("sdk", [openai, anthropic])
@pytest.mark.parametrize(
    ("error_name", "status", "expected"),
    [
        ("AuthenticationError", 401, "Authentication failed"),
        ("NotFoundError", 404, "not found"),
        ("RateLimitError", 429, "Rate limit"),
    ],
)
def test_status_errors_from_both_sdks_are_explained(
    sdk, error_name: str, status: int, expected: str
) -> None:
    exc = getattr(sdk, error_name)("boom", response=_response(status), body=None)
    assert isinstance(exc, LLM_ERRORS)
    assert expected in describe_llm_error(exc)
