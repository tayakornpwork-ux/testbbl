"""Tests for reading LLM settings from the environment."""

from __future__ import annotations

import pytest
from langchain_anthropic import ChatAnthropic

from config import DEFAULT_AZURE_API_VERSION, ConfigError, build_chat_model, load_settings

AZURE_ENV = {
    "LLM_PROVIDER": "azure",
    "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
    "AZURE_OPENAI_API_KEY": "test-key",
    "AZURE_OPENAI_DEPLOYMENT": "gpt-5-mini",
}


def test_anthropic_is_the_default_provider() -> None:
    settings = load_settings({"ANTHROPIC_API_KEY": "test-key"})
    assert (settings.provider, settings.model) == ("anthropic", "claude-haiku-5-5")
    assert load_settings({"LLM_PROVIDER": " ", "ANTHROPIC_API_KEY": "k"}).provider == "anthropic"


def test_azure_provider_settings() -> None:
    settings = load_settings(AZURE_ENV)
    assert settings.provider == "azure"
    assert settings.model == "gpt-5-mini"
    assert settings.azure_api_version == DEFAULT_AZURE_API_VERSION


def test_missing_variables_are_named_in_the_error() -> None:
    env = {**AZURE_ENV, "AZURE_OPENAI_API_KEY": "  "}
    with pytest.raises(ConfigError, match="AZURE_OPENAI_API_KEY"):
        load_settings(env)


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ConfigError, match="Unknown LLM_PROVIDER"):
        load_settings({"LLM_PROVIDER": "nope"})


def test_openai_provider_swap() -> None:
    settings = load_settings({"LLM_PROVIDER": "OpenAI", "OPENAI_API_KEY": "test-key"})
    assert (settings.provider, settings.model) == ("openai", "gpt-5-mini")


@pytest.mark.parametrize(
    "env", [AZURE_ENV, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}]
)
def test_models_use_the_responses_api(env: dict[str, str]) -> None:
    model = build_chat_model(load_settings(env), "low")
    assert model.use_responses_api is True
    assert model.reasoning == {"effort": "low"}


def test_each_agent_has_its_own_reasoning_effort() -> None:
    defaults = load_settings(AZURE_ENV)
    assert (defaults.retriever_reasoning_effort, defaults.generator_reasoning_effort) == (
        "minimal",
        "low",
    )
    custom = load_settings({**AZURE_ENV, "RETRIEVER_REASONING_EFFORT": "Low"})
    assert custom.retriever_reasoning_effort == "low"


def test_invalid_reasoning_effort_is_rejected() -> None:
    with pytest.raises(ConfigError, match="GENERATOR_REASONING_EFFORT"):
        load_settings({**AZURE_ENV, "GENERATOR_REASONING_EFFORT": "turbo"})


def test_anthropic_provider_builds_claude_without_reasoning_settings() -> None:
    settings = load_settings({"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test-key"})
    assert (settings.provider, settings.model) == ("anthropic", "claude-haiku-5-5")
    model = build_chat_model(settings, "minimal", disable_streaming=True)
    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-haiku-5-5"
    assert model.thinking is None
    assert model.disable_streaming is True


def test_anthropic_requires_its_key() -> None:
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        load_settings({"LLM_PROVIDER": "anthropic"})


def test_streaming_can_be_disabled_per_model() -> None:
    settings = load_settings(AZURE_ENV)
    assert build_chat_model(settings, "minimal", disable_streaming=True).disable_streaming is True
    assert build_chat_model(settings, "low").disable_streaming is False
