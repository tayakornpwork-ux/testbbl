"""LLM configuration: the single place to choose and build the chat model.

Set ``LLM_PROVIDER`` to ``anthropic`` (default), ``azure`` or ``openai``. The
OpenAI providers are called through the Responses API.

Each OpenAI agent gets its own reasoning effort. The Data Retriever only picks
keywords and filters chunks, so it defaults to ``minimal`` for speed; the Report
Generator writes the answer and defaults to ``low``. Claude runs without extended
thinking: the Data Retriever forces its first tool call, and the Anthropic API
does not allow a forced tool choice while thinking is enabled.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, ChatOpenAI

DEFAULT_PROVIDER = "anthropic"
DEFAULT_AZURE_API_VERSION = "2025-04-01-preview"
DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-5-5"
ANTHROPIC_MAX_TOKENS = 2048
DEFAULT_RETRIEVER_REASONING_EFFORT = "minimal"
DEFAULT_GENERATOR_REASONING_EFFORT = "low"
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high")
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 2

_REQUIRED_VARS = {
    "azure": (
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
    ),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
}


class ConfigError(RuntimeError):
    """Raised when required LLM settings are missing or invalid."""


@dataclass(frozen=True)
class LLMSettings:
    """Resolved LLM settings, read from environment variables."""

    provider: str
    model: str
    api_key: str
    retriever_reasoning_effort: str
    generator_reasoning_effort: str
    azure_endpoint: str | None = None
    azure_api_version: str | None = None


def _reasoning_effort(env: dict[str, str], name: str, default: str) -> str:
    """Read and validate one reasoning-effort variable."""
    effort = env.get(name, "").strip().lower() or default
    if effort not in REASONING_EFFORTS:
        raise ConfigError(f"{name} must be one of {', '.join(REASONING_EFFORTS)}; got '{effort}'.")
    return effort


def load_settings(env: dict[str, str] | None = None) -> LLMSettings:
    """Read LLM settings from ``env`` (defaults to ``os.environ``).

    Raises:
        ConfigError: if the provider is unknown or a required variable is unset.
    """
    env = dict(os.environ) if env is None else env
    provider = env.get("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower() or DEFAULT_PROVIDER
    if provider not in _REQUIRED_VARS:
        raise ConfigError(
            f"Unknown LLM_PROVIDER '{provider}'. Use one of: {', '.join(_REQUIRED_VARS)}."
        )
    missing = [name for name in _REQUIRED_VARS[provider] if not env.get(name, "").strip()]
    if missing:
        raise ConfigError(
            f"Missing environment variable(s) for provider '{provider}': "
            f"{', '.join(missing)}. Copy .env.example to .env and fill them in."
        )

    efforts = {
        "retriever_reasoning_effort": _reasoning_effort(
            env, "RETRIEVER_REASONING_EFFORT", DEFAULT_RETRIEVER_REASONING_EFFORT
        ),
        "generator_reasoning_effort": _reasoning_effort(
            env, "GENERATOR_REASONING_EFFORT", DEFAULT_GENERATOR_REASONING_EFFORT
        ),
    }
    if provider == "azure":
        return LLMSettings(
            provider=provider,
            model=env["AZURE_OPENAI_DEPLOYMENT"],
            api_key=env["AZURE_OPENAI_API_KEY"],
            azure_endpoint=env["AZURE_OPENAI_ENDPOINT"],
            azure_api_version=env.get("AZURE_OPENAI_API_VERSION", DEFAULT_AZURE_API_VERSION),
            **efforts,
        )
    if provider == "anthropic":
        return LLMSettings(
            provider=provider,
            model=env.get("ANTHROPIC_MODEL", "").strip() or DEFAULT_ANTHROPIC_MODEL,
            api_key=env["ANTHROPIC_API_KEY"],
            **efforts,
        )
    return LLMSettings(
        provider=provider,
        model=env.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        api_key=env["OPENAI_API_KEY"],
        **efforts,
    )


def build_chat_model(
    settings: LLMSettings, reasoning_effort: str, disable_streaming: bool = False
) -> BaseChatModel:
    """Create a chat model for ``settings``.

    ``reasoning_effort`` applies to the OpenAI providers only. ``disable_streaming``
    suits agents whose tokens are never shown to the user.
    """
    if settings.provider == "anthropic":
        return ChatAnthropic(
            model=settings.model,
            api_key=settings.api_key,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            disable_streaming=disable_streaming,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=MAX_RETRIES,
        )
    common = {
        "api_key": settings.api_key,
        "use_responses_api": True,
        "reasoning": {"effort": reasoning_effort},
        "disable_streaming": disable_streaming,
        "timeout": REQUEST_TIMEOUT_SECONDS,
        "max_retries": MAX_RETRIES,
    }
    if settings.provider == "azure":
        return AzureChatOpenAI(
            azure_endpoint=settings.azure_endpoint,
            azure_deployment=settings.model,
            api_version=settings.azure_api_version,
            **common,
        )
    return ChatOpenAI(model=settings.model, **common)
