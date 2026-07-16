"""LLM call path: retries, prompt-cache markers, Anthropic request hygiene.

Ported from archipelago ``runner/utils/llm.py`` with the runner-specific
plumbing removed (LiteLLM proxy routing, Datadog latency baselines, spend
tags, streaming, Responses API). What remains is the behavior the agent
loop depends on:

- retry with backoff that skips deterministic failures (context overflow,
  validation 400s),
- ``cache_control`` markers for Anthropic-family models (system prompt,
  last tool, rolling last message),
- empty-content normalization that Anthropic would otherwise 400 on.
"""

import asyncio
import random
from typing import Any

from litellm import acompletion
from litellm.exceptions import (
    APIConnectionError,
    BadGatewayError,
    BadRequestError,
    ContextWindowExceededError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from litellm.files.main import ModelResponse
from litellm.types.utils import Message
from loguru import logger

from .messages import AnyMessage

_RETRYABLE_EXCEPTIONS = (
    RateLimitError,
    Timeout,
    BadRequestError,
    ServiceUnavailableError,
    APIConnectionError,
    InternalServerError,
    BadGatewayError,
)

_MAX_RETRIES = 10
_BASE_BACKOFF = 5.0
_JITTER = 5.0


def responses_args_to_completions(extra_args: dict[str, Any]) -> dict[str, Any]:
    """Convert Responses API extra_args to Chat Completions API equivalents.

    The Responses API uses ``{"reasoning": {"effort": "high", ...}}`` while
    Chat Completions uses ``{"reasoning_effort": "high"}`` as a top-level
    param. Sending the Responses-API shape to a Chat-Completions endpoint
    yields ``Unknown parameter: 'reasoning'`` 400s.
    """
    result = {k: v for k, v in extra_args.items() if k != "reasoning"}
    reasoning = extra_args.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if effort and "reasoning_effort" not in result:
            result["reasoning_effort"] = effort
    return result


# Anthropic-family models *require* an explicit ``cache_control`` field to
# enable prompt caching; the other providers cache input prefixes
# automatically, so markers are only attached for these prefixes.
_CACHE_CONTROL_ALLOWLIST: frozenset[str] = frozenset({"anthropic", "bedrock"})
# Explicit 5-minute TTL so a future change to Anthropic's default is not
# silently inherited.
_EPHEMERAL_CACHE: dict[str, str] = {"type": "ephemeral", "ttl": "5m"}


def _is_cache_control_allowed(model: str) -> bool:
    """Whether ``model`` accepts Anthropic-style ``cache_control`` markers.

    Matches on the ``provider/`` prefix only — ``openrouter/anthropic/foo``
    deliberately does NOT match (OpenRouter caches automatically).
    """
    if model in _CACHE_CONTROL_ALLOWLIST:
        return True
    segments = model.split("/")
    return bool(segments) and segments[0] in _CACHE_CONTROL_ALLOWLIST


def _is_empty_text_block(block: Any) -> bool:
    """Whether ``block`` is a text block with empty/whitespace-only text.

    Anthropic rejects ``cache_control`` on empty text blocks, so the cache
    helpers must never mark one.
    """
    return block.get("type") == "text" and not (block.get("text") or "").strip()


def _content_is_empty(content: Any) -> bool:
    """Whether a message's ``content`` carries no usable text or blocks."""
    if content is None:
        return True
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        return all(isinstance(b, dict) and _is_empty_text_block(b) for b in content)
    return False


def _with_cached_system_prompt(
    messages: list[AnyMessage], model: str
) -> list[AnyMessage]:
    """Return ``messages`` with the system prompt marked ephemerally cacheable."""
    if not _is_cache_control_allowed(model):
        return messages

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content:
            block = {"type": "text", "text": content, "cache_control": _EPHEMERAL_CACHE}
            new_msg: AnyMessage = {"role": "system", "content": [block]}
            return [*messages[:i], new_msg, *messages[i + 1 :]]
        if isinstance(content, list) and content:
            last = content[-1]
            if (
                isinstance(last, dict)
                and "cache_control" not in last
                and not _is_empty_text_block(last)
            ):
                cached_last = {**last, "cache_control": _EPHEMERAL_CACHE}
                new_msg = {**msg, "content": [*content[:-1], cached_last]}
                return [*messages[:i], new_msg, *messages[i + 1 :]]
        return messages
    return messages


def _with_cached_tools(tools: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    """Return ``tools`` with the last tool marked ephemerally cacheable.

    Anthropic's prompt cache extends from the start of the request through
    the *last* ``cache_control`` marker; placing one on the final tool
    definition caches the entire system+tools prefix together.
    """
    if not _is_cache_control_allowed(model):
        return tools
    if not tools:
        return tools
    last = tools[-1]
    if not isinstance(last, dict) or last.get("cache_control") is not None:
        return tools
    cached_last = {**last, "cache_control": _EPHEMERAL_CACHE}
    return [*tools[:-1], cached_last]


def _with_cached_last_message(
    messages: list[AnyMessage], model: str
) -> list[AnyMessage]:
    """Return ``messages`` with the most recent message marked ephemerally cacheable.

    History is append-only across the agent loop, so marking the last
    message extends the cached prefix through every prior message; the next
    turn's lookup falls back to it as the longest matching prefix.
    Pydantic ``Message`` instances are skipped — the system + tools
    breakpoints still cover the static prefix when the rolling marker no-ops.
    """
    if not _is_cache_control_allowed(model):
        return messages
    if not messages:
        return messages
    last = messages[-1]
    if not isinstance(last, dict):
        return messages

    content = last.get("content")

    if isinstance(content, str):
        if not content:
            return messages
        block = {"type": "text", "text": content, "cache_control": _EPHEMERAL_CACHE}
        new_msg: AnyMessage = {**last, "content": [block]}
        return [*messages[:-1], new_msg]

    if isinstance(content, list) and content:
        last_block = content[-1]
        if (
            isinstance(last_block, dict)
            and "cache_control" not in last_block
            and not _is_empty_text_block(last_block)
        ):
            cached_block = {**last_block, "cache_control": _EPHEMERAL_CACHE}
            new_msg = {**last, "content": [*content[:-1], cached_block]}
            return [*messages[:-1], new_msg]

    return messages


_ANTHROPIC_EMPTY_TEXT_PLACEHOLDER = "[empty]"


def _with_nonempty_text_content(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Replace empty text content with a non-whitespace placeholder.

    Anthropic rejects any request containing an empty text block (HTTP 400).
    ``None`` content is left untouched — it is omitted from the serialized
    request, which is valid (e.g. an assistant turn that is purely
    tool_calls).
    """
    placeholder = _ANTHROPIC_EMPTY_TEXT_PLACEHOLDER
    normalized: list[AnyMessage] = []
    for msg in messages:
        if isinstance(msg, Message):
            if msg.content == "":
                msg = msg.model_copy(update={"content": placeholder})
            normalized.append(msg)
            continue
        if isinstance(msg, dict):
            content = msg.get("content")
            if content == "" or content == []:
                msg = {**msg, "content": placeholder}
            elif isinstance(content, list):
                new_content = [
                    {**block, "text": placeholder}
                    if isinstance(block, dict)
                    and block.get("type") == "text"
                    and not block.get("text")
                    else block
                    for block in content
                ]
                if new_content != content:
                    msg = {**msg, "content": new_content}
        normalized.append(msg)
    return normalized


def _drop_trailing_empty_assistant(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Drop a trailing assistant turn that carries no content.

    An empty assistant "prefill" turn is invalid downstream: filling it lets
    the rolling cache-control breakpoint land on it, and Anthropic rejects
    the empty prefill outright under extended thinking. Turns carrying
    tool_calls, thinking_blocks, or reasoning_content are preserved.
    """
    if not messages:
        return messages
    last = messages[-1]
    if isinstance(last, Message):
        if (
            last.role == "assistant"
            and not last.tool_calls
            and not getattr(last, "thinking_blocks", None)
            and not getattr(last, "reasoning_content", None)
            and _content_is_empty(last.content)
        ):
            return messages[:-1]
        return messages
    if (
        isinstance(last, dict)
        and last.get("role") == "assistant"
        and not last.get("tool_calls")
        and not last.get("thinking_blocks")
        and not last.get("reasoning_content")
        and _content_is_empty(last.get("content"))
    ):
        return messages[:-1]
    return messages


def _is_context_window_error(e: Exception) -> bool:
    """Detect context window errors that LiteLLM doesn't properly classify.

    Some providers (notably Gemini) return context window errors as
    BadRequestError instead of ContextWindowExceededError.
    """
    error_str = str(e).lower()
    context_patterns = [
        "token count exceeds",
        "context_length_exceeded",
        "context length exceeded",
        "maximum context length",
        "maximum number of tokens",
        "prompt is too long",
        "input too long",
        "exceeds the model's maximum context",
        "exceeds the context window",
    ]
    return any(pattern in error_str for pattern in context_patterns)


def _is_non_retriable_bad_request(e: Exception) -> bool:
    """Detect BadRequestErrors that are deterministic and should NOT be retried.

    Patterns must be specific enough to avoid matching transient errors like
    rate limits (e.g., "maximum of 100 requests" should NOT match).
    """
    error_str = str(e).lower()
    non_retriable_patterns = [
        # Tool count errors - be specific to avoid matching rate limits
        "tools are supported",  # "Maximum of 128 tools are supported"
        "too many tools",
        # Model/auth errors
        "model not found",
        "does not exist",
        "invalid api key",
        "authentication failed",
        "unauthorized",
        "unsupported parameter",
        "unsupported value",
        "unknown parameter",
        # Model capability mismatch
        "does not support multimodal",
        "is not a multimodal model",
        # Anthropic request validation (deterministic; retrying won't help)
        "text content blocks must be non-empty",
        "text content blocks must contain non-whitespace text",
        "max allowed size for many-image",
        "2000 pixels",
        "exceeds 5 mb",
        "5242880",
        "file format is invalid or unsupported",
        "image dimensions exceed max allowed size",
        "at least one non-system message",
    ]
    return any(pattern in error_str for pattern in non_retriable_patterns)


def _should_skip_retry(e: Exception) -> bool:
    """Combined check for all non-retriable errors."""
    if isinstance(e, ContextWindowExceededError):
        return True
    return _is_context_window_error(e) or _is_non_retriable_bad_request(e)


async def generate_response(
    model: str,
    messages: list[AnyMessage],
    tools: list[dict[str, Any]],
    llm_response_timeout: int,
    extra_args: dict[str, Any] | None = None,
) -> ModelResponse:
    """Generate a response from the LLM with retry logic.

    Args:
        model: LiteLLM model identifier (``provider/model``)
        messages: The conversation messages (input dicts or output Messages)
        tools: Available tools for the model to call
        llm_response_timeout: Timeout in seconds for the LLM response
        extra_args: Additional top-level arguments for the completion call

    Returns:
        The model response
    """
    extra = responses_args_to_completions(extra_args or {})

    if model.startswith("anthropic/"):
        # A trailing empty assistant "prefill" turn would otherwise be
        # placeholder-filled below and used as an invalid assistant prefill
        # under extended thinking. Drop it so the request ends on the user turn.
        messages = _drop_trailing_empty_assistant(messages)
        # Anthropic 400s on any empty/whitespace-only text block.
        messages = _with_nonempty_text_content(messages)

    cached_messages = _with_cached_last_message(
        _with_cached_system_prompt(messages, model), model
    )
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": cached_messages,
        "timeout": llm_response_timeout,
        # This loop owns retries — pin num_retries=0 so LiteLLM doesn't retry
        # on top, compounding attempts. Caller's extra wins via the spread.
        "num_retries": 0,
        **extra,
    }
    if tools:
        kwargs["tools"] = _with_cached_tools(tools, model)

    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = await acompletion(**kwargs)
            return ModelResponse.model_validate(response)
        except _RETRYABLE_EXCEPTIONS as e:
            if _should_skip_retry(e):
                raise
            last_exc = e
            if attempt >= _MAX_RETRIES:
                break
            backoff = _BASE_BACKOFF * (2**attempt) + random.uniform(0, _JITTER)
            backoff = min(backoff, 120.0)
            logger.warning(
                f"LLM call failed (attempt {attempt + 1}/{_MAX_RETRIES + 1}), "
                f"retrying in {backoff:.1f}s: {repr(e)}"
            )
            await asyncio.sleep(backoff)

    assert last_exc is not None
    raise last_exc
