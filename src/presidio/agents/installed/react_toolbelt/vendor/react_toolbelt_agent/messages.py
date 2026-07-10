"""Message helpers shared across the agent loop.

Slimmed from archipelago ``runner/agents/models.py``: keeps the
TypedDict-or-Pydantic accessors and content normalizers; drops the runner's
AgentRunInput/AgentDefn registry models (replaced by ``RunConfig`` in
``__main__``).
"""

from enum import StrEnum
from typing import Any

from litellm.types.utils import Message

# Messages are either plain dicts (inputs, tool results) or litellm's
# pydantic ``Message`` (LLM outputs). Type alias kept for readability.
AnyMessage = dict[str, Any] | Message


class AgentStatus(StrEnum):
    """Status of an agent run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    ERROR = "error"


def get_msg_role(msg: AnyMessage) -> str:
    """Get role from either TypedDict or Pydantic Message."""
    if isinstance(msg, Message):
        return msg.role
    return msg["role"]


def get_msg_content(msg: AnyMessage) -> Any:
    """Get content from either TypedDict or Pydantic Message."""
    if isinstance(msg, Message):
        return msg.content
    return msg.get("content")


def get_msg_attr(msg: AnyMessage, key: str, default: Any = None) -> Any:
    """Get arbitrary attribute from either TypedDict or Pydantic Message."""
    if isinstance(msg, Message):
        return getattr(msg, key, default)
    return msg.get(key, default)


def materialize_msg_content(content: Any) -> Any:
    """Return message content as a plain string, list of blocks, or None.

    LiteLLM/Pydantic may expose multipart assistant content as a lazy
    ``ValidatorIterator`` rather than a materialized list.
    """
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        return content
    if hasattr(content, "__iter__") and not isinstance(content, (bytes, dict)):
        try:
            blocks = list(content)
        except TypeError:
            return content
        if blocks:
            return blocks
    return content


def content_to_str(content: Any) -> str:
    """Normalize message content to a string.

    Some providers (e.g. Anthropic) return content as a list of blocks
    like [{'type': 'text', 'text': '...'}]. Use this when you need a plain string.
    """
    content = materialize_msg_content(content)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text", "")
                if isinstance(text, str) and text:
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else ""
    return str(content)


def message_to_dict(msg: AnyMessage) -> dict[str, Any]:
    """Serialize a message to a plain JSON-safe dict for trajectory output."""
    if isinstance(msg, Message):
        data = msg.model_dump(exclude_none=True)
    else:
        data = dict(msg)
    content = materialize_msg_content(data.get("content"))
    if content is not None:
        data["content"] = content
    return data
