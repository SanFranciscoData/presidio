"""MCP client helpers for the agent loop.

Ported from archipelago ``runner/utils/mcp.py``. Differences vs. upstream:
- ``build_mcp_config`` builds a FastMCP config from presidio's per-task MCP
  server list instead of a single archipelago gateway URL.
- Images are passed through as data URIs unchanged (no Anthropic downscale
  policy; see package docstring).
"""

import asyncio
from typing import Any

from loguru import logger
from mcp.types import ContentBlock, ImageContent, TextContent

# Grace period (seconds) to wait for a shielded MCP tool call to finish
# after the primary timeout expires, before forcibly cancelling it.
SHIELDED_TASK_GRACE_SECONDS = 5.0


async def drain_shielded_task(task: asyncio.Task[Any]) -> None:
    """Wait for a shielded MCP task to finish, cancelling if it takes too long.

    After an ``asyncio.wait_for`` timeout, the shielded inner task is still
    running and holds the MCP session open.  Attempting a new tool call on the
    same session while the old one is in-flight causes a
    ``RuntimeError("Client is not connected")`` from the streamable-http
    transport.

    This helper gives the task a short grace period to complete naturally.  If
    it doesn't finish in time, the task is cancelled so the session is released
    before the next call.
    """
    if task.done():
        return
    try:
        await asyncio.wait_for(task, timeout=SHIELDED_TASK_GRACE_SECONDS)
    except (TimeoutError, Exception):
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


def build_mcp_config(servers: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the FastMCP client config from presidio task MCP server entries.

    Each entry mirrors presidio's ``MCPServerConfig``:
    ``{"name", "transport", "url"?, "command"?, "args"?}`` where transport is
    ``sse`` | ``streamable-http`` | ``stdio``.

    With multiple servers FastMCP prefixes tool names with the server name;
    with a single server tool names are unprefixed.
    """
    mcp_servers: dict[str, Any] = {}
    for server in servers:
        name = server["name"]
        transport = server.get("transport", "sse")
        if transport == "stdio":
            mcp_servers[name] = {
                "command": server["command"],
                "args": server.get("args", []),
            }
        else:
            mcp_servers[name] = {
                "transport": transport,
                "url": server["url"],
            }
    return {"mcpServers": mcp_servers}


def content_blocks_to_messages(
    content_blocks: list[ContentBlock],
    tool_call_id: str,
    name: str,
    model: str,
    deferred_image_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert MCP content blocks to a single LiteLLM tool message.

    Each tool_use must have exactly one tool_result. This function combines all
    content blocks into a single tool message to satisfy API requirements for
    Anthropic, OpenAI, and other providers.

    For non-Anthropic models, images cannot be embedded in tool results, so they
    are appended to deferred_image_messages as user messages. The caller is
    responsible for adding them to self.messages after all tool responses.
    This list is mutated in place.

    Args:
        content_blocks: MCP content blocks from tool result
        tool_call_id: The tool call ID to associate with the result
        name: The tool name
        model: The model being used
        deferred_image_messages: Mutable list that image user messages are
            appended to (mutated in place). Callers should extend self.messages
            with this list after all tool responses are added.

    Returns:
        List containing exactly one tool message.
    """
    # Anthropic supports images directly in tool results
    supports_image_tool_results = model.startswith("anthropic/")

    text_contents: list[str] = []
    image_data_uris: list[str] = []

    for content_block in content_blocks:
        match content_block:
            case TextContent():
                block = TextContent.model_validate(content_block)
                text_contents.append(block.text)

            case ImageContent():
                block = ImageContent.model_validate(content_block)
                image_data_uris.append(f"data:{block.mimeType};base64,{block.data}")

            case _:
                logger.warning(f"Content block type {content_block.type} not supported")
                text_contents.append("Unable to parse tool call response")

    messages: list[dict[str, Any]] = []

    if supports_image_tool_results:
        content: list[dict[str, Any]] = []
        for text in text_contents:
            content.append({"type": "text", "text": text or " "})
        for data_uri in image_data_uris:
            content.append({"type": "image_url", "image_url": {"url": data_uri}})

        if image_data_uris and not any(
            block.get("type") == "text" for block in content
        ):
            content.insert(0, {"type": "text", "text": " "})

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": name,
                "content": content if content else [{"type": "text", "text": " "}],
            }
        )
    else:
        content = [{"type": "text", "text": text or " "} for text in text_contents]

        if image_data_uris and not content:
            content.append(
                {"type": "text", "text": f"Image(s) returned by {name} tool"}
            )

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": name,
                "content": content if content else [{"type": "text", "text": " "}],
            }
        )

        # Image workaround: non-Anthropic models don't support images in tool
        # results, so they are deferred as user messages added after all tool
        # responses.
        for data_uri in image_data_uris:
            deferred_image_messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            )

    return messages
