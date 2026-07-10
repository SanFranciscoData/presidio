"""ReAct Toolbelt Agent with ReSum context management.

Ported from archipelago ``react_toolbelt_agent/main.py``. Structural changes:
- Input is a plain ``RunConfig`` (model, instruction, MCP server list) instead
  of the archipelago runner's ``AgentRunInput``; the MCP connection is built
  from presidio's per-task server list rather than a single gateway URL.
- Per-call usage/timestamps are kept in a sidecar (keyed by message object id)
  so the trajectory writer can attach per-step metrics without mutating the
  messages sent back to the LLM.
- An optional ``on_step`` checkpoint callback persists the trajectory after
  every step, so a hard kill (trial timeout) still leaves a usable trajectory.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fastmcp import Client as FastMCPClient
from litellm import Choices
from litellm.exceptions import ContextWindowExceededError, Timeout
from litellm.experimental_mcp_client import call_openai_tool, load_mcp_tools
from litellm.files.main import ModelResponse
from litellm.types.utils import Message as LitellmOutputMessage
from loguru import logger

from .errors import is_fatal_mcp_error, is_system_error
from .llm import generate_response
from .mcp_utils import build_mcp_config, content_blocks_to_messages, drain_shielded_task
from .messages import AgentStatus, AnyMessage
from .resum import ReSumManager
from .tool_result import truncate_tool_messages
from .tools import (
    FINAL_ANSWER_TOOL,
    META_TOOL_NAMES,
    META_TOOLS,
    MetaToolHandler,
    parse_final_answer,
)
from .usage import UsageTracker


@dataclass
class RunConfig:
    """Everything the agent needs for one run."""

    model: str
    instruction: str
    mcp_servers: list[dict[str, Any]]
    system_prompt: str | None = None
    timeout: int = 10800
    max_steps: int = 250
    tool_call_timeout: int = 60
    llm_response_timeout: int = 600
    max_toolbelt_size: int = 80
    extra_args: dict[str, Any] = field(default_factory=dict)

    def initial_messages(self) -> list[AnyMessage]:
        messages: list[AnyMessage] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.instruction})
        return messages


@dataclass
class RunResult:
    """Final state of an agent run."""

    messages: list[AnyMessage]
    status: AgentStatus
    final_answer: str | None
    final_status: str
    time_elapsed: float
    usage: dict[str, Any]
    # Sidecar metadata per message object id: {"usage": ..., "timestamp": ...}
    msg_meta: dict[int, dict[str, Any]]


class ReActAgent:
    """ReAct Toolbelt Agent with ReSum context management."""

    def __init__(
        self,
        config: RunConfig,
        on_step: Callable[["ReActAgent"], None] | None = None,
    ):
        self.config = config
        self.model = config.model
        self.messages: list[AnyMessage] = config.initial_messages()
        self.on_step = on_step

        if not config.mcp_servers:
            raise ValueError(
                "At least one MCP server is required for the react toolbelt agent"
            )
        self.mcp_client = FastMCPClient(build_mcp_config(config.mcp_servers))

        # Components
        self.resum = ReSumManager(self.model, config.extra_args)

        # Toolbelt state
        self.all_tools: dict[str, dict[str, Any]] = {}
        self.toolbelt: set[str] = set()
        self.meta_tool_handler: MetaToolHandler | None = None

        # Agent state
        self._finalized: bool = False
        self._final_answer: str | None = None
        self._final_status: str = "completed"
        self.status: AgentStatus = AgentStatus.PENDING
        self.start_time: float | None = None
        self.usage_tracker = UsageTracker(model=self.model)
        # Per-message sidecar (usage dict, wall-clock timestamp) keyed by
        # id(message). Message objects survive ReSum history moves intact,
        # so ids stay valid for the final serialization pass.
        self.msg_meta: dict[int, dict[str, Any]] = {}

    def _get_tools(self) -> list[dict[str, Any]]:
        """Get tools for LLM: meta-tools + toolbelt + final_answer."""
        toolbelt_tools = [self.all_tools[name] for name in self.toolbelt]
        return list(META_TOOLS) + toolbelt_tools + [FINAL_ANSWER_TOOL]

    async def _initialize_tools(self, client: Any) -> None:
        """Load tools from the MCP server(s)."""
        tools: list[dict[str, Any]] = await load_mcp_tools(
            client.session, format="openai"
        )  # pyright: ignore[reportAssignmentType]

        for tool in tools:
            name = tool.get("function", {}).get("name")
            if name:
                self.all_tools[name] = tool

        self.meta_tool_handler = MetaToolHandler(
            self.all_tools, self.toolbelt, self.config.max_toolbelt_size
        )

        logger.info(
            f"Loaded {len(self.all_tools)} MCP tools (toolbelt starts empty): "
            f"{sorted(self.all_tools.keys())}"
        )

    async def step(self, client: Any) -> None:
        """Execute one step of the ReAct loop."""
        # Proactive ReSum check
        if self.resum.should_summarize(self.messages):
            logger.info("Summarizing context")
            try:
                before = len(self.messages)
                self.messages = await self.resum.summarize(self.messages)
                # Only flag a compaction when context was actually reduced;
                # summarize() can no-op and return the messages unchanged.
                if len(self.messages) < before:
                    self.usage_tracker.track_compaction()
            except Exception as e:
                logger.error(f"Summarization failed: {e}")

        # Call LLM
        try:
            response: ModelResponse = await generate_response(
                self.model,
                self.messages,
                self._get_tools(),
                self.config.llm_response_timeout,
                self.config.extra_args,
            )
        except ContextWindowExceededError:
            logger.warning("Context exceeded, summarizing")
            before = len(self.messages)
            self.messages = await self.resum.summarize(self.messages)
            if len(self.messages) < before:
                self.usage_tracker.track_compaction()
            return
        except Timeout:
            logger.error("LLM timeout")
            return
        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise

        call_usage = self.usage_tracker.track(response)
        choices = response.choices
        if not choices or not isinstance(choices[0], Choices):
            logger.warning(
                "LLM returned an empty response with no choices, "
                "re-prompting with 'continue'"
            )
            self.messages.append(
                {"role": "user", "content": "Continue. Use final_answer when done."}
            )
            return

        response_message = LitellmOutputMessage.model_validate(choices[0].message)
        self.msg_meta[id(response_message)] = {
            "usage": call_usage,
            "timestamp": time.time(),
        }
        tool_calls = getattr(response_message, "tool_calls", None)
        content = getattr(response_message, "content", None)

        # Log reasoning if present (reasoning models)
        if getattr(response_message, "reasoning_content", None):
            logger.info(f"[reasoning] {response_message.reasoning_content}")

        if content:
            logger.info(f"[response] {content}")

        if tool_calls:
            tool_names = [tc.function.name for tc in tool_calls]
            logger.info(f"Calling {len(tool_calls)} tool(s): {', '.join(tool_names)}")
        elif not content:
            finish_reason = choices[0].finish_reason if choices else None
            logger.warning(
                f"No content and no tool calls (finish_reason={finish_reason})"
            )

        self.messages.append(response_message)

        if tool_calls:
            await self._handle_tool_calls(client, tool_calls)
        else:
            self.messages.append(
                {
                    "role": "user",
                    "content": "No tools called. Use final_answer to submit your "
                    "answer. Please continue completing the task.",
                }
            )

    async def _handle_tool_calls(self, client: Any, tool_calls: list[Any]) -> None:
        """Process tool calls."""
        mcp_tool_calls: list[Any] = []

        for tool_call in tool_calls:
            name = tool_call.function.name

            # Final answer - validate todos, then handle and return
            if name == "final_answer":
                assert self.meta_tool_handler
                incomplete = self.meta_tool_handler.get_incomplete_todos()
                if incomplete:
                    incomplete_list = ", ".join(
                        f"'{t.id}' ({t.status.value})" for t in incomplete
                    )
                    error_msg = (
                        f"ERROR: Cannot submit final_answer with incomplete todos. "
                        f"You have {len(incomplete)} incomplete task(s): "
                        f"{incomplete_list}. "
                        f"Use todo_write to mark each as 'completed' or 'cancelled' "
                        f"first."
                    )
                    logger.warning(
                        f"final_answer rejected: {len(incomplete)} incomplete todos"
                    )
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": "final_answer",
                            "content": error_msg,
                        }
                    )
                    return

                answer, status = parse_final_answer(tool_call.function.arguments)
                logger.info(f"[final_answer] {answer}")

                self._finalized = True
                self._final_answer = answer
                self._final_status = status

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": "final_answer",
                        "content": answer,
                    }
                )
                return

            # Meta-tool - handle locally
            if name in META_TOOL_NAMES:
                logger.info(f"Meta-tool: {name}({tool_call.function.arguments})")
                assert self.meta_tool_handler
                result = self.meta_tool_handler.handle(
                    name, tool_call.function.arguments
                )
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": result,
                    }
                )
                continue

            # MCP tool - collect for batch execution
            mcp_tool_calls.append(tool_call)

        # Execute MCP tools (using shared client connection)
        deferred_image_messages: list[dict[str, Any]] = []
        for tool_call in mcp_tool_calls:
            await self._execute_mcp_tool(client, tool_call, deferred_image_messages)
        self.messages.extend(deferred_image_messages)

    async def _execute_mcp_tool(
        self,
        client: Any,
        tool_call: Any,
        deferred_image_messages: list[dict[str, Any]],
    ) -> None:
        """Execute an MCP tool call."""
        name = tool_call.function.name

        if name not in self.toolbelt:
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": f"Error: '{name}' not in toolbelt. "
                    "Use toolbelt_add_tool first.",
                }
            )
            return

        logger.info(f"Calling tool {name}({tool_call.function.arguments})")

        shielded_task = asyncio.ensure_future(
            call_openai_tool(client.session, tool_call)
        )
        try:
            result = await asyncio.wait_for(
                asyncio.shield(shielded_task),
                timeout=self.config.tool_call_timeout,
            )
        except TimeoutError:
            logger.error(f"Tool call {name} timed out")
            await drain_shielded_task(shielded_task)
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": "Tool call timed out",
                }
            )
            return
        except Exception as e:
            if is_fatal_mcp_error(e):
                logger.error(f"Fatal MCP error, ending run: {repr(e)}")
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": f"Fatal error: {e}",
                    }
                )
                raise
            logger.error(f"Error calling tool {name}: {repr(e)}")
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": f"Error: {e}",
                }
            )
            return

        if not result.content:
            logger.error(f"Tool {name} returned no content")
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": "No content returned",
                }
            )
            return

        messages = content_blocks_to_messages(
            result.content,
            tool_call.id,
            name,
            self.model,
            deferred_image_messages=deferred_image_messages,
        )
        truncate_tool_messages(messages, self.model)
        self.messages.extend(messages)

    def _build_result(self) -> RunResult:
        return RunResult(
            messages=self.resum.get_full_history(self.messages),
            status=self.status,
            final_answer=self._final_answer,
            final_status=self._final_status,
            time_elapsed=time.time() - self.start_time if self.start_time else 0,
            usage=self.usage_tracker.to_dict(),
            msg_meta=self.msg_meta,
        )

    def _checkpoint(self) -> None:
        if self.on_step is None:
            return
        try:
            self.on_step(self)
        except Exception as e:
            logger.warning(f"Checkpoint callback failed: {e}")

    async def run(self) -> RunResult:
        """Run the agent loop with a single MCP connection."""
        try:
            async with asyncio.timeout(self.config.timeout):
                # Single MCP connection for entire agent lifecycle
                async with self.mcp_client as client:
                    logger.info(f"Starting ReAct Toolbelt agent with {self.model}")
                    await self._initialize_tools(client)

                    self.start_time = time.time()
                    self.status = AgentStatus.RUNNING

                    for step in range(self.config.max_steps):
                        if self._finalized:
                            logger.info(f"Finalized after {step} steps")
                            break
                        logger.info(f"Starting step {step + 1}")
                        await self.step(client)
                        self._checkpoint()

                    if not self._finalized:
                        logger.error(
                            f"Not finalized after {self.config.max_steps} steps"
                        )
                        self.status = AgentStatus.FAILED
                    else:
                        self.status = AgentStatus.COMPLETED

                    return self._build_result()

        except TimeoutError:
            logger.error(f"Timeout after {self.config.timeout}s")
            self.status = AgentStatus.ERROR
            return self._build_result()

        except asyncio.CancelledError:
            logger.error("Cancelled")
            self.status = AgentStatus.CANCELLED
            return self._build_result()

        except Exception as e:
            logger.error(f"Error: {e}")
            self.status = (
                AgentStatus.ERROR if is_system_error(e) else AgentStatus.FAILED
            )
            return self._build_result()
