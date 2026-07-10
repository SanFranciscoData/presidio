import asyncio
import json
from pathlib import Path

import pytest

from presidio.agents.factory import AgentFactory
from presidio.agents.installed.mini_swe_agent import convert_mini_swe_agent_to_atif
from presidio.agents.installed.react_toolbelt import ReactToolbelt
from presidio.agents.installed.react_toolbelt.vendor.react_toolbelt_agent.resum import (
    _find_safe_cut_index,
)
from presidio.agents.installed.react_toolbelt.vendor.react_toolbelt_agent.tool_result import (
    truncate_tool_messages,
)
from presidio.agents.installed.react_toolbelt.vendor.react_toolbelt_agent.tools import (
    FINAL_ANSWER_TOOL,
    META_TOOL_NAMES,
    MetaToolHandler,
    parse_final_answer,
)
from presidio.agents.installed.react_toolbelt.vendor.react_toolbelt_agent.trajectory import (
    build_trajectory_dict,
)
from presidio.models.agent.context import AgentContext
from presidio.models.agent.name import AgentName


# ---------------------------------------------------------------------------
# Registration / wrapper
# ---------------------------------------------------------------------------


def test_react_toolbelt_registered_in_factory(tmp_path: Path):
    agent = AgentFactory.create_agent_from_name(
        AgentName.REACT_TOOLBELT,
        logs_dir=tmp_path,
        model_name="anthropic/claude-sonnet-5",
    )
    assert isinstance(agent, ReactToolbelt)
    assert agent.name() == "react-toolbelt"
    assert agent.SUPPORTS_ATIF


def test_install_spec_creates_venv_with_agent_deps(tmp_path: Path):
    agent = ReactToolbelt(logs_dir=tmp_path, model_name="anthropic/claude-sonnet-5")
    spec = agent.install_spec()
    venv_step = spec.steps[-1].run
    assert "uv venv" in venv_step
    assert "litellm" in venv_step
    assert "fastmcp" in venv_step
    # The python-dotenv conflict override must be applied (litellm pins
    # ==1.0.1, fastmcp needs >=1.1.0).
    assert "--override" in venv_step
    assert "python-dotenv>=1.1.0" in venv_step


def test_agent_config_kwargs_flow_into_run_config(tmp_path: Path):
    agent = ReactToolbelt(
        logs_dir=tmp_path,
        model_name="anthropic/claude-sonnet-5",
        max_steps=50,
        agent_timeout=600,
        max_toolbelt_size=10,
    )
    assert agent._agent_config == {
        "timeout": 600,
        "max_steps": 50,
        "max_toolbelt_size": 10,
    }


def test_run_requires_mcp_servers(tmp_path: Path):
    agent = ReactToolbelt(logs_dir=tmp_path, model_name="anthropic/claude-sonnet-5")
    with pytest.raises(ValueError, match="MCP server"):
        asyncio.run(agent.run("do the thing", environment=None, context=AgentContext()))


def test_run_requires_provider_prefixed_model(tmp_path: Path):
    agent = ReactToolbelt(logs_dir=tmp_path, model_name="claude-sonnet-5")
    with pytest.raises(ValueError, match="provider/model_name"):
        asyncio.run(agent.run("do the thing", environment=None, context=AgentContext()))


# ---------------------------------------------------------------------------
# Vendored meta-tools
# ---------------------------------------------------------------------------


def _make_handler(max_size: int = 3) -> MetaToolHandler:
    all_tools = {
        name: {"type": "function", "function": {"name": name, "description": name}}
        for name in ("browser_click", "browser_snapshot", "browser_type", "extra")
    }
    return MetaToolHandler(all_tools, set(), max_size)


def test_toolbelt_add_list_remove_cycle():
    handler = _make_handler()

    listed = json.loads(handler.handle("toolbelt_list_tools", "{}"))
    assert "browser_click" in listed

    result = json.loads(
        handler.handle("toolbelt_add_tool", json.dumps({"tool_name": "browser_click"}))
    )
    assert result == {"success": True, "toolbelt_size": 1}
    assert "browser_click" not in json.loads(
        handler.handle("toolbelt_list_tools", "{}")
    )

    # Duplicate add is rejected
    dup = json.loads(
        handler.handle("toolbelt_add_tool", json.dumps({"tool_name": "browser_click"}))
    )
    assert "error" in dup

    result = json.loads(
        handler.handle(
            "toolbelt_remove_tool", json.dumps({"tool_name": "browser_click"})
        )
    )
    assert result == {"success": True, "toolbelt_size": 0}


def test_toolbelt_enforces_max_size():
    handler = _make_handler(max_size=2)
    for name in ("browser_click", "browser_snapshot"):
        assert json.loads(
            handler.handle("toolbelt_add_tool", json.dumps({"tool_name": name}))
        )["success"]
    full = json.loads(
        handler.handle("toolbelt_add_tool", json.dumps({"tool_name": "browser_type"}))
    )
    assert "error" in full


def test_todo_write_and_final_answer_gating():
    handler = _make_handler()

    result = json.loads(
        handler.handle(
            "todo_write",
            json.dumps(
                {
                    "todos": [
                        {"id": "t1", "content": "first", "status": "in_progress"},
                        {"id": "t2", "content": "second", "status": "pending"},
                    ],
                    "merge": False,
                }
            ),
        )
    )
    assert result["success"]
    assert {t.id for t in handler.get_incomplete_todos()} == {"t1", "t2"}

    handler.handle(
        "todo_write",
        json.dumps(
            {
                "todos": [
                    {"id": "t1", "status": "completed"},
                    {"id": "t2", "status": "cancelled"},
                ],
                "merge": True,
            }
        ),
    )
    assert not handler.has_incomplete_todos()


def test_meta_tool_names_and_final_answer_parse():
    assert "todo_write" in META_TOOL_NAMES
    assert FINAL_ANSWER_TOOL["function"]["name"] == "final_answer"
    answer, status = parse_final_answer(
        json.dumps({"answer": "42", "status": "completed"})
    )
    assert (answer, status) == ("42", "completed")
    # Malformed arguments degrade gracefully
    answer, status = parse_final_answer("not json")
    assert (answer, status) == ("not json", "completed")


# ---------------------------------------------------------------------------
# Vendored ReSum / truncation helpers
# ---------------------------------------------------------------------------


def test_resum_safe_cut_never_orphans_tool_messages():
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "r1", "tool_call_id": "1"},
        {"role": "tool", "content": "r2", "tool_call_id": "1"},
        {"role": "assistant", "content": "done"},
    ]
    # A naive cut of the last 3 would start on a tool message; the safe cut
    # walks back to the assistant message that owns the tool results.
    cut = _find_safe_cut_index(messages, target_keep=3)
    assert messages[cut]["role"] != "tool"
    assert cut == 1


def test_truncate_tool_messages_head_tail():
    big = "x" * 300_000
    messages = [{"role": "tool", "content": big, "tool_call_id": "1", "name": "t"}]
    truncate_tool_messages(messages, model="anthropic/claude-sonnet-5")
    content = messages[0]["content"]
    assert len(content) < len(big)
    assert "characters omitted" in content or content.startswith("Error:")


# ---------------------------------------------------------------------------
# Trajectory output → ATIF conversion
# ---------------------------------------------------------------------------


def test_trajectory_dict_converts_to_atif():
    assistant_msg = {
        "role": "assistant",
        "content": "clicking the button",
        "tool_calls": [
            {
                "id": "call_1",
                "function": {
                    "name": "browser_click",
                    "arguments": json.dumps({"ref": "e12"}),
                },
            }
        ],
    }
    messages = [
        {"role": "system", "content": "you are an agent"},
        {"role": "user", "content": "click the button"},
        assistant_msg,
        {"role": "tool", "content": "clicked", "tool_call_id": "call_1"},
    ]
    msg_meta = {
        id(assistant_msg): {
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            "timestamp": 1780000000.0,
        }
    }
    data = build_trajectory_dict(
        model="anthropic/claude-sonnet-5",
        messages=messages,
        msg_meta=msg_meta,
        usage={"cost_usd": 0.12, "compactions": 1},
        agent_config={"max_steps": 250},
        result={"status": "completed", "final_answer": "done"},
    )

    trajectory = convert_mini_swe_agent_to_atif(data, session_id="s1")

    agent_steps = [s for s in trajectory.steps if s.source == "agent"]
    assert len(agent_steps) == 1
    step = agent_steps[0]
    assert step.tool_calls and step.tool_calls[0].function_name == "browser_click"
    assert step.observation and step.observation.results[0].content == "clicked"
    assert step.metrics and step.metrics.prompt_tokens == 100

    assert trajectory.final_metrics
    assert trajectory.final_metrics.total_prompt_tokens == 100
    assert trajectory.final_metrics.total_completion_tokens == 20
    assert trajectory.final_metrics.total_cost_usd == pytest.approx(0.12)
