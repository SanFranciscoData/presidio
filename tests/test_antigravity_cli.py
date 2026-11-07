import asyncio
import json
import os
import shlex
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from presidio.agents.factory import AgentFactory
from presidio.agents.installed.antigravity_cli import AntigravityCli
from presidio.agents.installed.gemini_cli import GeminiCli
from presidio.errors import InvalidModelError, MissingCredentialError
from presidio.models.agent.context import AgentContext
from presidio.models.task.config import MCPServerConfig
from presidio.models.trial.config import AgentConfig


def make_agent(tmp_path, **kwargs):
    return AntigravityCli(
        logs_dir=tmp_path, model_name="gemini/gemini-3.8-flash", **kwargs
    )


def test_explicit_agent_configuration_does_not_replace_gemini_cli(tmp_path):
    for name, agent_type in [
        ("antigravity-cli", AntigravityCli),
        ("gemini-cli", GeminiCli),
    ]:
        config = AgentConfig(name=name, model_name="gemini/gemini-3.8-flash")
        assert isinstance(
            AgentFactory.create_agent_from_config(config, logs_dir=tmp_path), agent_type
        )


def test_install_spec_uses_native_official_installer(tmp_path):
    spec = make_agent(tmp_path).install_spec()
    assert spec.agent_name == "antigravity-cli"
    assert "https://antigravity.google/cli/install.sh" in spec.steps[1].run
    assert "pipefail" in spec.steps[1].run
    assert "npm" not in spec.steps[1].run
    assert "agy" in spec.verification_command


def test_version_pinning_is_not_silently_ignored(tmp_path):
    with pytest.raises(ValueError, match="latest"):
        make_agent(tmp_path, version="1.2.3")


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_effort_is_explicitly_passed_to_agy(tmp_path, effort):
    command = make_agent(tmp_path, reasoning_effort=effort)._build_run_command(
        "gemini-3.8-flash"
    )
    assert f"--effort {effort}" in command


def test_unsupported_effort_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="reasoning_effort"):
        make_agent(tmp_path, reasoning_effort="unlimited")


def test_headless_env_supports_google_api_key_alias(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("AGY_ADC_AUTH", raising=False)
    monkeypatch.setenv("AGY_CLI_TEST", "host")
    agent = make_agent(
        tmp_path,
        extra_env={
            "GOOGLE_API_KEY": "test-key",
            "GOOGLE_GEMINI_BASE_URL": "https://gateway.example.com",
            "AGY_CLI_TEST": "job",
        },
    )
    env = agent._runtime_env()
    assert env["GEMINI_API_KEY"] == "test-key"
    assert env["GOOGLE_GEMINI_BASE_URL"] == "https://gateway.example.com"
    assert env["AGY_CLI_TEST"] == "job"
    assert {
        "generativelanguage.googleapis.com",
        "oauth2.googleapis.com",
        "gateway.example.com",
    } <= set(agent.network_allowlist().domains)


def test_job_api_key_alias_takes_priority_over_host_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "host-key")
    monkeypatch.delenv("AGY_ADC_AUTH", raising=False)
    agent = make_agent(tmp_path, extra_env={"GOOGLE_API_KEY": "job-key"})
    assert agent._runtime_env()["GEMINI_API_KEY"] == "job-key"


def test_missing_credentials_fail_without_starting_interactive_auth(
    tmp_path, monkeypatch
):
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "AGY_ADC_AUTH"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(MissingCredentialError, match="GEMINI_API_KEY"):
        make_agent(tmp_path)._runtime_env()


def test_adc_cannot_silently_change_authentication_mode(tmp_path):
    with pytest.raises(ValueError, match="API-key auth only"):
        make_agent(tmp_path, extra_env={"AGY_ADC_AUTH": "true"})._runtime_env()


def test_settings_match_agy_native_paths_and_do_not_contain_credentials(tmp_path):
    servers = [
        MCPServerConfig(
            name="stdio", transport="stdio", command="python", args=["server.py"]
        ),
        MCPServerConfig(
            name="http", transport="streamable-http", url="https://mcp.example.com"
        ),
    ]
    agent = make_agent(
        tmp_path, mcp_servers=servers, extra_env={"GEMINI_API_KEY": "not-in-settings"}
    )
    command = agent._settings_command("gemini-custom")
    completed = subprocess.run(
        ["bash", "-c", command], env={**os.environ, "HOME": str(tmp_path)}, check=True
    )
    assert completed.returncode == 0
    settings = json.loads(
        (tmp_path / ".gemini/antigravity-cli/settings.json").read_text()
    )
    mcp = json.loads((tmp_path / ".gemini/config/mcp_config.json").read_text())
    assert settings["modelProvider"] == "gemini"
    assert (
        settings["customModelsConfig"]["customModels"]["gemini-custom"]["modelName"]
        == "gemini-custom"
    )
    assert mcp["mcpServers"]["http"]["serverUrl"] == "https://mcp.example.com"
    assert mcp["mcpServers"]["stdio"]["args"] == ["server.py"]
    assert "not-in-settings" not in command


@pytest.mark.parametrize("known", [True, False])
def test_run_preserves_model_and_uploads_prompt_instead_of_inlining(
    tmp_path, monkeypatch, known
):
    agent = make_agent(
        tmp_path, extra_env={"GEMINI_API_KEY": "test-key", "AGY_ADC_AUTH": "false"}
    )
    agent.exec_as_agent = AsyncMock(
        return_value=SimpleNamespace(
            stdout="gemini-3.8-flash\tGemini Flash\n" if known else ""
        )
    )
    agent.exec_as_root = AsyncMock()
    environment = SimpleNamespace(upload_file=AsyncMock())
    instruction = "don't expand $(echo bad); " * 10000
    asyncio.run(agent.run(instruction, environment, AgentContext()))
    calls = agent.exec_as_agent.call_args_list
    commands = [call.kwargs["command"] for call in calls]
    assert all(instruction not in command for command in commands)
    assert commands[-1].count("--model gemini-3.8-flash") == 1
    assert "--add-dir" in commands[-1]
    assert "</dev/null" in commands[-1]
    assert (any("customModelsConfig" in command for command in commands)) is not known
    assert environment.upload_file.await_count == 1
    assert calls[1].kwargs["timeout_sec"] == 15


def test_model_discovery_failure_still_registers_exact_requested_model(tmp_path):
    agent = make_agent(
        tmp_path, extra_env={"GEMINI_API_KEY": "test-key", "AGY_ADC_AUTH": "false"}
    )
    agent.exec_as_agent = AsyncMock(side_effect=[None, TimeoutError(), None, None])
    agent.exec_as_root = AsyncMock()
    environment = SimpleNamespace(upload_file=AsyncMock())
    asyncio.run(agent.run("hello", environment, AgentContext()))
    assert "gemini-3.8-flash" in agent.exec_as_agent.call_args_list[2].kwargs["command"]


def test_wrong_provider_is_not_silently_substituted(tmp_path):
    agent = AntigravityCli(logs_dir=tmp_path, model_name="vertex_ai/gemini-3.8-flash")
    with pytest.raises(InvalidModelError, match="Gemini API"):
        asyncio.run(agent.run("hello", SimpleNamespace(), AgentContext()))


def test_pipeline_preserves_failure_and_keeps_stderr_out_of_jsonl(tmp_path):
    binary = tmp_path / ".local/bin"
    binary.mkdir(parents=True)
    agy = binary / "agy"
    agy.write_text('#!/bin/bash\necho \'{"event":"init"}\'\necho "fatal" >&2\nexit 7\n')
    agy.chmod(0o755)
    prompt = tmp_path / "prompt"
    prompt.write_text("hello")
    command = make_agent(tmp_path)._build_run_command("gemini-3.8-flash")
    command = command.replace("/logs/agent", shlex.quote(str(tmp_path))).replace(
        AntigravityCli._PROMPT_PATH, str(prompt)
    )
    result = subprocess.run(
        ["bash", "-c", "set -o pipefail; " + command],
        capture_output=True,
        text=True,
        env={**os.environ, "HOME": str(tmp_path)},
    )
    assert result.returncode == 7
    assert "fatal" in result.stderr
    assert (
        json.loads((tmp_path / AntigravityCli._OUTPUT_FILENAME).read_text())["event"]
        == "init"
    )


def step_event(index, kind, **kwargs):
    return {
        "event": "step_update",
        "step_update": {
            "step_index": index,
            "step_type": kind,
            "state": "DONE",
            **kwargs,
        },
    }


def test_stream_conversion_merges_deltas_tools_and_authoritative_totals(tmp_path):
    agent = make_agent(tmp_path)
    agent._instruction = "Write hello"
    events = [
        {"event": "init", "conversation_id": "session-1"},
        step_event(0, "user_input"),
        step_event(1, "agent_response", text_delta="I'll ", state="ACTIVE"),
        step_event(
            1,
            "agent_response",
            text_delta="do it.",
            usage={
                "input_tokens": 100,
                "output_tokens": 10,
                "cache_read_tokens": 50,
                "thinking_tokens": 8,
            },
        ),
        step_event(
            2,
            "tool",
            tool_name="run_command",
            tool_info={
                "name": "run_command",
                "parameters": {"CommandLine": "echo hello"},
                "output": "hello\n",
            },
        ),
        step_event(3, "checkpoint", usage={"input_tokens": 5, "output_tokens": 2}),
        {
            "event": "result",
            "result": {
                "conversation_id": "session-1",
                "status": "SUCCESS",
                "response": "I'll do it.",
                "usage": {
                    "input_tokens": 105,
                    "output_tokens": 12,
                    "cache_read_tokens": 50,
                },
            },
        },
    ]
    trajectory = agent._convert_events(events)
    assert trajectory.agent.name == "antigravity-cli"
    assert trajectory.session_id == "session-1"
    assert [s.message for s in trajectory.steps] == ["Write hello", "I'll do it.", ""]
    tool = trajectory.steps[-1]
    assert tool.tool_calls[0].arguments == {"CommandLine": "echo hello"}
    assert tool.observation.results[0].source_call_id == tool.tool_calls[0].tool_call_id
    assert tool.observation.results[0].content == "hello\n"
    assert trajectory.final_metrics.total_prompt_tokens == 155
    assert trajectory.final_metrics.total_completion_tokens == 12
    assert trajectory.final_metrics.total_cached_tokens == 50
    assert trajectory.final_metrics.extra["peak_context_tokens"] == 150
    assert trajectory.final_metrics.total_cost_usd > 0


def test_partial_stream_is_retained_after_failure(tmp_path):
    agent = make_agent(tmp_path)
    events = [
        step_event(
            1,
            "agent_response",
            text_delta="working",
            usage={"input_tokens": 100, "output_tokens": 10},
        )
    ]
    path = tmp_path / agent._OUTPUT_FILENAME
    path.write_text("diagnostic\n[]\n" + json.dumps(events[0]) + '\n{"event":')
    context = AgentContext()
    agent.populate_context_post_run(context)
    assert context.n_input_tokens == 100
    assert context.n_output_tokens == 10
    assert (
        json.loads((tmp_path / "trajectory.json").read_text())["steps"][0]["message"]
        == "working"
    )


def test_missing_stream_does_not_invent_a_trajectory(tmp_path):
    assert make_agent(tmp_path).convert_trajectory(tmp_path) is None
