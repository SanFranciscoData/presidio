"""Provider/agent-agnostic per-command timeout budget propagation.

An installed agent runs its CLI as a single long-running ``environment.exec``.
If the caller does not pin a per-command timeout, the agent must fall back to
its whole-run execution budget so no backend's short-command default (e.g. a
sandbox session-command default) can guillotine the run mid-solve. See
``BaseInstalledAgent._exec`` and ``TrialExecution._create_agent``.
"""

import asyncio
from pathlib import Path

from presidio.agents.factory import AgentFactory
from presidio.agents.installed.cursor_cli import CursorCli
from presidio.models.agent.name import AgentName


class _RecordingEnv:
    """Minimal environment double capturing the ``timeout_sec`` per exec."""

    def __init__(self):
        self.calls: list[int | None] = []

    def agent_process_env(self, env):
        return env or {}

    async def exec(self, command, user=None, env=None, cwd=None, timeout_sec=None):
        self.calls.append(timeout_sec)

        class _R:
            return_code = 0
            stdout = ""
            stderr = ""

        return _R()


def _agent(tmp_path: Path, budget: float | None) -> CursorCli:
    return CursorCli(
        logs_dir=tmp_path,
        model_name="cursor/composer-2.5",
        agent_timeout_sec=budget,
    )


def test_exec_defaults_to_agent_budget(tmp_path: Path):
    env = _RecordingEnv()
    agent = _agent(tmp_path, budget=14400.0)
    asyncio.run(agent.exec_as_agent(env, command="claude --print -- hi"))
    assert env.calls == [14400]


def test_explicit_timeout_wins_over_budget(tmp_path: Path):
    env = _RecordingEnv()
    agent = _agent(tmp_path, budget=14400.0)
    asyncio.run(agent.exec_as_agent(env, command="quick", timeout_sec=30))
    assert env.calls == [30]


def test_no_budget_preserves_backend_default(tmp_path: Path):
    env = _RecordingEnv()
    agent = _agent(tmp_path, budget=None)
    asyncio.run(agent.exec_as_agent(env, command="claude --print -- hi"))
    # ``None`` is forwarded so each backend keeps its own legacy default.
    assert env.calls == [None]


def test_budget_is_threaded_through_factory(tmp_path: Path):
    agent = AgentFactory.create_agent_from_name(
        AgentName.CURSOR_CLI,
        logs_dir=tmp_path,
        model_name="cursor/composer-2.5",
        agent_timeout_sec=1234.0,
    )
    assert agent._agent_timeout_sec == 1234.0
