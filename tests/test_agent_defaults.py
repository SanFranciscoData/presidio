import json

import pytest
from typer.testing import CliRunner

from presidio.agents.factory import AgentFactory
from presidio.agents.installed.antigravity_cli import AntigravityCli
from presidio.cli.main import app
from presidio.models.trial.config import AgentConfig


@pytest.mark.parametrize(
    "model",
    ["gemini/gemini-3.8-flash", "google/gemini-3.8-flash", "gemini-3.8-flash"],
)
def test_gemini_model_defaults_to_antigravity(model, tmp_path):
    config = AgentConfig(model_name=model)
    assert config.name == "antigravity-cli"
    assert config.model_dump(mode="json")["name"] == "antigravity-cli"
    assert isinstance(
        AgentFactory.create_agent_from_config(config, logs_dir=tmp_path), AntigravityCli
    )


@pytest.mark.parametrize(
    "model", [None, "anthropic/claude-sonnet-4", "vertex_ai/gemini-pro"]
)
def test_non_gemini_defaults_are_unchanged(model):
    assert AgentConfig(model_name=model).name == "oracle"


def test_custom_agent_import_is_not_replaced():
    config = AgentConfig(
        import_path="presidio.agents.installed.codex:Codex",
        model_name="gemini/gemini-3.8-flash",
    )
    assert config.name is None


@pytest.mark.parametrize("name", ["gemini-cli", "antigravity-cli", "oracle", "codex"])
def test_explicit_agent_is_not_replaced(name):
    assert AgentConfig(name=name, model_name="gemini/gemini-3.8-flash").name == name


@pytest.mark.parametrize("name", [None, "gemini-cli", "antigravity-cli", "oracle"])
def test_run_selects_default_only_without_explicit_agent(monkeypatch, name):
    captured = []

    async def preflight(config, **kwargs):
        captured.extend(config.agents)
        raise RuntimeError("stop before running a job")

    monkeypatch.setattr("presidio.cli.jobs.run_preflight", preflight)
    monkeypatch.setattr(
        "presidio.agents.factory.AgentFactory.run_preflight",
        lambda config: None,
        raising=False,
    )
    monkeypatch.setattr(
        "presidio.environments.factory.EnvironmentFactory.run_preflight",
        lambda **kwargs: None,
    )
    args = ["run", "-m", "gemini/gemini-3.8-flash"]
    if name:
        args.extend(["-a", name])
    result = CliRunner().invoke(app, args)
    assert isinstance(result.exception, RuntimeError), result.output
    assert [(agent.name, agent.model_name) for agent in captured] == [
        (name or "antigravity-cli", "gemini/gemini-3.8-flash")
    ]


@pytest.mark.parametrize("name", [None, "gemini-cli"])
def test_job_file_preserves_explicit_agent(monkeypatch, tmp_path, name):
    config_path = tmp_path / "job.json"
    config_path.write_text(
        json.dumps(
            {"agents": [{"name": name, "model_name": "gemini/gemini-3.8-flash"}]}
        )
    )
    captured = []

    async def preflight(config, **kwargs):
        captured.extend(config.agents)
        raise RuntimeError("stop before running a job")

    monkeypatch.setattr("presidio.cli.jobs.run_preflight", preflight)
    monkeypatch.setattr(
        "presidio.agents.factory.AgentFactory.run_preflight",
        lambda config: None,
        raising=False,
    )
    monkeypatch.setattr(
        "presidio.environments.factory.EnvironmentFactory.run_preflight",
        lambda **kwargs: None,
    )
    result = CliRunner().invoke(app, ["run", "-c", str(config_path)])
    assert isinstance(result.exception, RuntimeError), result.output
    assert [agent.name for agent in captured] == [name or "antigravity-cli"]
