from types import SimpleNamespace

import toml

from presidio.agents.installed.claude_code import ClaudeCode
from presidio.agents.installed.codex import Codex
from presidio.agents.installed.gemini_cli import GeminiCli


def _env(allow_internet: bool, network_mode: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        task_env_config=SimpleNamespace(
            allow_internet=allow_internet, network_mode=network_mode
        )
    )


def test_should_disable_web_tools_follows_task_network_policy():
    for agent in (Codex, ClaudeCode, GeminiCli):
        assert agent._should_disable_web_tools(
            _env(allow_internet=True, network_mode="allowlist")
        )
        assert (
            agent._should_disable_web_tools(
                _env(allow_internet=True, network_mode="public")
            )
            is False
        )


def test_explicit_web_tools_suppression_overrides_public_network():
    environment = _env(allow_internet=True, network_mode="public")

    for agent in (Codex, ClaudeCode, GeminiCli):
        assert agent._should_disable_web_tools(environment, disable_web_tools=True)


def test_agent_kwarg_plumbs_explicit_web_tools_suppression(tmp_path):
    for agent in (Codex, ClaudeCode, GeminiCli):
        instance = agent(logs_dir=tmp_path, disable_web_tools=True)
        assert instance._disable_web_tools


def test_codex_disable_web_search_overrides_user_config():
    user = 'web_search = "live"\n[features]\nweb_search_request = true\nmodel = "x"\n'
    parsed = toml.loads(Codex._disable_web_search_config_toml(user))

    assert parsed["web_search"] == "disabled"
    assert parsed["features"]["web_search_request"] is False
    assert parsed["features"]["web_search_cached"] is False
    assert parsed["features"]["model"] == "x"


def test_codex_disable_web_search_with_no_user_config():
    parsed = toml.loads(Codex._disable_web_search_config_toml(None))

    assert parsed["web_search"] == "disabled"
    assert parsed["features"]["web_search_request"] is False
    assert parsed["features"]["web_search_cached"] is False


def test_codex_disable_web_search_preserves_top_level_user_keys():
    user = 'model = "gpt-5.6-sol"\nsandbox_mode = "danger-full-access"\n'
    parsed = toml.loads(Codex._disable_web_search_config_toml(user))

    assert parsed["model"] == "gpt-5.6-sol"
    assert parsed["sandbox_mode"] == "danger-full-access"
    assert parsed["web_search"] == "disabled"


def test_gemini_settings_exclude_web_tools_when_disabled(tmp_path):
    agent = GeminiCli(logs_dir=tmp_path, model_name="gemini-3.7-flash")

    config, _ = agent._build_settings_config(disable_web_tools=True)

    assert config is not None
    assert config["tools"] == {"exclude": ["google_web_search", "web_fetch"]}


def test_gemini_settings_unchanged_when_internet_allowed(tmp_path):
    agent = GeminiCli(logs_dir=tmp_path, model_name="gemini-3.7-flash")

    config, _ = agent._build_settings_config(disable_web_tools=False)

    assert config is None or "tools" not in config
