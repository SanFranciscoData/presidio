from pathlib import Path

from presidio.agents.installed.codex import Codex
from presidio.agents.installed.gemini_cli import GeminiCli
from presidio.agents.installed.mini_swe_agent import MiniSweAgent
from presidio.agents.installed.opencode import OpenCode


def domains(agent) -> set[str]:
    return set(agent.network_allowlist().domains)


def test_codex_reports_custom_and_default_openai_domains(tmp_path: Path):
    agent = Codex(
        logs_dir=tmp_path,
        model_name="gpt-5.5",
        extra_env={"OPENAI_BASE_URL": "https://endpoint.respan.ai/api/"},
    )

    assert {"api.openai.com", "endpoint.respan.ai"} <= domains(agent)


def test_codex_reports_config_toml_provider_base_url(tmp_path: Path):
    agent = Codex(
        logs_dir=tmp_path,
        model_name="gpt-5.5",
        config_toml='openai_base_url = "https://gateway.example.com/v1"\n',
    )

    assert {"api.openai.com", "gateway.example.com"} <= domains(agent)


def test_gemini_cli_reports_custom_respan_base(tmp_path: Path):
    agent = GeminiCli(
        logs_dir=tmp_path,
        model_name="gemini/gemini-3-flash-preview",
        extra_env={
            "GOOGLE_GEMINI_BASE_URL": "https://endpoint.respan.ai/api/google/vertexai/",
            "GEMINI_API_BASE": "https://endpoint.respan.ai/api/google/vertexai/v1beta",
        },
    )

    assert {".googleapis.com", "endpoint.respan.ai"} <= domains(agent)


def test_gemini_cli_includes_exact_resolvable_hosts(tmp_path: Path):
    """Daytona network limits only accept IPv4 CIDRs, so it drops suffix
    domains like ``.googleapis.com`` and resolves the remaining exact hosts.
    The gemini-cli default must therefore include exact hostnames so a
    non-empty CIDR set is produced instead of blocking all network access."""
    agent = GeminiCli(
        logs_dir=tmp_path,
        model_name="gemini/gemini-3-flash-preview",
    )

    resolved = domains(agent)
    assert {
        "generativelanguage.googleapis.com",
        "oauth2.googleapis.com",
    } <= resolved
    assert any(not d.startswith(".") for d in resolved)


def test_opencode_reports_openrouter_default_domain(tmp_path: Path):
    agent = OpenCode(
        logs_dir=tmp_path,
        model_name="openrouter/moonshotai/kimi-k2.6",
    )

    assert "openrouter.ai" in domains(agent)


def test_opencode_reports_provider_config_base_url(tmp_path: Path):
    agent = OpenCode(
        logs_dir=tmp_path,
        model_name="google/gemini-3.1-pro-preview",
        opencode_config={
            "provider": {
                "google": {
                    "options": {
                        "baseURL": "https://endpoint.respan.ai/api/google/vertexai/v1beta"
                    }
                }
            }
        },
    )

    assert {".googleapis.com", "endpoint.respan.ai"} <= domains(agent)


def test_opencode_ignores_non_url_provider_api_values(tmp_path: Path):
    agent = OpenCode(
        logs_dir=tmp_path,
        model_name="opencode/custom-model",
        opencode_config={"provider": {"opencode": {"api": "openai-compatible"}}},
    )

    assert "openai-compatible" not in domains(agent)


def test_mini_swe_reports_provider_env_base_urls(tmp_path: Path):
    agent = MiniSweAgent(
        logs_dir=tmp_path,
        model_name="gemini/gemini-3-flash-preview",
        extra_env={
            "GEMINI_API_BASE": "https://endpoint.respan.ai/api/google/vertexai/v1beta"
        },
    )

    assert {".googleapis.com", "endpoint.respan.ai"} <= domains(agent)


def test_mini_swe_reports_config_yaml_base_url(tmp_path: Path):
    agent = MiniSweAgent(
        logs_dir=tmp_path,
        model_name="openai/gpt-5.5",
        config_yaml="""
model:
  model_kwargs:
    api_base: https://gateway.example.com/v1
""",
    )

    assert {"api.openai.com", "gateway.example.com"} <= domains(agent)
