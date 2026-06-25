"""Unit tests for the E2B environment's native network policy mapping.

The E2B SDK is driven through a stub sandbox so these tests exercise the
policy-to-SDK translation (and the dynamic phase-override seam) without a live
E2B account.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path

import pytest

from presidio.environments.capabilities import EnvironmentCapabilities
from presidio.environments.e2b import (
    E2B_DEFAULT_EXEC_TIMEOUT_SEC,
    E2B_DEFAULT_SANDBOX_TIMEOUT_SEC,
    E2B_MAX_SANDBOX_LIFETIME_SEC,
    _KEEPALIVE_WINDOW_SEC,
    E2bEnvironment,
    _is_template_alias_conflict,
    network_allowlist_to_e2b_hosts,
)
from presidio.models.agent.install import AgentInstallSpec, InstallStep
from presidio.models.agent.network import NetworkAllowlist
from presidio.models.environment_type import EnvironmentType
from presidio.models.task.config import NetworkMode


def _env(mode: NetworkMode, domains: list[str] | None = None) -> E2bEnvironment:
    """An E2bEnvironment with network state injected, bypassing __init__/SDK."""
    env = E2bEnvironment.__new__(E2bEnvironment)
    env._network_mode = mode
    env.network_allowlist = NetworkAllowlist(domains=domains or [])
    env.logger = logging.getLogger("test-e2b")
    env._sandbox = None
    env._active_network_policy = None
    return env


class _StubSandbox:
    def __init__(self) -> None:
        self.update_network_calls: list[dict] = []

    async def update_network(self, payload: dict) -> None:
        self.update_network_calls.append(payload)


def test_capabilities_declare_full_native_model():
    caps = E2bEnvironment.capabilities.fget(_env(NetworkMode.PUBLIC))
    assert caps == EnvironmentCapabilities(
        disable_internet=True,
        network_allowlist=True,
        dynamic_network_policy=True,
        preinstall_agents=True,
    )


def test_type_is_e2b():
    assert E2bEnvironment.type() == EnvironmentType.E2B


def test_wildcard_conversion_to_e2b_glob():
    # Leading-dot entries match squid's dstdomain semantics (apex + subdomains),
    # so each expands to both the apex and the E2B subdomain glob. Bare hosts
    # pass through unchanged.
    assert network_allowlist_to_e2b_hosts(
        [".example.com", "api.openai.com", "  ", ".bar.io"]
    ) == [
        "*.bar.io",
        "*.example.com",
        "api.openai.com",
        "bar.io",
        "example.com",
    ]


def test_public_maps_to_allow_internet_access():
    env = _env(NetworkMode.PUBLIC)
    assert env._create_network_kwargs() == {"allow_internet_access": True}


def test_no_network_maps_to_disabled_internet():
    env = _env(NetworkMode.NO_NETWORK)
    assert env._create_network_kwargs() == {"allow_internet_access": False}


def test_allowlist_maps_to_deny_all_plus_allow_out():
    env = _env(NetworkMode.ALLOWLIST, ["api.anthropic.com", ".openai.com"])
    assert env._create_network_kwargs() == {
        "network": {
            "allow_out": ["*.openai.com", "api.anthropic.com", "openai.com"],
            "deny_out": ["0.0.0.0/0"],
        }
    }


def test_update_payload_for_each_mode():
    env = _env(NetworkMode.PUBLIC)
    assert env._network_update_payload(NetworkMode.PUBLIC, []) == {
        "allow_internet_access": True
    }
    assert env._network_update_payload(NetworkMode.NO_NETWORK, []) == {
        "allow_internet_access": False
    }
    assert env._network_update_payload(
        NetworkMode.ALLOWLIST, ["api.anthropic.com", ".openai.com"]
    ) == {
        "allow_out": ["*.openai.com", "api.anthropic.com", "openai.com"],
        "deny_out": ["0.0.0.0/0"],
    }


def test_apply_network_policy_calls_update_network():
    env = _env(NetworkMode.NO_NETWORK)
    sandbox = _StubSandbox()
    env._sandbox = sandbox

    asyncio.run(env.apply_network_policy(NetworkMode.ALLOWLIST, ["api.anthropic.com"]))

    assert sandbox.update_network_calls == [
        {"allow_out": ["api.anthropic.com"], "deny_out": ["0.0.0.0/0"]}
    ]


def test_apply_network_policy_without_sandbox_raises():
    env = _env(NetworkMode.NO_NETWORK)
    with pytest.raises(RuntimeError, match="before the sandbox is started"):
        asyncio.run(env.apply_network_policy(NetworkMode.PUBLIC, []))


def test_enter_phase_policy_is_idempotent_and_reverts_to_baseline():
    # Baseline no-network (as start() would seed it).
    env = _env(NetworkMode.NO_NETWORK)
    env._active_network_policy = (NetworkMode.NO_NETWORK, ())
    sandbox = _StubSandbox()
    env._sandbox = sandbox

    async def scenario() -> None:
        # Re-entering the baseline issues no update.
        await env.enter_phase_network_policy(NetworkMode.NO_NETWORK, [])
        # Agent widens to an allowlist -> one update.
        await env.enter_phase_network_policy(
            NetworkMode.ALLOWLIST, ["api.anthropic.com"]
        )
        # Same policy again -> no redundant update.
        await env.enter_phase_network_policy(
            NetworkMode.ALLOWLIST, ["api.anthropic.com"]
        )
        # Verifier phase reverts to the baseline -> one update.
        await env.enter_phase_network_policy(*env.baseline_network_policy())

    asyncio.run(scenario())

    assert sandbox.update_network_calls == [
        {"allow_out": ["api.anthropic.com"], "deny_out": ["0.0.0.0/0"]},
        {"allow_internet_access": False},
    ]


def test_baseline_network_policy_reports_mode_and_hosts():
    env = _env(NetworkMode.ALLOWLIST, [".openai.com", "api.anthropic.com"])
    assert env.baseline_network_policy() == (
        NetworkMode.ALLOWLIST,
        [".openai.com", "api.anthropic.com"],
    )


class _StubFiles:
    def __init__(self, read_chunks: list[bytes] | None = None) -> None:
        self.written: dict[str, bytes] = {}
        self._read_chunks = read_chunks or []

    async def write(self, path: str, data) -> None:
        # Mirror the SDK contract: a file-like object is streamed, not buffered.
        self.written[path] = data.read()

    async def read(self, path: str, format: str = "text"):
        assert format == "stream"

        async def _gen():
            for chunk in self._read_chunks:
                yield chunk

        return _gen()


class _StubFileSandbox:
    def __init__(self, read_chunks: list[bytes] | None = None) -> None:
        self.files = _StubFiles(read_chunks)


def test_upload_file_streams_handle_to_sdk(tmp_path: Path):
    env = _env(NetworkMode.PUBLIC)
    sandbox = _StubFileSandbox()
    env._sandbox = sandbox
    src = tmp_path / "patch.diff"
    src.write_bytes(b"hello world")

    # target parent is "/", so no mkdir exec is triggered.
    asyncio.run(env.upload_file(src, "/patch.diff"))

    assert sandbox.files.written == {"/patch.diff": b"hello world"}


def test_download_file_streams_chunks_to_disk(tmp_path: Path):
    env = _env(NetworkMode.PUBLIC)
    sandbox = _StubFileSandbox(read_chunks=[b"chunk-1;", b"chunk-2;", b"chunk-3"])
    env._sandbox = sandbox
    dest = tmp_path / "nested" / "out.bin"

    asyncio.run(env.download_file("/remote/out.bin", dest))

    assert dest.read_bytes() == b"chunk-1;chunk-2;chunk-3"


# --- sandbox lifetime / keepalive -----------------------------------------


class _StubLifetimeSandbox:
    def __init__(self) -> None:
        self.set_timeout_calls: list[int] = []
        self.killed = False

    async def set_timeout(self, timeout: int) -> None:
        self.set_timeout_calls.append(timeout)

    async def kill(self) -> None:
        self.killed = True


def _lifetime_env(
    *,
    min_lifetime: float | None,
    sandbox_timeout: int = E2B_DEFAULT_SANDBOX_TIMEOUT_SEC,
) -> E2bEnvironment:
    env = _env(NetworkMode.PUBLIC)
    env._sandbox_timeout_sec = sandbox_timeout
    env._min_lifetime_sec = min_lifetime
    env._keepalive_task = None
    return env


def test_provider_max_lifetime_defaults_to_24h():
    assert _lifetime_env(min_lifetime=None).provider_max_lifetime_sec() == float(
        E2B_MAX_SANDBOX_LIFETIME_SEC
    )


def test_provider_max_lifetime_honors_env_override(monkeypatch):
    monkeypatch.setenv("E2B_MAX_SANDBOX_LIFETIME_SEC", "3600")
    assert _lifetime_env(min_lifetime=None).provider_max_lifetime_sec() == 3600.0


def test_initial_timeout_uses_floor_when_budget_unknown():
    env = _lifetime_env(min_lifetime=None)
    assert env._initial_timeout_sec() == max(
        E2B_DEFAULT_SANDBOX_TIMEOUT_SEC, _KEEPALIVE_WINDOW_SEC
    )


def test_initial_timeout_grows_with_budget():
    env = _lifetime_env(min_lifetime=5000)
    assert env._initial_timeout_sec() == 5000 + _KEEPALIVE_WINDOW_SEC


def test_initial_timeout_caps_at_provider_max():
    env = _lifetime_env(min_lifetime=10**9)
    assert env._initial_timeout_sec() == E2B_MAX_SANDBOX_LIFETIME_SEC


def test_default_exec_timeout_uses_budget_when_set():
    assert _lifetime_env(min_lifetime=1234)._default_exec_timeout_sec() == 1234


def test_default_exec_timeout_falls_back_to_constant():
    assert (
        _lifetime_env(min_lifetime=None)._default_exec_timeout_sec()
        == E2B_DEFAULT_EXEC_TIMEOUT_SEC
    )


def test_keepalive_extends_ttl_then_stop_cancels(monkeypatch):
    monkeypatch.setattr("presidio.environments.e2b._KEEPALIVE_INTERVAL_SEC", 0.01)

    async def scenario() -> _StubLifetimeSandbox:
        env = _lifetime_env(min_lifetime=None)
        sandbox = _StubLifetimeSandbox()
        env._sandbox = sandbox
        env._keepalive_task = asyncio.create_task(env._keepalive_loop())
        await asyncio.sleep(0.05)
        await env.stop(delete=False)
        # stop cancels the keepalive and kills the sandbox.
        assert env._keepalive_task is None
        assert sandbox.killed
        return sandbox

    sandbox = asyncio.run(scenario())
    assert sandbox.set_timeout_calls
    assert all(call == _KEEPALIVE_WINDOW_SEC for call in sandbox.set_timeout_calls)


class _StubTaskEnvConfig:
    """Minimal stand-in: ``_resolve_template`` only reads ``docker_image``."""

    def __init__(self, docker_image: str | None = None) -> None:
        self.docker_image = docker_image


def _install_spec(agent_name: str = "codex") -> AgentInstallSpec:
    return AgentInstallSpec(
        agent_name=agent_name,
        steps=[InstallStep(run="npm install -g @openai/codex", user="agent")],
    )


def _template_env(
    task_dir: Path,
    docker_image: str | None = None,
    agent_install_spec: AgentInstallSpec | None = None,
) -> E2bEnvironment:
    env = E2bEnvironment.__new__(E2bEnvironment)
    env.logger = logging.getLogger("test-e2b")
    env.task_env_config = _StubTaskEnvConfig(docker_image=docker_image)
    env.environment_dir = task_dir
    env.agent_install_spec = agent_install_spec
    env.default_user = None
    return env


def test_resolve_template_prefers_declared_docker_image(tmp_path):
    # A pre-declared template always wins; a present Dockerfile is never built
    # over it (preserves every existing prebuilt-template task shape).
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\n")
    env = _template_env(tmp_path, docker_image="my-prebuilt-template")
    assert (
        asyncio.run(env._resolve_template(force_build=False)) == "my-prebuilt-template"
    )


def test_resolve_template_none_without_dockerfile_or_image(tmp_path):
    # Neither a declared image nor a Dockerfile: fall back to E2B's bare default
    # template (None), exactly as before.
    env = _template_env(tmp_path, docker_image=None)
    assert asyncio.run(env._resolve_template(force_build=False)) is None


def test_resolve_template_builds_from_dockerfile(tmp_path, monkeypatch):
    # The only new behavior: no declared image + a Dockerfile present builds an
    # ephemeral template from it and boots that.
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\nWORKDIR /app\n")
    calls: dict = {}

    class _StubTemplate:
        def __init__(self, file_context_path=None, **kwargs):
            calls["file_context_path"] = file_context_path

        def from_dockerfile(self, content_or_path):
            calls["from_dockerfile"] = content_or_path
            return self

        @staticmethod
        async def build(builder, *, name, skip_cache, on_build_logs):
            calls["build"] = {"name": name, "skip_cache": skip_cache}

    monkeypatch.setattr("presidio.environments.e2b.AsyncTemplate", _StubTemplate)
    env = _template_env(tmp_path, docker_image=None)

    name = asyncio.run(env._resolve_template(force_build=True))
    assert name.startswith("presidio-")
    assert calls["build"]["name"] == name  # the built template is the one booted
    assert calls["build"]["skip_cache"] is True  # force_build maps to skip_cache
    assert calls["file_context_path"] == tmp_path  # COPY resolves against the context


def _stub_async_template(monkeypatch) -> dict:
    """Patch ``AsyncTemplate`` and capture the build inputs.

    ``run_cmd`` calls are recorded (command, user) and their base64 payloads
    are decoded so tests can assert on the install scripts that get baked in.
    """
    calls: dict = {"run_cmds": []}

    class _StubTemplate:
        def __init__(self, file_context_path=None, **kwargs):
            calls["file_context_path"] = file_context_path

        def from_dockerfile(self, content_or_path):
            calls["from_dockerfile"] = content_or_path
            return self

        def run_cmd(self, command, user=None):
            decoded = command
            if command.startswith("echo "):
                payload = command.split(" ", 2)[1]
                decoded = base64.b64decode(payload).decode()
            calls["run_cmds"].append({"user": user, "script": decoded})
            return self

        @staticmethod
        async def build(builder, *, name, skip_cache, on_build_logs):
            calls["build"] = {"name": name, "skip_cache": skip_cache}

    monkeypatch.setattr("presidio.environments.e2b.AsyncTemplate", _StubTemplate)
    return calls


def test_resolve_template_bakes_agent_install_via_run_cmd(tmp_path, monkeypatch):
    # An agent_install_spec is baked into the built template (build-time bake,
    # full network) via run_cmd -- agent-agnostic: the spec's own step is
    # base64-piped to bash so arbitrary quoting survives.
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\nWORKDIR /app\n")
    calls = _stub_async_template(monkeypatch)
    env = _template_env(tmp_path, agent_install_spec=_install_spec())

    name = asyncio.run(env._resolve_template(force_build=False))
    assert calls["from_dockerfile"].startswith("FROM python:3.12-slim")  # task base
    assert calls["run_cmds"] == [
        {"user": "user", "script": "npm install -g @openai/codex"}
    ]  # the agent's own step, baked in as e2b's default sandbox user
    assert calls["build"]["name"] == name
    assert calls["file_context_path"] == tmp_path  # COPY still resolves against context


def test_resolve_template_bakes_agent_onto_prebuilt_image(tmp_path, monkeypatch):
    # docker_image + an agent to install: extend the image via FROM and bake.
    calls = _stub_async_template(monkeypatch)
    env = _template_env(
        tmp_path, docker_image="my-base:latest", agent_install_spec=_install_spec()
    )

    asyncio.run(env._resolve_template(force_build=False))
    assert calls["from_dockerfile"] == "FROM my-base:latest\n"
    assert calls["run_cmds"][0]["script"] == "npm install -g @openai/codex"


def test_resolve_template_root_step_runs_as_root(tmp_path, monkeypatch):
    # A root install step is baked as root; a non-root step uses a configured
    # agent user when one is set (parity with the agent's runtime user).
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\n")
    calls = _stub_async_template(monkeypatch)
    spec = AgentInstallSpec(
        agent_name="codex",
        steps=[
            InstallStep(run="apt-get install -y curl", user="root"),
            InstallStep(run="npm install -g @openai/codex", user="agent"),
        ],
    )
    env = _template_env(tmp_path, agent_install_spec=spec)
    env.default_user = "agent"

    asyncio.run(env._resolve_template(force_build=False))
    assert [(c["user"], c["script"]) for c in calls["run_cmds"]] == [
        ("root", "apt-get install -y curl"),
        ("agent", "npm install -g @openai/codex"),
    ]


def test_resolve_template_inlines_step_env(tmp_path, monkeypatch):
    # Per-step env is inlined as exports ahead of the script (shared helper).
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\n")
    calls = _stub_async_template(monkeypatch)
    spec = AgentInstallSpec(
        agent_name="codex",
        steps=[InstallStep(run="do-install", user="root", env={"FOO": "bar"})],
    )
    env = _template_env(tmp_path, agent_install_spec=spec)

    asyncio.run(env._resolve_template(force_build=False))
    assert calls["run_cmds"][0]["script"] == "export FOO=bar; do-install"


def test_resolve_template_builds_plain_dockerfile_without_run_cmds(
    tmp_path, monkeypatch
):
    # No agent: build straight from the task Dockerfile, no install layers.
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\nWORKDIR /app\n")
    calls = _stub_async_template(monkeypatch)
    env = _template_env(tmp_path)

    asyncio.run(env._resolve_template(force_build=False))
    assert calls["from_dockerfile"].startswith("FROM python:3.12-slim")
    assert calls["run_cmds"] == []


def test_resolve_template_prefers_prebuilt_image_when_no_agent(tmp_path):
    # docker_image with no agent: boot it directly, never build.
    env = _template_env(tmp_path, docker_image="my-prebuilt-template")
    assert (
        asyncio.run(env._resolve_template(force_build=False)) == "my-prebuilt-template"
    )


def test_resolve_template_raises_when_agent_has_nothing_to_bake_into(tmp_path):
    # No Dockerfile and no prebuilt image: an agent install cannot be baked.
    env = _template_env(tmp_path, agent_install_spec=_install_spec("claude-code"))
    with pytest.raises(ValueError, match="claude-code"):
        asyncio.run(env._resolve_template(force_build=False))


def test_ephemeral_template_name_keys_on_agent_and_image(tmp_path):
    # Two agents must not collide on one cached template name.
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\n")
    base = _template_env(tmp_path)._ephemeral_template_name()
    codex = _template_env(
        tmp_path, agent_install_spec=_install_spec("codex")
    )._ephemeral_template_name()
    claude = _template_env(
        tmp_path, agent_install_spec=_install_spec("claude-code")
    )._ephemeral_template_name()
    assert len({base, codex, claude}) == 3


def _stub_async_template_raising(monkeypatch, exc: BaseException) -> dict:
    """Patch ``AsyncTemplate`` so ``build`` raises ``exc`` (to drive the
    duplicate-alias reuse path); record that build was attempted."""
    calls: dict = {"build_attempts": 0}

    class _StubTemplate:
        def __init__(self, file_context_path=None, **kwargs):
            pass

        def from_dockerfile(self, content_or_path):
            return self

        @staticmethod
        async def build(builder, *, name, skip_cache, on_build_logs):
            calls["build_attempts"] += 1
            raise exc

    monkeypatch.setattr("presidio.environments.e2b.AsyncTemplate", _StubTemplate)
    return calls


def test_is_template_alias_conflict_matches_provider_signatures():
    # The real e2b message and its constituent signatures all count as a conflict.
    real = (
        "500: Error when inserting alias 'presidio-ea13d699': ERROR: duplicate "
        "key value violates unique constraint "
        '"idx_env_aliases_alias_namespace_unique" (SQLSTATE 23505)'
    )
    assert _is_template_alias_conflict(Exception(real))
    assert _is_template_alias_conflict(Exception("duplicate key value violates"))
    assert _is_template_alias_conflict(Exception("boom (SQLSTATE 23505)"))
    assert _is_template_alias_conflict(Exception("alias already exists"))


def test_is_template_alias_conflict_rejects_unrelated_errors():
    assert not _is_template_alias_conflict(Exception("network unreachable"))
    assert not _is_template_alias_conflict(Exception("build failed: exit 127"))


def test_resolve_template_reuses_template_on_alias_conflict(tmp_path, monkeypatch):
    # A concurrent/previous build already registered the alias -> the duplicate-key
    # failure is swallowed and the named template is reused (no trial failure).
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\n")
    conflict = Exception(
        'duplicate key value violates unique constraint "idx_env_aliases" '
        "(SQLSTATE 23505)"
    )
    calls = _stub_async_template_raising(monkeypatch, conflict)
    env = _template_env(tmp_path)

    name = asyncio.run(env._resolve_template(force_build=False))
    assert name.startswith("presidio-")  # resolved to the (now-existing) template
    assert calls["build_attempts"] == 1


def test_resolve_template_reraises_unrelated_build_error(tmp_path, monkeypatch):
    # A genuine build failure (not an alias conflict) must NOT be swallowed.
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\n")
    boom = RuntimeError("build failed: base image not found")
    _stub_async_template_raising(monkeypatch, boom)
    env = _template_env(tmp_path)

    with pytest.raises(RuntimeError, match="base image not found"):
        asyncio.run(env._resolve_template(force_build=False))


def test_ephemeral_template_name_is_deterministic_and_content_sensitive(tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\n")
    env = _template_env(tmp_path)
    first = env._ephemeral_template_name()
    assert (
        first == env._ephemeral_template_name()
    )  # stable -> trials reuse the template
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\nRUN echo changed\n")
    assert env._ephemeral_template_name() != first  # content change -> new name
