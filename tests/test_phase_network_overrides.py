"""Tests for per-phase ([agent]/[verifier]) network policy overrides."""
from __future__ import annotations

import asyncio
import types

import pytest

from presidio.environments.base import BaseEnvironment
from presidio.environments.capabilities import EnvironmentCapabilities
from presidio.models.environment_type import EnvironmentType
from presidio.models.task.config import (
    AgentConfig,
    EnvironmentConfig,
    NetworkMode,
    StepConfig,
    TaskConfig,
    VerifierConfig,
    VerifierEnvironmentMode,
)
from presidio.trial.trial import Trial


# --- config resolution ----------------------------------------------------


def test_agent_no_override_resolves_to_none():
    assert AgentConfig().resolve_phase_network() is None


def test_verifier_no_override_resolves_to_none():
    assert VerifierConfig().resolve_phase_network() is None


def test_agent_allowlist_override_normalizes_hosts():
    cfg = AgentConfig(
        network_mode=NetworkMode.ALLOWLIST,
        allowed_hosts=["*.openai.com", "api.anthropic.com"],
    )
    assert cfg.resolve_phase_network() == (
        NetworkMode.ALLOWLIST,
        [".openai.com", "api.anthropic.com"],
    )


def test_verifier_public_override_carries_no_hosts():
    cfg = VerifierConfig(network_mode=NetworkMode.PUBLIC)
    assert cfg.resolve_phase_network() == (NetworkMode.PUBLIC, [])


def test_no_network_override_carries_no_hosts():
    cfg = AgentConfig(network_mode=NetworkMode.NO_NETWORK)
    assert cfg.resolve_phase_network() == (NetworkMode.NO_NETWORK, [])


def test_allowed_hosts_validator_rejects_urls():
    with pytest.raises(ValueError, match="bare hostnames"):
        AgentConfig(
            network_mode=NetworkMode.ALLOWLIST,
            allowed_hosts=["https://api.openai.com"],
        )


# --- Trial seam ------------------------------------------------------------


class _FakeEnv:
    """Minimal env that reuses the real idempotent phase-switch logic."""

    # Reuse the production helpers so tests exercise the real behavior.
    enter_phase_network_policy = BaseEnvironment.enter_phase_network_policy

    def __init__(
        self,
        *,
        dynamic: bool,
        baseline: tuple[NetworkMode, list[str]] = (NetworkMode.NO_NETWORK, []),
    ) -> None:
        self.capabilities = EnvironmentCapabilities(
            disable_internet=True,
            network_allowlist=True,
            dynamic_network_policy=dynamic,
        )
        self.calls: list[tuple[NetworkMode, list[str]]] = []
        self._env_type = (
            EnvironmentType.E2B if dynamic else EnvironmentType.DOCKER
        )
        self._baseline = baseline
        # Mirror start(): the baseline is the initially-active policy.
        self._active_network_policy = (baseline[0], tuple(sorted(baseline[1])))

    def type(self) -> EnvironmentType:
        return self._env_type

    def baseline_network_policy(self) -> tuple[NetworkMode, list[str]]:
        return self._baseline[0], list(self._baseline[1])

    async def apply_network_policy(
        self, network_mode: NetworkMode, allowed_hosts: list[str]
    ) -> None:
        self.calls.append((network_mode, allowed_hosts))


def _apply(env: _FakeEnv, override, phase: str = "agent") -> None:
    fake_self = types.SimpleNamespace(_environment=env)
    asyncio.run(Trial._enter_phase_network(fake_self, override, phase=phase))


def test_no_override_is_a_noop_on_static_env():
    env = _FakeEnv(dynamic=False)
    _apply(env, None)
    assert env.calls == []


def test_override_on_dynamic_env_calls_apply_network_policy():
    env = _FakeEnv(dynamic=True)
    _apply(env, (NetworkMode.ALLOWLIST, ["api.anthropic.com"]), phase="verifier")
    assert env.calls == [(NetworkMode.ALLOWLIST, ["api.anthropic.com"])]


def test_override_on_non_dynamic_env_raises():
    env = _FakeEnv(dynamic=False)
    with pytest.raises(ValueError, match="dynamic_network_policy"):
        _apply(env, (NetworkMode.PUBLIC, []), phase="agent")
    assert env.calls == []


def test_phase_with_no_override_reverts_to_baseline():
    # Baseline no-network; agent widens to allowlist; verifier (no override)
    # must revert to the baseline rather than inherit the agent's egress.
    env = _FakeEnv(dynamic=True, baseline=(NetworkMode.NO_NETWORK, []))
    _apply(env, (NetworkMode.ALLOWLIST, ["pypi.org"]), phase="agent")
    _apply(env, None, phase="verifier")
    assert env.calls == [
        (NetworkMode.ALLOWLIST, ["pypi.org"]),
        (NetworkMode.NO_NETWORK, []),
    ]


def test_baseline_phase_when_already_baseline_is_noop():
    # Active policy starts at the baseline; an agent phase that declares no
    # override should not issue a redundant switch.
    env = _FakeEnv(dynamic=True, baseline=(NetworkMode.NO_NETWORK, []))
    _apply(env, None, phase="agent")
    assert env.calls == []


def test_repeated_identical_policy_is_idempotent():
    env = _FakeEnv(dynamic=True, baseline=(NetworkMode.NO_NETWORK, []))
    override = (NetworkMode.ALLOWLIST, ["pypi.org"])
    _apply(env, override, phase="agent")
    _apply(env, override, phase="verifier")
    assert env.calls == [(NetworkMode.ALLOWLIST, ["pypi.org"])]


# --- separate-verifier footgun --------------------------------------------


def test_verifier_phase_override_with_separate_env_rejected():
    with pytest.raises(ValueError, match="verifier.environment"):
        VerifierConfig(
            network_mode=NetworkMode.ALLOWLIST,
            allowed_hosts=["api.anthropic.com"],
            environment_mode=VerifierEnvironmentMode.SEPARATE,
        )


def test_verifier_phase_override_with_env_block_rejected():
    with pytest.raises(ValueError, match="verifier.environment"):
        VerifierConfig(
            network_mode=NetworkMode.PUBLIC,
            environment=EnvironmentConfig(),
        )


def test_verifier_phase_override_shared_is_allowed():
    cfg = VerifierConfig(
        network_mode=NetworkMode.ALLOWLIST,
        allowed_hosts=["api.anthropic.com"],
    )
    assert cfg.resolve_phase_network() == (
        NetworkMode.ALLOWLIST,
        ["api.anthropic.com"],
    )


# --- load-time capability validation --------------------------------------


def _validate(env: _FakeEnv, config: TaskConfig) -> None:
    fake_self = types.SimpleNamespace(
        _environment=env,
        _task=types.SimpleNamespace(config=config, has_steps=bool(config.steps)),
    )
    Trial._validate_phase_network_capability(fake_self)


def test_agent_override_on_static_env_rejected_at_load():
    env = _FakeEnv(dynamic=False)
    config = TaskConfig(
        agent=AgentConfig(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=["pypi.org"])
    )
    with pytest.raises(ValueError, match=r"\[agent\].*dynamic_network_policy"):
        _validate(env, config)


def test_shared_verifier_override_on_static_env_rejected_at_load():
    env = _FakeEnv(dynamic=False)
    config = TaskConfig(
        verifier=VerifierConfig(
            network_mode=NetworkMode.ALLOWLIST, allowed_hosts=["api.anthropic.com"]
        )
    )
    with pytest.raises(ValueError, match=r"\[verifier\].*dynamic_network_policy"):
        _validate(env, config)


def test_step_agent_override_on_static_env_names_the_step():
    env = _FakeEnv(dynamic=False)
    config = TaskConfig(
        steps=[
            StepConfig(
                name="build",
                agent=AgentConfig(
                    network_mode=NetworkMode.ALLOWLIST, allowed_hosts=["pypi.org"]
                ),
            )
        ]
    )
    with pytest.raises(ValueError, match=r"step 'build' agent"):
        _validate(env, config)


def test_no_overrides_on_static_env_passes():
    env = _FakeEnv(dynamic=False)
    _validate(env, TaskConfig())


def test_overrides_on_dynamic_env_pass():
    env = _FakeEnv(dynamic=True)
    config = TaskConfig(
        agent=AgentConfig(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=["pypi.org"])
    )
    _validate(env, config)
