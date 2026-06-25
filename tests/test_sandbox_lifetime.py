"""Tests for task.toml-driven sandbox lifetime (provider TTL non-blocking).

The harness (``asyncio.wait_for`` around each phase) is the single source of
truth for when a run ends. These tests cover the budget the trial derives from
task.toml timeouts, the fail-fast check against a provider's hard lifetime cap,
and Modal's create-time TTL sizing. (E2B's keepalive/sizing live in
``test_e2b_environment.py``.)
"""
from __future__ import annotations

import types

import pytest

from presidio.environments.modal import _MODAL_LIFETIME_MARGIN_SEC, ModalEnvironment
from presidio.models.task.config import (
    TaskConfig,
    VerifierConfig,
    VerifierEnvironmentMode,
)
from presidio.trial.trial import Trial


def _trial(
    *,
    config: TaskConfig,
    has_steps: bool,
    agent_timeout: float | None,
    setup_timeout: float = 360.0,
    verifier_timeout: float = 600.0,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        _task=types.SimpleNamespace(config=config, has_steps=has_steps),
        _execution=types.SimpleNamespace(
            agent_setup_timeout_sec=setup_timeout,
            agent_timeout_sec=agent_timeout,
        ),
        _verifier_timeout_sec=verifier_timeout,
    )


# --- lifetime budget -------------------------------------------------------


def test_budget_sums_setup_agent_and_shared_verifier():
    fake = _trial(
        config=TaskConfig(),
        has_steps=False,
        agent_timeout=600.0,
        setup_timeout=360.0,
        verifier_timeout=600.0,
    )
    # shared verifier runs in the primary env -> included.
    assert Trial._environment_lifetime_budget(fake) == 360.0 + 600.0 + 600.0


def test_budget_excludes_separate_verifier():
    config = TaskConfig(
        verifier=VerifierConfig(environment_mode=VerifierEnvironmentMode.SEPARATE)
    )
    fake = _trial(
        config=config,
        has_steps=False,
        agent_timeout=600.0,
        setup_timeout=360.0,
        verifier_timeout=600.0,
    )
    # separate verifier carries its own env/budget -> not in the primary budget.
    assert Trial._environment_lifetime_budget(fake) == 360.0 + 600.0


def test_budget_is_none_when_agent_unbounded():
    fake = _trial(config=TaskConfig(), has_steps=False, agent_timeout=None)
    assert Trial._environment_lifetime_budget(fake) is None


# --- fail-fast against a provider's hard cap -------------------------------


def _env_with_cap(cap: float | None) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        provider_max_lifetime_sec=lambda: cap,
        type=lambda: types.SimpleNamespace(value="e2b"),
    )


def test_validate_rejects_budget_over_provider_cap():
    with pytest.raises(ValueError, match="exceeds the maximum sandbox lifetime"):
        Trial._validate_lifetime_budget(_env_with_cap(3600.0), 7200.0)


def test_validate_passes_budget_under_cap():
    Trial._validate_lifetime_budget(_env_with_cap(3600.0), 1800.0)


def test_validate_ignores_unbounded_budget():
    # None budget (unbounded agent) cannot be compared to a cap; never rejected.
    Trial._validate_lifetime_budget(_env_with_cap(3600.0), None)


def test_validate_passes_when_no_provider_cap():
    Trial._validate_lifetime_budget(_env_with_cap(None), 10**9)


# --- modal create-time sizing ----------------------------------------------


def _modal_env(*, min_lifetime: float | None, configured: int) -> ModalEnvironment:
    env = ModalEnvironment.__new__(ModalEnvironment)
    env._sandbox_timeout = configured
    env._min_lifetime_sec = min_lifetime
    return env


def test_modal_keeps_configured_ttl_when_budget_unknown():
    env = _modal_env(min_lifetime=None, configured=86400)
    assert env._effective_sandbox_timeout() == 86400


def test_modal_keeps_configured_ttl_when_budget_smaller():
    env = _modal_env(min_lifetime=1000, configured=86400)
    assert env._effective_sandbox_timeout() == 86400


def test_modal_grows_ttl_when_budget_exceeds_configured():
    env = _modal_env(min_lifetime=100000, configured=86400)
    assert env._effective_sandbox_timeout() == 100000 + _MODAL_LIFETIME_MARGIN_SEC
