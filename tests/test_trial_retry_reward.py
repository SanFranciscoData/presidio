"""Retry policy: a trial that produced a verifier reward is authoritative and
must not be retried, even when the agent process exited non-zero.

gemini-cli (and other agents) can exit non-zero *after* reaching a fully
gradeable state — a late tool error, a flaky shutdown. Presidio's Trial.run
catches ``NonZeroAgentExitCodeError`` and still runs the verifier, so such a
trial records a real reward. Retrying it re-runs a full, expensive agent episode
only to recompute a reward we already have (tripling wall-clock on the common
case). Only genuinely signal-less trials (a crash before any gradeable state, no
reward recorded) should be retried.
"""

import asyncio
from types import SimpleNamespace

import pytest

from presidio.models.job.config import RetryConfig
from presidio.models.trial.result import ExceptionInfo
from presidio.models.verifier.result import VerifierResult
from presidio.trial.queue import TrialQueue


def _exc(kind: str = "NonZeroAgentExitCodeError") -> ExceptionInfo:
    return ExceptionInfo(
        exception_type=kind,
        exception_message="boom",
        exception_traceback="tb",
        occurred_at=__import__("datetime").datetime.now(),
    )


class _FakeTrial:
    """Minimal stand-in for presidio.trial.trial.Trial used by the queue."""

    def __init__(self, result, trial_dir):
        self._result = result
        self.trial_dir = trial_dir

    def add_hook(self, event, callback):  # queue._setup_hooks wires these
        return self

    async def run(self):
        return self._result


def _install_fake_trials(monkeypatch, tmp_path, results):
    """Patch Trial.create to hand out fake trials returning `results` in order."""
    seq = iter(results)
    created = {"n": 0}

    async def _create(_config):
        created["n"] += 1
        return _FakeTrial(next(seq), tmp_path / f"trial_{created['n']}")

    import presidio.trial.trial as trial_mod

    monkeypatch.setattr(trial_mod, "Trial", SimpleNamespace(create=_create))
    return created


def _run(queue, config=None):
    if config is None:
        config = SimpleNamespace(trial_name="t")
    return asyncio.run(queue._execute_trial_with_retries(config))


def test_reward_bearing_error_trial_is_not_retried(monkeypatch, tmp_path):
    # Agent exited non-zero but the verifier still graded (reward present).
    # The reward is authoritative -> return immediately, no retry.
    graded = SimpleNamespace(
        exception_info=_exc(),
        verifier_result=VerifierResult(rewards={"reward": 0.0}),
    )
    # A second result would only be consumed if a retry (wrongly) happened.
    sentinel = SimpleNamespace(exception_info=None, verifier_result=None)
    created = _install_fake_trials(monkeypatch, tmp_path, [graded, sentinel])

    queue = TrialQueue(n_concurrent=1, retry_config=RetryConfig(max_retries=2))
    result = _run(queue)

    assert result is graded  # returned the graded trial
    assert created["n"] == 1  # exactly one attempt — no retry


def test_full_reward_error_trial_is_not_retried(monkeypatch, tmp_path):
    # Same, but the agent actually solved it (reward 1.0) before exiting non-zero.
    graded = SimpleNamespace(
        exception_info=_exc(),
        verifier_result=VerifierResult(rewards={"reward": 1.0}),
    )
    created = _install_fake_trials(monkeypatch, tmp_path, [graded, graded])
    queue = TrialQueue(n_concurrent=1, retry_config=RetryConfig(max_retries=2))
    result = _run(queue)
    assert result is graded and created["n"] == 1


def test_signal_less_crash_is_still_retried(monkeypatch, tmp_path):
    # No verifier reward recorded (crash before any gradeable state) -> retry,
    # and the retry that produces a reward is returned.
    crashed = SimpleNamespace(exception_info=_exc(), verifier_result=None)
    recovered = SimpleNamespace(
        exception_info=None,
        verifier_result=VerifierResult(rewards={"reward": 1.0}),
    )
    created = _install_fake_trials(monkeypatch, tmp_path, [crashed, recovered])
    queue = TrialQueue(n_concurrent=1, retry_config=RetryConfig(max_retries=2))
    result = _run(queue)
    assert result is recovered and created["n"] == 2


def test_empty_reward_dict_is_treated_as_no_signal(monkeypatch, tmp_path):
    # verifier_result present but rewards is empty/None -> no signal -> retry.
    empty = SimpleNamespace(
        exception_info=_exc(), verifier_result=VerifierResult(rewards={})
    )
    recovered = SimpleNamespace(
        exception_info=None,
        verifier_result=VerifierResult(rewards={"reward": 0.0}),
    )
    created = _install_fake_trials(monkeypatch, tmp_path, [empty, recovered])
    queue = TrialQueue(n_concurrent=1, retry_config=RetryConfig(max_retries=2))
    result = _run(queue)
    assert result is recovered and created["n"] == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
