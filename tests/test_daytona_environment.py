"""Tests for Daytona sandbox creation contracts: labels, resource conversion,
rate-limit handling, orphan cleanup, and auto-stop/auto-delete defaults."""

from __future__ import annotations

import asyncio
import logging
import types
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

import pytest

from presidio.environments.daytona import (
    DAYTONA_DEFAULT_AUTO_DELETE_INTERVAL_MINS,
    DAYTONA_DEFAULT_AUTO_STOP_INTERVAL_MINS,
    DAYTONA_OWNER_LABEL,
    DAYTONA_RATE_LIMIT_MAX_ATTEMPTS,
    DaytonaEnvironment,
    _mb_to_gib_ceil,
    _retry_after_seconds,
)
from presidio.models.task.config import EnvironmentConfig
from presidio.models.trial.paths import TrialPaths

daytona = pytest.importorskip("daytona")
from daytona import DaytonaRateLimitError  # noqa: E402


def _make_env(tmp_path: Path, **kwargs) -> DaytonaEnvironment:
    env_dir = tmp_path / "environment"
    env_dir.mkdir(exist_ok=True)
    (env_dir / "Dockerfile").write_text("FROM alpine\n")
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir(exist_ok=True)
    return DaytonaEnvironment(
        environment_dir=env_dir,
        environment_name="test-env",
        session_id="test-session",
        trial_paths=TrialPaths(trial_dir=trial_dir),
        task_env_config=EnvironmentConfig(),
        **kwargs,
    )


# --- MiB -> GiB conversion (never underprovision) --------------------------


@pytest.mark.parametrize(
    ("mb", "gib"),
    [
        (1, 1),
        (1023, 1),
        (1024, 1),
        (1025, 2),
        (2048, 2),
        (10240, 10),
        (10241, 11),
    ],
)
def test_mb_to_gib_rounds_up(mb: int, gib: int):
    assert _mb_to_gib_ceil(mb) == gib


def test_sandbox_resources_round_up(tmp_path):
    env = _make_env(tmp_path)
    env.task_env_config = EnvironmentConfig(
        cpus=2, memory_mb=1536, storage_mb=10241
    )
    resources = env._sandbox_resources()
    assert resources is not None
    assert resources.cpu == 2
    assert resources.memory == 2
    assert resources.disk == 11


# --- labels -----------------------------------------------------------------


def test_caller_sandbox_labels_are_included(tmp_path):
    env = _make_env(tmp_path, sandbox_labels={"team": "swe-farm", "run": "42"})
    labels = env._sandbox_labels()
    assert labels["team"] == "swe-farm"
    assert labels["run"] == "42"
    assert labels[DAYTONA_OWNER_LABEL] == env._owner_token


def test_labels_kwarg_alias_is_supported(tmp_path):
    env = _make_env(tmp_path, labels={"team": "swe-farm"})
    assert env._sandbox_labels()["team"] == "swe-farm"


def test_caller_labels_cannot_override_ownership_label(tmp_path, caplog):
    env = _make_env(tmp_path, sandbox_labels={DAYTONA_OWNER_LABEL: "spoofed"})
    with caplog.at_level(logging.WARNING):
        labels = env._sandbox_labels()
    assert labels[DAYTONA_OWNER_LABEL] == env._owner_token
    assert any("ownership" in record.message for record in caplog.records)


def test_label_values_are_coerced_to_strings(tmp_path):
    env = _make_env(tmp_path, sandbox_labels={"attempt": 3})
    assert env._sandbox_labels()["attempt"] == "3"


def test_base_sandbox_params_carry_labels_and_intervals(tmp_path):
    env = _make_env(tmp_path, sandbox_labels={"team": "swe-farm"})
    params = env._base_sandbox_params()
    assert params["labels"]["team"] == "swe-farm"
    assert params["labels"][DAYTONA_OWNER_LABEL] == env._owner_token
    assert params["auto_stop_interval"] == DAYTONA_DEFAULT_AUTO_STOP_INTERVAL_MINS
    assert params["auto_delete_interval"] == DAYTONA_DEFAULT_AUTO_DELETE_INTERVAL_MINS


# --- auto-stop / auto-delete defaults ---------------------------------------


def test_default_auto_intervals_are_safe(tmp_path):
    env = _make_env(tmp_path)
    assert env._auto_stop_interval == DAYTONA_DEFAULT_AUTO_STOP_INTERVAL_MINS
    assert env._auto_stop_interval > 0
    assert env._auto_delete_interval == DAYTONA_DEFAULT_AUTO_DELETE_INTERVAL_MINS
    assert env._auto_delete_interval >= 0


def test_explicit_auto_intervals_are_honored(tmp_path):
    env = _make_env(tmp_path, auto_stop_interval_mins=0, auto_delete_interval_mins=30)
    assert env._auto_stop_interval == 0
    assert env._auto_delete_interval == 30


def test_fully_disabled_lifecycle_warns(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        env = _make_env(
            tmp_path, auto_stop_interval_mins=0, auto_delete_interval_mins=-1
        )
    assert env._auto_stop_interval == 0
    assert env._auto_delete_interval == -1
    assert any("auto-stop" in record.message for record in caplog.records)


# --- Retry-After parsing -----------------------------------------------------


def test_retry_after_numeric_seconds():
    exc = DaytonaRateLimitError("429", status_code=429, headers={"Retry-After": "7"})
    assert _retry_after_seconds(exc) == 7.0


def test_retry_after_is_case_insensitive():
    exc = DaytonaRateLimitError("429", status_code=429, headers={"retry-after": "3"})
    assert _retry_after_seconds(exc) == 3.0


def test_retry_after_http_date():
    retry_at = datetime.now(timezone.utc) + timedelta(seconds=60)
    exc = DaytonaRateLimitError(
        "429", status_code=429, headers={"Retry-After": format_datetime(retry_at)}
    )
    delay = _retry_after_seconds(exc)
    assert delay is not None
    assert 50.0 <= delay <= 60.0


def test_retry_after_missing_or_invalid():
    assert _retry_after_seconds(DaytonaRateLimitError("429", status_code=429)) is None
    exc = DaytonaRateLimitError(
        "429", status_code=429, headers={"Retry-After": "soon"}
    )
    assert _retry_after_seconds(exc) is None


# --- create retry behavior ---------------------------------------------------


def _stub_env_for_create() -> DaytonaEnvironment:
    env = DaytonaEnvironment.__new__(DaytonaEnvironment)
    env.logger = logging.getLogger("test-daytona")
    env._sandbox = None
    env._client_manager = None
    return env


def test_rate_limit_retry_honors_retry_after(monkeypatch):
    env = _stub_env_for_create()
    sleeps: list[float] = []
    attempts = 0

    async def fake_sleep(delay):
        sleeps.append(delay)

    async def fake_create_once(params):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise DaytonaRateLimitError(
                "429", status_code=429, headers={"Retry-After": "11"}
            )

    async def fake_cleanup():
        pass

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    env._create_sandbox_once = fake_create_once
    env._cleanup_orphan_sandboxes = fake_cleanup

    asyncio.run(DaytonaEnvironment._create_sandbox(env, params=None))

    assert attempts == 3
    assert sleeps == [11.0, 11.0]


def test_rate_limit_retry_gives_up_after_max_attempts(monkeypatch):
    env = _stub_env_for_create()

    async def fake_sleep(delay):
        pass

    async def fake_create_once(params):
        raise DaytonaRateLimitError("429", status_code=429)

    async def fake_cleanup():
        pass

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    env._create_sandbox_once = fake_create_once
    env._cleanup_orphan_sandboxes = fake_cleanup

    with pytest.raises(DaytonaRateLimitError):
        asyncio.run(DaytonaEnvironment._create_sandbox(env, params=None))


def test_generic_error_retries_once_with_cleanup(monkeypatch):
    env = _stub_env_for_create()
    cleanups = 0
    attempts = 0

    async def fake_sleep(delay):
        pass

    async def fake_create_once(params):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("boom")

    async def fake_cleanup():
        nonlocal cleanups
        cleanups += 1

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    env._create_sandbox_once = fake_create_once
    env._cleanup_orphan_sandboxes = fake_cleanup

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(DaytonaEnvironment._create_sandbox(env, params=None))

    assert attempts == 2
    assert cleanups == 1


def test_rate_limit_max_attempts_is_bounded():
    assert 2 <= DAYTONA_RATE_LIMIT_MAX_ATTEMPTS <= 10


# --- orphan cleanup ----------------------------------------------------------


def test_cleanup_orphans_deletes_only_owned_sandboxes(tmp_path):
    env = _make_env(tmp_path)
    deleted: list[str] = []

    class FakeSandbox:
        def __init__(self, sandbox_id: str):
            self.id = sandbox_id

        async def delete(self):
            deleted.append(self.id)

    current = FakeSandbox("current")
    env._sandbox = current  # type: ignore[assignment]
    listed_labels: list[dict] = []

    class FakeClient:
        async def list(self, labels=None):
            listed_labels.append(labels)
            return types.SimpleNamespace(
                items=[current, FakeSandbox("orphan-1"), FakeSandbox("orphan-2")]
            )

    class FakeManager:
        async def get_client(self):
            return FakeClient()

    env._client_manager = FakeManager()  # type: ignore[assignment]

    asyncio.run(env._cleanup_orphan_sandboxes())

    assert deleted == ["orphan-1", "orphan-2"]
    assert listed_labels == [{DAYTONA_OWNER_LABEL: env._owner_token}]
