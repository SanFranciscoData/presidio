"""Tests for Daytona sandbox creation contracts: labels, resource conversion,
rate-limit handling, orphan cleanup, and auto-stop/auto-delete defaults."""

from __future__ import annotations

import asyncio
import logging
import time
import types
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

import pytest

import presidio.environments.daytona as daytona_environment
from presidio.environments.daytona import (
    DAYTONA_DEFAULT_AUTO_DELETE_INTERVAL_MINS,
    DAYTONA_DEFAULT_AUTO_STOP_INTERVAL_MINS,
    DAYTONA_KEEPALIVE_MIN_INTERVAL_SEC,
    DAYTONA_OWNER_LABEL,
    DAYTONA_RATE_LIMIT_MAX_ATTEMPTS,
    DAYTONA_SESSION_COMMAND_GRACE_SEC,
    DaytonaEnvironment,
    _DaytonaDirect,
    _mb_to_gib_ceil,
    _retry_after_seconds,
)
from presidio.models.task.config import EnvironmentConfig
from presidio.models.trial.paths import TrialPaths

daytona = pytest.importorskip("daytona")
from daytona import DaytonaNotFoundError, DaytonaRateLimitError  # noqa: E402


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


# --- session command polling ------------------------------------------------


def test_session_command_poll_timeout_is_bounded(tmp_path, monkeypatch):
    env = _make_env(tmp_path)
    env._sandbox = types.SimpleNamespace()
    monkeypatch.setattr(
        daytona_environment,
        "DAYTONA_SESSION_COMMAND_DEFAULT_TIMEOUT_SEC",
        0.01,
    )

    async def get_response(session_id, command_id):
        return types.SimpleNamespace(exit_code=None, id=command_id)

    env._get_session_command_with_retry = get_response  # type: ignore[method-assign]
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="session_id=session-1"):
        asyncio.run(
            env._poll_response(
                "session-1",
                "command-1",
                command="mkdir -p /logs",
            )
        )
    assert time.monotonic() - started < 1


def test_session_command_poll_returns_terminal_response(tmp_path, monkeypatch):
    env = _make_env(tmp_path)
    env._sandbox = types.SimpleNamespace()
    responses = iter(
        [
            types.SimpleNamespace(exit_code=None, id="command-1"),
            types.SimpleNamespace(exit_code=0, id="command-1"),
        ]
    )
    logs = types.SimpleNamespace(stdout="out", stderr="err")

    async def get_response(session_id, command_id):
        return next(responses)

    async def get_logs(session_id, command_id):
        return logs

    async def fake_sleep(delay):
        assert delay <= 1

    env._get_session_command_with_retry = get_response  # type: ignore[method-assign]
    env._get_session_command_logs_with_retry = get_logs  # type: ignore[method-assign]
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    result = asyncio.run(
        env._poll_response(
            "session-1",
            "command-1",
            timeout_sec=30,
            command="mkdir -p /logs",
        )
    )

    assert result.stdout == "out"
    assert result.stderr == "err"
    assert result.return_code == 0


def test_explicit_session_timeout_includes_grace():
    assert (
        DaytonaEnvironment._session_command_poll_timeout(600)
        == 600 + DAYTONA_SESSION_COMMAND_GRACE_SEC
    )


def test_daytona_directory_layer_restores_dockerfile_user(tmp_path):
    env = _make_env(tmp_path)
    env._dockerfile_path.write_text("FROM alpine\nUSER user\n")

    image = env._with_daytona_directory_layer(
        daytona.Image.from_dockerfile(env._dockerfile_path),
        runtime_user=env._dockerfile_runtime_user(),
    )

    dockerfile = image.dockerfile()
    assert "USER root" in dockerfile
    # /logs/* is world-writable; /tests and /solution are 0755 owned by the
    # restored run user so the agent cannot tamper with them (anti-cheat).
    assert (
        "RUN mkdir -p /logs /logs/agent /logs/verifier /logs/artifacts "
        "/tests /solution && chmod -R 0777 /logs && chmod 0755 /tests /solution "
        "&& chown user /tests /solution"
    ) in dockerfile
    assert "chmod -R 777 /logs /tests /solution" not in dockerfile
    assert dockerfile.rstrip().endswith("USER user")


def test_daytona_directory_layer_root_image_no_chown(tmp_path):
    # A root-default image gets no USER-restore and no chown: /tests and
    # /solution stay root-owned 0755 (agent runs as root anyway), never 777.
    env = _make_env(tmp_path)
    env._dockerfile_path.write_text("FROM alpine\n")

    image = env._with_daytona_directory_layer(
        daytona.Image.from_dockerfile(env._dockerfile_path),
        runtime_user=env._dockerfile_runtime_user(),
    )

    dockerfile = image.dockerfile()
    assert "chmod 0755 /tests /solution" in dockerfile
    assert "chown" not in dockerfile
    assert "chmod -R 777 /logs /tests /solution" not in dockerfile


def test_provision_directories_skips_existing_directories(tmp_path):
    env = _make_env(tmp_path)
    env._sandbox = types.SimpleNamespace(fs=types.SimpleNamespace())
    calls: list[str] = []

    async def is_dir(path, user=None):
        calls.append(path)
        return True

    env._strategy.is_dir = is_dir  # type: ignore[method-assign]

    asyncio.run(env._provision_directories())

    assert len(calls) == 6


def test_daytona_exec_defaults_cwd_to_filesystem_root(tmp_path):
    env = _make_env(tmp_path)
    calls: list[dict] = []

    async def fake_exec(*args, **kwargs):
        calls.append(kwargs)
        return None

    env._strategy.exec = fake_exec  # type: ignore[method-assign]

    asyncio.run(env.exec("pwd"))

    assert calls[0]["cwd"] == "/"


def test_daytona_exec_prefers_explicit_cwd(tmp_path):
    env = _make_env(tmp_path)
    env.task_env_config = EnvironmentConfig(workdir="/task")
    calls: list[dict] = []

    async def fake_exec(*args, **kwargs):
        calls.append(kwargs)
        return None

    env._strategy.exec = fake_exec  # type: ignore[method-assign]

    asyncio.run(env.exec("pwd", cwd="/explicit"))

    assert calls[0]["cwd"] == "/explicit"


def test_daytona_exec_uses_task_workdir(tmp_path):
    env = _make_env(tmp_path)
    env.task_env_config = EnvironmentConfig(workdir="/task")
    calls: list[dict] = []

    async def fake_exec(*args, **kwargs):
        calls.append(kwargs)
        return None

    env._strategy.exec = fake_exec  # type: ignore[method-assign]

    asyncio.run(env.exec("pwd"))

    assert calls[0]["cwd"] == "/task"


@pytest.mark.parametrize("method_name", ["is_dir", "is_file"])
def test_daytona_direct_missing_path_is_not_found(tmp_path, method_name):
    env = _make_env(tmp_path)
    strategy = _DaytonaDirect(env)

    async def get_file_info(path):
        raise DaytonaNotFoundError("missing path")

    env._sandbox = types.SimpleNamespace(
        fs=types.SimpleNamespace(get_file_info=get_file_info)
    )

    assert asyncio.run(getattr(strategy, method_name)("/missing")) is False


def test_daytona_root_toolbox_keeps_su_wrapper(tmp_path, monkeypatch):
    env = _make_env(tmp_path)

    async def effective_user():
        return (0, "root")

    monkeypatch.setattr(env, "_get_daytona_effective_user", effective_user)

    assert asyncio.run(env._daytona_su_target("user")) == "user"


def test_daytona_non_root_toolbox_skips_su_for_same_user(tmp_path, monkeypatch):
    env = _make_env(tmp_path)

    async def effective_user():
        return (1000, "user")

    monkeypatch.setattr(env, "_get_daytona_effective_user", effective_user)

    assert asyncio.run(env._daytona_su_target("user")) is None
    assert asyncio.run(env._daytona_su_target(1000)) is None


def test_daytona_non_root_toolbox_skips_su_for_different_user(
    tmp_path,
    monkeypatch,
    caplog,
):
    env = _make_env(tmp_path)

    async def effective_user():
        return (1000, "user")

    monkeypatch.setattr(env, "_get_daytona_effective_user", effective_user)

    with caplog.at_level(logging.WARNING):
        assert asyncio.run(env._daytona_su_target("root")) is None
        assert asyncio.run(env._daytona_su_target("root")) is None
    # The warning is deduped per distinct target user (pure noise if repeated).
    warnings = [
        r for r in caplog.records if "without runtime privilege escalation" in r.message
    ]
    assert len(warnings) == 1


# --- sandbox keepalive (defeat auto-stop for long, active phases) -----------


class _StubKeepaliveSandbox:
    def __init__(self) -> None:
        self.id = "ka-sandbox"
        self.refresh_calls = 0
        self.deleted = False

    async def refresh_activity(self) -> None:
        self.refresh_calls += 1

    async def delete(self) -> None:
        self.deleted = True


def test_keepalive_interval_is_fraction_of_autostop(tmp_path):
    env = _make_env(tmp_path, auto_stop_interval_mins=60)
    # 60 min / divisor(4) = 15 min = 900s, well under the 3600s deadline.
    assert env._keepalive_interval_sec() == 900.0


def test_keepalive_interval_respects_floor(tmp_path):
    env = _make_env(tmp_path, auto_stop_interval_mins=1)
    assert env._keepalive_interval_sec() == DAYTONA_KEEPALIVE_MIN_INTERVAL_SEC


def test_keepalive_refreshes_then_stop_cancels(tmp_path, monkeypatch):
    monkeypatch.setattr(
        DaytonaEnvironment, "_keepalive_interval_sec", lambda self: 0.01
    )

    async def scenario() -> _StubKeepaliveSandbox:
        env = _make_env(tmp_path, auto_stop_interval_mins=60)
        sandbox = _StubKeepaliveSandbox()
        env._sandbox = sandbox  # type: ignore[assignment]
        env._start_keepalive()
        assert env._keepalive_task is not None
        await asyncio.sleep(0.05)
        await env._stop_sandbox()
        # _stop_sandbox cancels the keepalive and deletes the sandbox.
        assert env._keepalive_task is None
        return sandbox

    sandbox = asyncio.run(scenario())
    assert sandbox.refresh_calls > 0
    assert sandbox.deleted


def test_keepalive_not_started_when_autostop_disabled(tmp_path):
    env = _make_env(tmp_path, auto_stop_interval_mins=0)
    env._sandbox = _StubKeepaliveSandbox()  # type: ignore[assignment]
    env._start_keepalive()
    # auto-stop disabled => the sandbox never idles out => no keepalive task.
    assert env._keepalive_task is None
