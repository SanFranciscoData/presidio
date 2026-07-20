import asyncio
import logging
import socket
from unittest.mock import AsyncMock, call

import pytest

from presidio.environments.daytona import (
    DAYTONA_MAX_NETWORK_ALLOWLIST_CIDRS,
    DaytonaClientManager,
    DaytonaEnvironment,
    _DaytonaDirect,
    _DaytonaDinD,
    resolve_network_allowlist_to_daytona_cidrs,
)
from presidio.models.agent.network import NetworkAllowlist
from presidio.models.task.config import TaskOS
from presidio.models.trial.paths import EnvironmentPaths


def _addr(ip: str):
    return (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))


def test_daytona_resolves_domains_to_ipv4_cidrs(monkeypatch):
    def fake_getaddrinfo(host, *_args, **_kwargs):
        assert host == "api.openai.com"
        return [_addr("203.0.113.10"), _addr("203.0.113.10")]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    resolution, cidrs = resolve_network_allowlist_to_daytona_cidrs(
        ["api.openai.com", ".anthropic.com"]
    )

    assert resolution == {"api.openai.com": ["203.0.113.10"]}
    assert cidrs == ["203.0.113.10/32"]


def test_daytona_collapses_resolved_cidrs_to_daytona_limit(monkeypatch):
    def fake_getaddrinfo(_host, *_args, **_kwargs):
        return [_addr(f"203.0.113.{i}") for i in range(1, 18)]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    _resolution, cidrs = resolve_network_allowlist_to_daytona_cidrs(["api.openai.com"])

    assert len(cidrs) <= DAYTONA_MAX_NETWORK_ALLOWLIST_CIDRS
    assert all(cidr.endswith(("/32", "/31", "/30", "/29", "/28")) for cidr in cidrs)


def test_daytona_network_params_use_resolved_allowlist(monkeypatch):
    env = DaytonaEnvironment.__new__(DaytonaEnvironment)
    env._explicit_network_allow_list = None
    env._explicit_network_block_all = None
    env._resolved_network_allow_list = None
    env._network_resolution_debug = {}
    env.task_env_config = type("TaskEnv", (), {"allow_internet": False})()
    env.network_allowlist = NetworkAllowlist(domains=["api.openai.com"])
    env.logger = logging.getLogger("test")

    monkeypatch.setattr(
        "presidio.environments.daytona.resolve_network_allowlist_to_daytona_cidrs",
        lambda domains: (
            {"api.openai.com": ["203.0.113.10"]},
            ["203.0.113.10/32"],
        ),
    )

    assert DaytonaEnvironment._network_params(env) == {
        "network_block_all": False,
        "network_allow_list": "203.0.113.10/32",
    }


def test_daytona_network_params_block_when_no_cidrs(monkeypatch):
    env = DaytonaEnvironment.__new__(DaytonaEnvironment)
    env._explicit_network_allow_list = None
    env._explicit_network_block_all = None
    env._resolved_network_allow_list = None
    env._network_resolution_debug = {}
    env.task_env_config = type("TaskEnv", (), {"allow_internet": False})()
    env.network_allowlist = NetworkAllowlist(domains=[".anthropic.com"])
    env.logger = logging.getLogger("test")

    monkeypatch.setattr(
        "presidio.environments.daytona.resolve_network_allowlist_to_daytona_cidrs",
        lambda domains: ({}, []),
    )

    assert DaytonaEnvironment._network_params(env) == {"network_block_all": True}


def test_daytona_compose_keeps_main_network_when_sandbox_allowlist_is_active():
    env = DaytonaEnvironment.__new__(DaytonaEnvironment)
    env.environment_name = "task"
    env.session_id = "task.1"
    env._persistent_env = {}
    env.task_env_config = type(
        "TaskEnv",
        (),
        {
            "allow_internet": False,
            "env": {},
            "cpus": 1,
            "memory_mb": 1024,
            "docker_image": None,
        },
    )()
    env._compose_should_block_main_network = lambda: False

    strategy = _DaytonaDinD(env)

    assert not any("no-network" in flag for flag in strategy._compose_file_flags())


def test_daytona_compose_does_not_advertise_agent_preinstall():
    env = DaytonaEnvironment.__new__(DaytonaEnvironment)
    env._compose_mode = True

    assert env.capabilities.network_allowlist is True
    assert env.capabilities.preinstall_agents is False


def test_daytona_direct_resets_directories_as_sandbox_user():
    env = DaytonaEnvironment.__new__(DaytonaEnvironment)
    env._compose_mode = False

    assert env._reset_dirs_user() is None


def test_daytona_compose_resets_directories_as_root():
    env = DaytonaEnvironment.__new__(DaytonaEnvironment)
    env._compose_mode = True
    env.task_env_config = type("TaskEnv", (), {"os": TaskOS.LINUX})()

    assert env._reset_dirs_user() == "root"


def _reset_test_env(*, compose_mode: bool, default_user: str | None):
    env = DaytonaEnvironment.__new__(DaytonaEnvironment)
    env._compose_mode = compose_mode
    env._persistent_env = {}
    env.default_user = default_user
    env.task_env_config = type(
        "TaskEnv",
        (),
        {"os": TaskOS.LINUX, "workdir": None},
    )()
    result = type("Result", (), {"return_code": 0, "stdout": "", "stderr": ""})()
    env._strategy = type(
        "Strategy",
        (),
        {"exec": AsyncMock(return_value=result)},
    )()
    return env


def test_daytona_direct_reset_bypasses_default_user():
    env = _reset_test_env(compose_mode=False, default_user="root")

    asyncio.run(
        env.reset_dirs(
            remove_dirs=[EnvironmentPaths.agent_dir],
            create_dirs=[EnvironmentPaths.agent_dir],
            chmod_dirs=[EnvironmentPaths.agent_dir],
        )
    )

    assert env._strategy.exec.await_args.kwargs["user"] is None


def test_daytona_direct_reset_preserves_root_directory_entries():
    env = _reset_test_env(compose_mode=False, default_user="root")

    asyncio.run(
        env.reset_dirs(
            remove_dirs=[EnvironmentPaths.tests_dir],
            create_dirs=[EnvironmentPaths.tests_dir],
            chmod_dirs=[EnvironmentPaths.tests_dir],
        )
    )

    command = env._strategy.exec.await_args.args[0]
    assert "find /tests -mindepth 1 -maxdepth 1" in command
    assert "rm -rf /tests" not in command


def test_daytona_exec_without_user_uses_default_user():
    env = _reset_test_env(compose_mode=False, default_user="root")

    asyncio.run(env.exec("echo ready"))

    assert env._strategy.exec.await_args.kwargs["user"] == "root"


def test_daytona_compose_reset_keeps_root_user():
    env = _reset_test_env(compose_mode=True, default_user="agent")

    asyncio.run(
        env.reset_dirs(
            remove_dirs=[EnvironmentPaths.agent_dir],
            create_dirs=[EnvironmentPaths.agent_dir],
            chmod_dirs=[EnvironmentPaths.agent_dir],
        )
    )

    assert env._strategy.exec.await_args.kwargs["user"] == "root"


def test_daytona_direct_provisions_canonical_directories():
    env = DaytonaEnvironment.__new__(DaytonaEnvironment)
    create_folder = AsyncMock()
    set_file_permissions = AsyncMock()
    env._sandbox = type(
        "Sandbox",
        (),
        {
            "fs": type(
                "FileSystem",
                (),
                {
                    "create_folder": create_folder,
                    "set_file_permissions": set_file_permissions,
                },
            )()
        },
    )()

    asyncio.run(env._provision_directories())

    # /logs/* world-writable; /tests and /solution 0755 (anti-cheat: the agent
    # must not be able to overwrite the verifier tests or oracle solution).
    assert create_folder.await_args_list == [
        call(str(EnvironmentPaths.logs_dir), "777"),
        call(str(EnvironmentPaths.agent_dir), "777"),
        call(str(EnvironmentPaths.verifier_dir), "777"),
        call(str(EnvironmentPaths.artifacts_dir), "777"),
        call(str(EnvironmentPaths.tests_dir), "755"),
        call(str(EnvironmentPaths.solution_dir), "755"),
    ]
    assert set_file_permissions.await_args_list == [
        call(str(EnvironmentPaths.logs_dir), mode="777"),
        call(str(EnvironmentPaths.agent_dir), mode="777"),
        call(str(EnvironmentPaths.verifier_dir), mode="777"),
        call(str(EnvironmentPaths.artifacts_dir), mode="777"),
        call(str(EnvironmentPaths.tests_dir), mode="755"),
        call(str(EnvironmentPaths.solution_dir), mode="755"),
    ]


def test_daytona_direct_directory_provisioning_fails_closed():
    env = DaytonaEnvironment.__new__(DaytonaEnvironment)
    create_folder = AsyncMock(side_effect=RuntimeError("permission denied"))
    env._sandbox = type(
        "Sandbox",
        (),
        {
            "fs": type(
                "FileSystem",
                (),
                {
                    "create_folder": create_folder,
                    "set_file_permissions": AsyncMock(),
                },
            )()
        },
    )()

    with pytest.raises(RuntimeError, match="permission denied"):
        asyncio.run(env._provision_directories())


def test_daytona_direct_start_uses_toolbox_directory_provisioning(monkeypatch):
    env = DaytonaEnvironment.__new__(DaytonaEnvironment)
    env._snapshot_template_name = None
    env._auto_delete_interval = 0
    env._auto_stop_interval = 0
    env.agent_install_spec = None
    env.task_env_config = type(
        "TaskEnv",
        (),
        {"docker_image": "example/image:tag"},
    )()
    env._sandbox_resources = lambda: None
    env._configure_daytona_client = AsyncMock()
    env._base_sandbox_params = lambda: {}
    env._with_agent_install = lambda image: image
    env._create_sandbox = AsyncMock()
    env._provision_directories = AsyncMock()
    env._pin_resolved_hosts = AsyncMock()
    env._sandbox_exec = AsyncMock()
    manager = type(
        "Manager",
        (),
        {"get_client": AsyncMock(return_value=object())},
    )()
    monkeypatch.setattr(
        DaytonaClientManager,
        "get_instance",
        AsyncMock(return_value=manager),
    )

    asyncio.run(_DaytonaDirect(env).start(force_build=False))

    env._create_sandbox.assert_awaited_once()
    env._provision_directories.assert_awaited_once_with()
    env._pin_resolved_hosts.assert_awaited_once_with()
    env._sandbox_exec.assert_not_awaited()


def test_daytona_pins_resolved_hosts():
    env = DaytonaEnvironment.__new__(DaytonaEnvironment)
    env._compose_mode = False
    env._network_resolution_debug = {
        "domain_resolution": {"api.example.com": ["203.0.113.10"]}
    }
    captured = {}

    async def sandbox_exec(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return type("Result", (), {"return_code": 0, "stdout": "", "stderr": ""})()

    env._sandbox_exec = sandbox_exec

    asyncio.run(DaytonaEnvironment._pin_resolved_hosts(env))

    assert "203.0.113.10 api.example.com" in captured["command"]
    assert captured["kwargs"]["shell"] == "bash -c"


def test_daytona_pins_all_resolved_ips_for_failover():
    # A domain backed by a rotating frontend pool (e.g. the Gemini API) must pin
    # EVERY resolved IP, not just the first, so the agent can fail over across
    # frontends instead of funneling all traffic through one (which triggers
    # connection resets under many concurrent sandboxes).
    env = DaytonaEnvironment.__new__(DaytonaEnvironment)
    env._compose_mode = False
    ips = ["216.239.32.223", "216.239.34.223", "216.239.36.223", "216.239.38.223"]
    env._network_resolution_debug = {
        "domain_resolution": {"generativelanguage.googleapis.com": list(ips)}
    }
    captured = {}

    async def sandbox_exec(command, **kwargs):
        captured["command"] = command
        return type("Result", (), {"return_code": 0, "stdout": "", "stderr": ""})()

    env._sandbox_exec = sandbox_exec
    asyncio.run(DaytonaEnvironment._pin_resolved_hosts(env))

    # Every resolved IP is pinned to the host (order is shuffled per sandbox).
    for ip in ips:
        assert f"{ip} generativelanguage.googleapis.com" in captured["command"]
    assert captured["command"].count("generativelanguage.googleapis.com") == len(ips)
