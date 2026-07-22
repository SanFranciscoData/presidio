import asyncio
import logging
from unittest.mock import AsyncMock, call

import pytest

from presidio.environments.daytona import (
    DAYTONA_MAX_NETWORK_ALLOWLIST_DOMAINS,
    DaytonaClientManager,
    DaytonaEnvironment,
    _DaytonaDirect,
    _DaytonaDinD,
)
from presidio.models.agent.network import NetworkAllowlist
from presidio.models.task.config import TaskOS
from presidio.models.trial.paths import EnvironmentPaths


def test_daytona_network_params_use_domain_allowlist():
    env = DaytonaEnvironment.__new__(DaytonaEnvironment)
    env._explicit_network_allow_list = None
    env._explicit_network_block_all = None
    env.task_env_config = type("TaskEnv", (), {"allow_internet": False})()
    env.network_allowlist = NetworkAllowlist(domains=["api.openai.com"])
    env.logger = logging.getLogger("test")

    assert DaytonaEnvironment._network_params(env) == {
        "network_block_all": False,
        "domain_allow_list": "api.openai.com",
    }


def test_daytona_network_params_preserves_wildcard_domain():
    env = DaytonaEnvironment.__new__(DaytonaEnvironment)
    env._explicit_network_allow_list = None
    env._explicit_network_block_all = None
    env.task_env_config = type("TaskEnv", (), {"allow_internet": False})()
    env.network_allowlist = NetworkAllowlist(domains=[".anthropic.com"])
    env.logger = logging.getLogger("test")

    assert DaytonaEnvironment._network_params(env) == {
        "network_block_all": False,
        "domain_allow_list": "*.anthropic.com",
    }


def test_daytona_network_params_block_when_domains_empty():
    env = DaytonaEnvironment.__new__(DaytonaEnvironment)
    env._explicit_network_allow_list = None
    env._explicit_network_block_all = None
    env.task_env_config = type("TaskEnv", (), {"allow_internet": False})()
    env.network_allowlist = NetworkAllowlist(domains=[])
    env.logger = logging.getLogger("test")

    assert DaytonaEnvironment._network_params(env) == {"network_block_all": True}


def test_daytona_network_params_caps_domain_allowlist(caplog):
    env = DaytonaEnvironment.__new__(DaytonaEnvironment)
    env._explicit_network_allow_list = None
    env._explicit_network_block_all = None
    env.task_env_config = type("TaskEnv", (), {"allow_internet": False})()
    domains = [f"api-{index}.example.com" for index in range(25)]
    env.network_allowlist = NetworkAllowlist(domains=domains)
    env.logger = logging.getLogger("test")

    with caplog.at_level(logging.WARNING, logger="test"):
        params = DaytonaEnvironment._network_params(env)

    assert params["network_block_all"] is False
    assert params["domain_allow_list"].split(",") == sorted(domains)[:DAYTONA_MAX_NETWORK_ALLOWLIST_DOMAINS]
    assert len([record for record in caplog.records if record.levelno == logging.WARNING]) == 1


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
    env._direct_image_built_from_dockerfile = True
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


def test_daytona_exec_without_user_uses_default_user(tmp_path):
    env = _reset_test_env(compose_mode=False, default_user="root")
    env.environment_dir = tmp_path
    (tmp_path / "Dockerfile").write_text("FROM alpine\n")

    asyncio.run(env.exec("echo ready"))

    assert env._strategy.exec.await_args.kwargs["user"] == "root"


@pytest.mark.parametrize(
    ("dockerfile", "expected_cwd"),
    [
        ("FROM alpine\n", "/"),
        ("FROM alpine\nWORKDIR /app\n", "/app"),
        ("FROM alpine\nWORKDIR /app\nWORKDIR data\n", "/app/data"),
    ],
)
def test_daytona_exec_uses_dockerfile_workdir_when_unset(
    tmp_path, dockerfile, expected_cwd
):
    env = _reset_test_env(compose_mode=False, default_user=None)
    env.environment_dir = tmp_path
    (tmp_path / "Dockerfile").write_text(dockerfile)

    asyncio.run(env.exec("echo ready"))

    assert env._strategy.exec.await_args.kwargs["cwd"] == expected_cwd


def test_daytona_dockerfile_workdir_expands_env_and_arg_variables(tmp_path):
    env = DaytonaEnvironment.__new__(DaytonaEnvironment)
    env.environment_dir = tmp_path
    (tmp_path / "Dockerfile").write_text(
        "FROM alpine\n"
        "ARG APP_HOME=/workspace\n"
        "ENV SOURCE_DIR=src\n"
        "WORKDIR $APP_HOME\n"
        "WORKDIR ${SOURCE_DIR}/$UNDECLARED\n"
    )

    assert env._dockerfile_workdir() == "/workspace/src"


def test_daytona_dockerfile_workdir_resets_between_stages(tmp_path):
    env = DaytonaEnvironment.__new__(DaytonaEnvironment)
    env.environment_dir = tmp_path
    (tmp_path / "Dockerfile").write_text(
        "FROM alpine AS builder\n"
        "WORKDIR /builder\n"
        "FROM alpine\n"
        "WORKDIR /app\n"
        "WORKDIR source\n"
    )

    assert env._dockerfile_workdir() == "/app/source"


def test_daytona_exec_empty_workdir_uses_dockerfile_workdir(tmp_path):
    env = _reset_test_env(compose_mode=False, default_user=None)
    env.environment_dir = tmp_path
    env.task_env_config.workdir = ""
    (tmp_path / "Dockerfile").write_text("FROM alpine\nWORKDIR /app\n")

    asyncio.run(env.exec("echo ready"))

    assert env._strategy.exec.await_args.kwargs["cwd"] == "/app"


def test_daytona_exec_prebuilt_image_ignores_dockerfile_workdir(tmp_path):
    env = _reset_test_env(compose_mode=False, default_user=None)
    env._direct_image_built_from_dockerfile = False
    env.environment_dir = tmp_path
    (tmp_path / "Dockerfile").write_text("FROM alpine\nWORKDIR /app\n")

    asyncio.run(env.exec("echo ready"))

    assert env._strategy.exec.await_args.kwargs["cwd"] == "/"


def test_daytona_exec_preserves_explicit_workdir_and_cwd(tmp_path):
    env = _reset_test_env(compose_mode=False, default_user=None)
    env.environment_dir = tmp_path
    (tmp_path / "Dockerfile").write_text("FROM alpine\nWORKDIR /app\n")

    asyncio.run(env.exec("echo ready", cwd="/tmp"))
    assert env._strategy.exec.await_args.kwargs["cwd"] == "/tmp"

    env.task_env_config.workdir = "/workspace"
    asyncio.run(env.exec("echo ready"))
    assert env._strategy.exec.await_args.kwargs["cwd"] == "/workspace"


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
    env._strategy = type("Strategy", (), {"is_dir": AsyncMock(return_value=False)})()
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
    env._strategy = type("Strategy", (), {"is_dir": AsyncMock(return_value=False)})()
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
    env.default_user = None
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
    env._sandbox_exec.assert_not_awaited()
