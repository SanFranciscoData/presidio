from types import SimpleNamespace

import pytest

from presidio.errors import EgressMisconfigError, InvalidModelError, MissingCredentialError
from presidio.environments.base import ExecResult
from presidio.models.environment_type import EnvironmentType
from presidio.models.job.config import JobConfig
from presidio.models.task.config import EnvironmentConfig as TaskEnvironmentConfig
from presidio.models.task.config import NetworkMode
from presidio.models.trial.config import AgentConfig
from presidio.models.trial.config import EnvironmentConfig
from presidio.preflight import check_model, probe_egress, run_model_preflight


def test_model_check_passes_with_one_token_completion():
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)

    result = check_model("openai/test", completion=completion)

    assert result.status == "pass"
    assert calls == [
        {
            "model": "openai/test",
            "messages": [{"role": "user", "content": "Reply with one token."}],
            "max_tokens": 1,
        }
    ]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("AuthenticationError: bad key"), MissingCredentialError),
        (RuntimeError("model not found"), InvalidModelError),
    ],
)
def test_model_check_maps_provider_errors(error, expected):
    with pytest.raises(expected):
        check_model("openai/test", completion=lambda **kwargs: (_ for _ in ()).throw(error))


def test_model_preflight_reports_unchecked_agents():
    messages = []
    config = JobConfig(agents=[AgentConfig(name="cursor-cli")])

    results = run_model_preflight(config, output=messages.append)

    assert results == []
    assert messages == ["cursor-cli: unchecked"]


def test_model_preflight_failure_happens_before_egress():
    config = JobConfig(agents=[AgentConfig(name="agent", model_name="openai/test")])

    with pytest.raises(MissingCredentialError):
        run_model_preflight(
            config,
            completion=lambda **kwargs: (_ for _ in ()).throw(
                RuntimeError("authentication failed")
            ),
        )


@pytest.mark.anyio
async def test_egress_probe_checks_hosts_and_deletes_sandbox(monkeypatch, tmp_path):
    task_config = SimpleNamespace(path=tmp_path)
    task = SimpleNamespace(
        name="task",
        paths=SimpleNamespace(environment_dir=tmp_path / "environment"),
        config=SimpleNamespace(
            environment=TaskEnvironmentConfig(
                network_mode=NetworkMode.ALLOWLIST,
                allowed_hosts=["api.example.com"],
            )
        ),
    )
    monkeypatch.setattr("presidio.models.task.task.Task", lambda path: task)

    class FakeEnvironment:
        def __init__(self):
            self.commands = []
            self.deleted = False

        async def start(self, force_build):
            return None

        async def exec(self, command, timeout_sec):
            self.commands.append((command, timeout_sec))
            return ExecResult(return_code=0)

        async def stop(self, delete):
            self.deleted = delete

    environment = FakeEnvironment()

    class FakeFactory:
        @staticmethod
        def create_environment_from_config(**kwargs):
            return environment

    output = []
    await probe_egress(
        [task_config],
        EnvironmentConfig(type=EnvironmentType.DAYTONA),
        environment_factory=FakeFactory,
        output=output.append,
    )

    assert environment.commands == [
        ("curl -sf --max-time 10 https://api.example.com", 15)
    ]
    assert environment.deleted is True
    assert "api.example.com: pass" in output


@pytest.mark.anyio
async def test_egress_probe_maps_host_failure(monkeypatch, tmp_path):
    task_config = SimpleNamespace(path=tmp_path)
    task = SimpleNamespace(
        name="task",
        paths=SimpleNamespace(environment_dir=tmp_path / "environment"),
        config=SimpleNamespace(
            environment=TaskEnvironmentConfig(
                network_mode=NetworkMode.ALLOWLIST,
                allowed_hosts=["api.example.com"],
            )
        ),
    )
    monkeypatch.setattr("presidio.models.task.task.Task", lambda path: task)

    class FakeEnvironment:
        async def start(self, force_build):
            return None

        async def exec(self, command, timeout_sec):
            return ExecResult(return_code=1, stderr="blocked")

        async def stop(self, delete):
            return None

    class FakeFactory:
        @staticmethod
        def create_environment_from_config(**kwargs):
            return FakeEnvironment()

    with pytest.raises(EgressMisconfigError):
        await probe_egress(
            [task_config],
            EnvironmentConfig(type=EnvironmentType.DAYTONA),
            environment_factory=FakeFactory,
        )
