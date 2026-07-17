from __future__ import annotations

import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import litellm

from presidio.errors import (
    EgressMisconfigError,
    CREDENTIAL_MARKERS,
    InvalidModelError,
    MissingCredentialError,
    MODEL_NOT_FOUND_MARKERS,
    message_matches_markers,
)
from presidio.models.agent.network import NetworkAllowlist
from presidio.models.environment_type import EnvironmentType
from presidio.models.job.config import JobConfig
from presidio.models.task.config import TaskConfig as HarborTaskConfig
from presidio.models.trial.config import EnvironmentConfig
from presidio.models.trial.paths import TrialPaths


@dataclass(frozen=True)
class ModelCheckResult:
    model: str
    status: str
    error: str | None = None


def _model_error(exc: Exception, model: str) -> Exception:
    text = f"{type(exc).__name__}: {exc}".lower()
    if message_matches_markers(text, CREDENTIAL_MARKERS):
        return MissingCredentialError(f"Missing credentials for model {model}: {exc}")
    if message_matches_markers(text, MODEL_NOT_FOUND_MARKERS):
        return InvalidModelError(f"Invalid model {model}: {exc}")
    return exc


def check_model(
    model: str,
    completion: Callable[..., Any] | None = None,
) -> ModelCheckResult:
    completion = completion or litellm.completion
    try:
        completion(
            model=model,
            messages=[{"role": "user", "content": "Reply with one token."}],
            max_tokens=1,
        )
    except Exception as exc:
        mapped = _model_error(exc, model)
        if mapped is not exc:
            raise mapped from exc
        raise
    return ModelCheckResult(model=model, status="pass")


def distinct_models(agents: Iterable[Any]) -> list[str]:
    models = {
        agent.model_name
        for agent in agents
        if getattr(agent, "model_name", None)
    }
    return sorted(models)


def run_model_preflight(
    config: JobConfig,
    *,
    output: Callable[[str], None] | None = None,
    completion: Callable[..., Any] | None = None,
) -> list[ModelCheckResult]:
    output = output or print
    results: list[ModelCheckResult] = []
    for model in distinct_models(config.agents):
        try:
            result = check_model(model, completion=completion)
        except (InvalidModelError, MissingCredentialError) as exc:
            output(f"{model}: fail ({exc})")
            raise
        except Exception as exc:
            output(f"{model}: fail ({exc})")
            raise
        results.append(result)
        output(f"{model}: pass")

    for agent in config.agents:
        if not getattr(agent, "model_name", None):
            output(f"{agent.name or agent.import_path or 'unknown'}: unchecked")
    return results


def _network_key(task: HarborTaskConfig) -> tuple[str, tuple[str, ...]]:
    environment = task.environment
    mode = environment.network_mode
    if mode is None:
        mode = "public" if environment.allow_internet else "no-network"
    return str(mode), tuple(sorted(environment.allowed_hosts))


def _network_hosts(task: HarborTaskConfig) -> list[str]:
    return list(task.environment.allowed_hosts)


async def probe_egress(
    task_configs: Iterable[Any],
    environment_config: EnvironmentConfig,
    *,
    environment_factory: Any | None = None,
    output: Callable[[str], None] | None = None,
) -> None:
    from presidio.environments.factory import EnvironmentFactory

    environment_factory = environment_factory or EnvironmentFactory
    output = output or print
    from presidio.models.task.task import Task

    groups: dict[tuple[str, tuple[str, ...]], tuple[Any, Any]] = {}
    for trial_task_config in task_configs:
        task = Task(trial_task_config.path)
        key = _network_key(task.config)
        if _network_hosts(task.config):
            groups.setdefault(key, (task, trial_task_config))

    if environment_config.type != EnvironmentType.DAYTONA:
        if groups:
            output("egress: unchecked (requires Daytona environment)")
        return

    for key, (task, _) in groups.items():
        mode, hosts = key
        output(f"egress {mode} ({', '.join(hosts)}): probing")
        with tempfile.TemporaryDirectory(prefix="presidio-preflight-") as temp:
            trial_dir = Path(temp) / "trial"
            trial_paths = TrialPaths(trial_dir)
            trial_paths.mkdir()
            minimal_config = environment_config.model_copy(deep=True)
            minimal_config.override_cpus = 1
            minimal_config.override_memory_mb = 1024
            minimal_config.override_storage_mb = 1024
            env = environment_factory.create_environment_from_config(
                config=minimal_config,
                environment_dir=task.paths.environment_dir,
                environment_name="preflight",
                session_id=f"preflight-{uuid4().hex[:12]}",
                trial_paths=trial_paths,
                task_env_config=task.config.environment,
                network_allowlist=NetworkAllowlist(domains=list(hosts)),
            )
            try:
                await env.start(force_build=False)
                for host in hosts:
                    result = await env.exec(
                        f"curl -sS --max-time 10 -o /dev/null -w '%{{http_code}}' "
                        f"https://{host}",
                        timeout_sec=15,
                    )
                    if result.return_code != 0:
                        raise EgressMisconfigError(
                            f"Egress probe failed for {host}: "
                            f"{result.stderr or result.stdout}"
                        )
                    status = (result.stdout or "").strip()
                    if status:
                        output(f"{host}: pass (HTTP {status})")
                    else:
                        output(f"{host}: pass")
            except EgressMisconfigError:
                raise
            except Exception as exc:
                raise EgressMisconfigError(
                    f"Egress probe failed for network configuration {key}: {exc}"
                ) from exc
            finally:
                await env.stop(delete=True)


async def run_preflight(
    config: JobConfig,
    *,
    include_egress: bool = False,
    output: Callable[[str], None] | None = None,
) -> None:
    run_model_preflight(config, output=output)
    if include_egress:
        from presidio.job import Job

        task_configs = await Job._resolve_task_configs(config)
        await probe_egress(
            task_configs,
            config.environment,
            output=output,
        )
