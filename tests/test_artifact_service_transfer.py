"""Tests for service-aware artifact transfer.

Artifacts may declare a compose ``service`` (e.g. a kafka sidecar) whose
container holds the file; downloads must run against that service, while
plain/main artifacts keep the historical single-container path.
"""

import asyncio
import functools
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from presidio.environments.base import ServiceTransferUnsupportedError
from presidio.models.task.config import ArtifactConfig, TaskConfig
from presidio.trial.artifact_handler import ArtifactHandler


def run_async(fn):
    """Drive an async test with asyncio.run (presidio has no pytest-asyncio)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


def _fake_env(*, is_dir: bool = False) -> AsyncMock:
    env = AsyncMock()
    env.capabilities = SimpleNamespace(mounted=False)
    env.is_dir.return_value = is_dir
    env.exec_in_service.return_value = SimpleNamespace(
        return_code=0 if is_dir else 1, stdout="", stderr=""
    )
    return env


def _handler() -> ArtifactHandler:
    return ArtifactHandler(artifacts=[], logger=logging.getLogger(__name__))


def test_artifact_service_parses_from_task_toml() -> None:
    config = TaskConfig.model_validate(
        {
            "task": {"name": "org/task"},
            "artifacts": [
                "/app/src/",
                {"source": "/tmp/kafka-snapshot.tgz", "service": "kafka"},
            ],
        }
    )
    plain, sidecar = config.artifacts
    assert isinstance(plain, str)
    assert isinstance(sidecar, ArtifactConfig)
    assert sidecar.service == "kafka"


def test_invalid_artifact_service_name_rejected() -> None:
    with pytest.raises(ValueError):
        ArtifactConfig(source="/tmp/x", service="-bad")


@run_async
async def test_main_file_artifact_uses_plain_download(tmp_path: Path) -> None:
    env = _fake_env()
    handler = _handler()
    manifest = await handler.download_artifacts(
        env,
        tmp_path,
        source_artifacts_dir="/logs/artifacts",
        artifacts=[ArtifactConfig(source="/tmp/out.txt")],
    )
    sources = [c.kwargs["source_path"] for c in env.download_file.await_args_list]
    assert "/tmp/out.txt" in sources
    env.download_file_from_service.assert_not_awaited()
    entry = next(e for e in manifest.entries if e.source == "/tmp/out.txt")
    assert entry.status == "ok"


@run_async
async def test_sidecar_file_artifact_downloads_from_service(tmp_path: Path) -> None:
    """The published-task class: a kafka collect hook writes a snapshot in the
    sidecar, so the artifact copy must run against the kafka service."""
    env = _fake_env()
    handler = _handler()
    manifest = await handler.download_artifacts(
        env,
        tmp_path,
        source_artifacts_dir="/logs/artifacts",
        artifacts=[
            ArtifactConfig(source="/tmp/kafka-snapshot.tgz", service="kafka")
        ],
    )
    main_sources = [
        c.kwargs["source_path"] for c in env.download_file.await_args_list
    ]
    assert "/tmp/kafka-snapshot.tgz" not in main_sources
    env.download_file_from_service.assert_awaited_once_with(
        "kafka",
        source_path="/tmp/kafka-snapshot.tgz",
        target_path=tmp_path / "kafka-snapshot.tgz",
    )
    entry = next(
        e for e in manifest.entries if e.source == "/tmp/kafka-snapshot.tgz"
    )
    assert entry.status == "ok"


@run_async
async def test_sidecar_dir_artifact_downloads_from_service(tmp_path: Path) -> None:
    env = _fake_env(is_dir=True)
    handler = _handler()
    await handler.download_artifacts(
        env,
        tmp_path,
        source_artifacts_dir="/logs/artifacts",
        artifacts=[ArtifactConfig(source="/data/logs", service="kafka")],
    )
    main_dirs = [c.kwargs["source_dir"] for c in env.download_dir.await_args_list]
    assert "/data/logs" not in main_dirs
    env.download_dir_from_service.assert_awaited_once_with(
        "kafka",
        source_dir="/data/logs",
        target_dir=tmp_path / "logs",
    )


@run_async
async def test_unsupported_service_transfer_is_best_effort(tmp_path: Path) -> None:
    """Environments without auxiliary transfer record a failed manifest entry
    instead of crashing the trial."""
    env = _fake_env()
    env.download_file_from_service.side_effect = ServiceTransferUnsupportedError(
        "kafka"
    )
    handler = _handler()
    manifest = await handler.download_artifacts(
        env,
        tmp_path,
        source_artifacts_dir="/logs/artifacts",
        artifacts=[ArtifactConfig(source="/tmp/snap.tgz", service="kafka")],
    )
    entry = next(e for e in manifest.entries if e.source == "/tmp/snap.tgz")
    assert entry.status == "failed"


@run_async
async def test_sidecar_dir_with_exclusions_is_unsupported(tmp_path: Path) -> None:
    """Exclusion downloads tar through main; a sidecar source cannot use them,
    so the entry fails explicitly rather than silently reading main's tree."""
    env = _fake_env(is_dir=True)
    handler = _handler()
    manifest = await handler.download_artifacts(
        env,
        tmp_path,
        source_artifacts_dir="/logs/artifacts",
        artifacts=[
            ArtifactConfig(source="/data", service="kafka", exclude=["*.tmp"])
        ],
    )
    env.download_dir_with_exclusions.assert_not_awaited()
    entry = next(e for e in manifest.entries if e.source == "/data")
    assert entry.status == "failed"
