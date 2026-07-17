import asyncio
from pathlib import Path
from types import SimpleNamespace

from presidio.errors import ErrorClass
from presidio.models.trial.config import AgentConfig, TaskConfig, TrialConfig
from presidio.trial.queue import TrialQueue


def _config(tmp_path: Path, model_name: str = "openai/gpt-test") -> TrialConfig:
    return TrialConfig(
        task=TaskConfig(path=tmp_path, source="test"),
        trial_name="trial",
        trials_dir=tmp_path / "trials",
        agent=AgentConfig(name="gemini-cli", model_name=model_name),
    )


def _fatal_result():
    return SimpleNamespace(
        exception_info=SimpleNamespace(
            error_class=ErrorClass.CONFIG_FATAL.value,
        )
    )


def _success_result():
    return SimpleNamespace(exception_info=None)


def test_three_config_fatal_results_poison_cohort(tmp_path):
    queue = TrialQueue(n_concurrent=1)
    config = _config(tmp_path)

    for _ in range(3):
        queue._record_cohort_outcome(config, _fatal_result())

    assert queue.cohort_key(config) in queue._poisoned_cohorts


def test_mixed_results_do_not_poison_cohort(tmp_path):
    queue = TrialQueue(n_concurrent=1)
    config = _config(tmp_path)

    queue._record_cohort_outcome(config, _fatal_result())
    queue._record_cohort_outcome(config, _fatal_result())
    queue._record_cohort_outcome(config, _success_result())

    assert queue.cohort_key(config) not in queue._poisoned_cohorts


def test_fail_fast_result_is_persisted_and_fires_end_hook(tmp_path, monkeypatch):
    config = _config(tmp_path)
    queue = TrialQueue(n_concurrent=1)
    queue._poisoned_cohorts.add(queue.cohort_key(config))
    events = []

    async def on_end(event):
        events.append(event)

    queue.on_trial_ended(on_end)

    class FakeTask:
        name = "task"
        checksum = "checksum"

    import presidio.trial.trial as trial_module

    monkeypatch.setattr(
        trial_module.Trial,
        "_load_task",
        staticmethod(lambda config: _fake_task(FakeTask())),
    )
    result = asyncio.run(queue._make_fail_fast_result(config))

    assert result.skipped_by_fail_fast is True
    assert result.exception_info.error_class == ErrorClass.CONFIG_FATAL.value
    assert result.exception_info.exception_message.startswith(
        "Trial skipped by fail-fast"
    )
    assert len(events) == 1
    assert events[0].result is result
    assert (
        config.trials_dir / config.trial_name / "result.json"
    ).exists()


async def _fake_task(task):
    return task


def test_fail_fast_skips_before_trial_creation(tmp_path, monkeypatch):
    config = _config(tmp_path)
    queue = TrialQueue(n_concurrent=1)
    queue._poisoned_cohorts.add(queue.cohort_key(config))
    called = False

    async def execute(_config):
        nonlocal called
        called = True

    monkeypatch.setattr(queue, "_execute_trial_with_retries", execute)
    monkeypatch.setattr(
        queue,
        "_make_fail_fast_result",
        lambda _config: _skipped_result(),
    )

    result = asyncio.run(queue._run_trial(config))

    assert result.skipped_by_fail_fast is True
    assert called is False


async def _skipped_result():
    return SimpleNamespace(skipped_by_fail_fast=True)
