import asyncio
import hashlib
import json
import shutil
from collections.abc import Coroutine
from datetime import datetime, timezone
from typing import Any

from presidio.errors import ErrorClass, InvalidModelError
from presidio.models.job.config import RetryConfig
from presidio.models.trial.config import TrialConfig
from presidio.models.trial.paths import TrialPaths
from presidio.models.trial.result import AgentInfo, ExceptionInfo, ModelInfo, TrialResult
from presidio.trial.hooks import HookCallback, TrialEvent, TrialHookEvent
from presidio.utils.logger import logger


class TrialQueue:
    """
    Handles orchestration of concurrent trials.

    Receives TrialConfigs, creates Trial objects internally, runs them
    with retry logic, and returns TrialResult tasks. Concurrency is
    bounded by an asyncio.Semaphore. Hooks are wired to each Trial
    instance — Trial handles all event invocations.
    """

    def __init__(
        self,
        n_concurrent: int,
        retry_config: RetryConfig | None = None,
        hooks: dict[TrialEvent, list[HookCallback]] | None = None,
        fail_fast: bool = True,
    ):
        if hooks is None:
            hooks = {event: [] for event in TrialEvent}
        else:
            for event in TrialEvent:
                hooks.setdefault(event, [])

        self._n_concurrent = n_concurrent
        self._retry_config = retry_config if retry_config is not None else RetryConfig()
        self._hooks = hooks
        self._fail_fast = fail_fast
        self._logger = logger.getChild(__name__)
        self._semaphore = asyncio.Semaphore(n_concurrent)
        self._cohort_outcomes: dict[str, list[ErrorClass | None]] = {}
        self._poisoned_cohorts: set[str] = set()
        self._pending_by_cohort: dict[str, int] = {}

    def add_hook(self, event: TrialEvent, callback: HookCallback) -> "TrialQueue":
        """Register a callback for a trial lifecycle event and return the queue."""
        self._hooks[event].append(callback)
        return self

    def on_trial_started(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when a queued trial starts."""
        return self.add_hook(TrialEvent.START, callback)

    def on_environment_started(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when a trial environment starts."""
        return self.add_hook(TrialEvent.ENVIRONMENT_START, callback)

    def on_agent_started(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when a trial agent starts."""
        return self.add_hook(TrialEvent.AGENT_START, callback)

    def on_verification_started(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when trial verification starts."""
        return self.add_hook(TrialEvent.VERIFICATION_START, callback)

    def on_trial_ended(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when a queued trial ends."""
        return self.add_hook(TrialEvent.END, callback)

    def on_trial_cancelled(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when a queued trial is cancelled."""
        return self.add_hook(TrialEvent.CANCEL, callback)

    def _should_retry_exception(
        self,
        exception_type: str,
        error_class: str | None = None,
        attempt: int = 0,
    ) -> bool:
        """Check if an exception should trigger a retry."""
        if self._retry_config.max_retries_by_class is not None:
            classified_error = self._resolve_error_class(error_class)

            max_retries = self._retry_config.max_retries_by_class.get(
                classified_error, 0
            )
            if attempt >= max_retries:
                self._logger.debug(
                    f"Exception {exception_type} classified as "
                    f"{classified_error.value} has exhausted its retry budget"
                )
                return False
            return True

        if (
            self._retry_config.exclude_exceptions
            and exception_type in self._retry_config.exclude_exceptions
        ):
            self._logger.debug(
                f"Exception {exception_type} is in exclude_exceptions, not retrying"
            )
            return False

        if (
            self._retry_config.include_exceptions
            and exception_type not in self._retry_config.include_exceptions
        ):
            self._logger.debug(
                f"Exception {exception_type} is not in include_exceptions, not retrying"
            )
            return False

        return True

    @staticmethod
    def _resolve_error_class(error_class: str | None) -> ErrorClass:
        try:
            return ErrorClass(error_class or ErrorClass.UNKNOWN.value)
        except ValueError:
            return ErrorClass.UNKNOWN

    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate the backoff delay for a retry attempt."""
        delay = self._retry_config.min_wait_sec * (
            self._retry_config.wait_multiplier**attempt
        )
        return min(delay, self._retry_config.max_wait_sec)

    def _setup_hooks(self, trial) -> None:
        """Wire queue-level hooks to the trial."""
        for event, hooks in self._hooks.items():
            for hook in hooks:
                trial.add_hook(event, hook)

    @staticmethod
    def cohort_key(trial_config: TrialConfig) -> str:
        agent = getattr(trial_config, "agent", None)
        payload = agent.model_dump(mode="json") if agent is not None else {}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()[:16]

    def _record_cohort_outcome(
        self, trial_config: TrialConfig, result: TrialResult
    ) -> TrialResult:
        if not self._fail_fast:
            return result
        cohort = self.cohort_key(trial_config)
        if cohort in self._poisoned_cohorts:
            return result
        outcomes = self._cohort_outcomes.setdefault(cohort, [])
        if len(outcomes) >= 3:
            return result
        error_class = (
            self._resolve_error_class(result.exception_info.error_class)
            if result.exception_info is not None
            else None
        )
        outcomes.append(error_class)
        if len(outcomes) == 3 and all(
            outcome == ErrorClass.CONFIG_FATAL for outcome in outcomes
        ):
            self._poisoned_cohorts.add(cohort)
            pending = self._pending_by_cohort.get(cohort, 0)
            self._logger.error(
                "Cohort %s poisoned after 3 consecutive CONFIG_FATAL results; "
                "skipping %d queued trials",
                cohort,
                pending,
            )
        return result

    async def _make_fail_fast_result(self, trial_config: TrialConfig) -> TrialResult:
        from presidio.trial.trial import Trial

        task = await Trial._load_task(trial_config)
        model_name = trial_config.agent.model_name
        model_info = None
        if model_name:
            if "/" in model_name:
                provider, name = model_name.split("/", 1)
            else:
                provider, name = None, model_name
            model_info = ModelInfo(name=name, provider=provider)
        agent_info = AgentInfo(
            name=trial_config.agent.name or trial_config.agent.import_path or "unknown",
            version="unknown",
            model_info=model_info,
        )
        trial_paths = TrialPaths(
            trial_dir=trial_config.trials_dir / trial_config.trial_name
        )
        trial_paths.mkdir()
        exception = InvalidModelError(
            "Trial skipped by fail-fast: its agent cohort was poisoned after "
            "three CONFIG_FATAL failures."
        )
        result = TrialResult(
            trial_name=trial_config.trial_name,
            task_name=task.name,
            task_id=trial_config.task.get_task_id(),
            trial_uri=trial_paths.trial_dir.resolve().as_uri(),
            task_checksum=task.checksum,
            config=trial_config,
            agent_info=agent_info,
            exception_info=ExceptionInfo.from_exception(exception),
            finished_at=datetime.now(timezone.utc),
            skipped_by_fail_fast=True,
            source=trial_config.task.source,
        )
        trial_paths.result_path.write_text(result.model_dump_json(indent=4))
        event = TrialHookEvent(
            event=TrialEvent.END,
            trial_id=trial_config.trial_name,
            task_name=task.name,
            config=trial_config,
            result=result,
        )
        for hook in self._hooks[TrialEvent.END]:
            await hook(event)
        return result

    async def _execute_trial_with_retries(
        self, trial_config: TrialConfig
    ) -> TrialResult:
        """Execute a trial with retry logic."""
        from presidio.trial.trial import Trial

        attempt = 0
        class_retry_counts: dict[ErrorClass, int] = {}
        while True:
            trial = await Trial.create(trial_config)
            self._setup_hooks(trial)
            result = await trial.run()

            if result.exception_info is None:
                return self._record_cohort_outcome(trial_config, result)

            # A trial that produced a verifier reward already carries the
            # authoritative signal, even though the agent process exited with an
            # exception. Verification runs regardless of a non-zero agent exit
            # (see Trial.run: NonZeroAgentExitCodeError is caught, then the
            # verifier still grades the final state), so an agent that exits
            # non-zero after reaching a gradeable state still records a real
            # reward. Retrying such a trial re-runs a full agent episode only to
            # recompute a reward we already have. Only signal-less trials (a crash
            # before any gradeable state, so no reward was recorded) are retried.
            if result.verifier_result is not None and result.verifier_result.rewards:
                self._logger.debug(
                    "Not retrying trial: it produced a verifier reward despite "
                    f"a {result.exception_info.exception_type} agent exit — the "
                    "reward is the authoritative signal."
                )
                return self._record_cohort_outcome(trial_config, result)

            classified_error = self._resolve_error_class(
                result.exception_info.error_class
            )
            class_attempt = class_retry_counts.get(classified_error, 0)
            if not self._should_retry_exception(
                result.exception_info.exception_type,
                result.exception_info.error_class,
                class_attempt,
            ):
                self._logger.debug(
                    "Not retrying trial because the exception is not in "
                    "include_exceptions or the maximum number of retries has been "
                    "reached"
                )
                return self._record_cohort_outcome(trial_config, result)
            if (
                self._retry_config.max_retries_by_class is None
                and attempt == self._retry_config.max_retries
            ):
                self._logger.debug(
                    "Not retrying trial because the maximum number of retries has been "
                    "reached"
                )
                return self._record_cohort_outcome(trial_config, result)

            class_retry_counts[classified_error] = class_attempt + 1
            attempt_dir = trial.trial_dir / "attempts" / f"attempt-{attempt + 1}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            for path in list(trial.trial_dir.iterdir()):
                if path.name == "attempts":
                    continue
                shutil.move(str(path), str(attempt_dir / path.name))

            delay = self._calculate_backoff_delay(attempt)

            self._logger.debug(
                f"Trial {trial_config.trial_name} failed with exception "
                f"{result.exception_info.exception_type}. Retrying in "
                f"{delay:.2f} seconds..."
            )

            await asyncio.sleep(delay)
            attempt += 1

    async def _run_trial(self, trial_config: TrialConfig) -> TrialResult:
        """Execute a single trial, acquiring the semaphore for concurrency control."""
        async with self._semaphore:
            cohort = self.cohort_key(trial_config)
            self._pending_by_cohort[cohort] = max(
                0, self._pending_by_cohort.get(cohort, 0) - 1
            )
            if self._fail_fast and cohort in self._poisoned_cohorts:
                return await self._make_fail_fast_result(trial_config)
            return await self._execute_trial_with_retries(trial_config)

    def submit(self, trial_config: TrialConfig) -> Coroutine[Any, Any, TrialResult]:
        """
        Return a coroutine that executes one trial.

        The caller decides how to schedule it (await, gather, TaskGroup).
        """
        return self._run_trial(trial_config)

    def submit_batch(
        self, configs: list[TrialConfig]
    ) -> list[Coroutine[Any, Any, TrialResult]]:
        """
        Return coroutines for multiple trials, ordered to match `configs`.
        """
        for config in configs:
            cohort = self.cohort_key(config)
            self._pending_by_cohort[cohort] = self._pending_by_cohort.get(cohort, 0) + 1
        return [self.submit(config) for config in configs]
