from pathlib import Path

import pytest

from presidio.agents.installed.base import NonZeroAgentExitCodeError
from presidio.errors import (
    CREDENTIAL_MARKERS,
    EgressMisconfigError,
    ErrorClass,
    InvalidModelError,
    MissingCredentialError,
    MODEL_NOT_FOUND_MARKERS,
    classify,
)
from presidio.models.trial.result import ExceptionInfo


def _nonzero_exit(message: str) -> NonZeroAgentExitCodeError:
    return NonZeroAgentExitCodeError(message)


@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (InvalidModelError("bad model"), ErrorClass.CONFIG_FATAL),
        (MissingCredentialError("missing key"), ErrorClass.CONFIG_FATAL),
        (EgressMisconfigError("blocked host"), ErrorClass.EGRESS_MISCONFIG),
    ],
)
def test_classify_configuration_errors(exception, expected):
    assert classify(exception) is expected


def test_classify_provider_quota_error():
    daytona = pytest.importorskip("daytona")
    assert classify(daytona.DaytonaRateLimitError("429", status_code=429)) is (
        ErrorClass.PROVIDER_QUOTA
    )


@pytest.mark.parametrize(
    "message",
    [
        "Failed to create sandbox: Total memory limit exceeded. "
        "Maximum allowed: 1000GiB.",
        "Failed to create sandbox: Total CPU limit exceeded. Maximum allowed: 500.",
    ],
)
def test_classify_daytona_capacity_limit_is_provider_quota(message):
    daytona = pytest.importorskip("daytona")
    assert classify(daytona.DaytonaValidationError(message)) is (
        ErrorClass.PROVIDER_QUOTA
    )


def test_classify_daytona_invalid_request_stays_provider_transient():
    daytona = pytest.importorskip("daytona")
    assert classify(
        daytona.DaytonaValidationError("Invalid snapshot name")
    ) is ErrorClass.PROVIDER_TRANSIENT


@pytest.mark.parametrize(
    "exception",
    [
        type("DaytonaNotFoundError", (Exception,), {})(),
        type("EnvironmentStartTimeoutError", (Exception,), {})(),
        type("HealthcheckError", (Exception,), {})(),
        type("DaytonaCreateError", (Exception,), {})(),
        type("UploadError", (Exception,), {})(),
        type("DownloadError", (Exception,), {})(),
    ],
)
def test_classify_provider_transient_name_fallback(exception):
    assert classify(exception) is ErrorClass.PROVIDER_TRANSIENT


@pytest.mark.parametrize(
    "exception",
    [
        type("AgentTimeoutError", (Exception,), {})(),
        type("VerifierTimeoutError", (Exception,), {})(),
        type("RewardFileNotFoundError", (Exception,), {})(),
        type("RewardFileEmptyError", (Exception,), {})(),
        type("VerifierOutputParseError", (Exception,), {})(),
    ],
)
def test_classify_agent_task_name_fallback(exception):
    assert classify(exception) is ErrorClass.AGENT_TASK


def test_classify_nonzero_agent_exit_model_not_found_fixture():
    fixture = Path(__file__).parent / "fixtures" / "gemini_not_found_stdout.txt"
    message = (
        "Command failed (exit 1): gemini --yolo --model=not-a-real-model\n"
        f"stdout: {fixture.read_text()}\n"
        "stderr: None"
    )

    assert classify(_nonzero_exit(message)) is ErrorClass.CONFIG_FATAL


@pytest.mark.parametrize(
    "message",
    [
        "Command failed (exit 1): some-agent\nstdout: API key not valid.\nstderr: None",
        "Command failed (exit 1): some-agent\nstdout: permission_denied\nstderr: None",
    ],
)
def test_classify_nonzero_agent_exit_credential_markers(message):
    assert classify(_nonzero_exit(message)) is ErrorClass.CONFIG_FATAL


def test_classify_generic_nonzero_exit_stays_agent_task():
    message = (
        "Command failed (exit 1): some-agent\n"
        "stdout: the task discussed authentication in plain English\n"
        "stderr: generic failure"
    )

    assert classify(_nonzero_exit(message)) is ErrorClass.AGENT_TASK


def test_classify_unknown_error():
    assert classify(RuntimeError("unknown")) is ErrorClass.UNKNOWN


def test_exception_info_serializes_error_class():
    info = ExceptionInfo.from_exception(InvalidModelError("bad model"))
    payload = info.model_dump(mode="json")

    assert payload["error_class"] == ErrorClass.CONFIG_FATAL.value
    assert ExceptionInfo.model_validate(
        {
            "exception_type": "LegacyError",
            "exception_message": "legacy",
            "exception_traceback": "",
            "occurred_at": info.occurred_at,
        }
    ).error_class is None


def test_agent_setup_timeout_is_provider_transient():
    from presidio.trial.execution import AgentSetupTimeoutError

    assert classify(AgentSetupTimeoutError()) is ErrorClass.PROVIDER_TRANSIENT


def test_marker_constants_are_specific():
    assert "api key not valid" in CREDENTIAL_MARKERS
    assert "not found for api version" in MODEL_NOT_FOUND_MARKERS
