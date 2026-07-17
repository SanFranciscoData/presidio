import json

import pytest

from presidio.errors import (
    ErrorClass,
    InvalidModelError,
    MissingCredentialError,
    classify,
)
from presidio.models.trial.result import ExceptionInfo


@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (InvalidModelError("bad model"), ErrorClass.CONFIG_FATAL),
        (MissingCredentialError("missing key"), ErrorClass.CONFIG_FATAL),
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
        type("NonZeroAgentExitCodeError", (Exception,), {})(),
    ],
)
def test_classify_agent_task_name_fallback(exception):
    assert classify(exception) is ErrorClass.AGENT_TASK


def test_classify_unknown_error():
    assert classify(RuntimeError("unknown")) is ErrorClass.UNKNOWN


def test_exception_info_serializes_error_class():
    info = ExceptionInfo.from_exception(InvalidModelError("bad model"))
    payload = json.loads(info.model_dump_json())

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
