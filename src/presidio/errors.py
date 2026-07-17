from __future__ import annotations

import re
from enum import Enum


# Daytona org-level capacity/quota exhaustion. These surface as
# DaytonaValidationError with messages like "Failed to create sandbox: Total
# memory limit exceeded. Maximum allowed: 1000GiB." or "Total CPU limit
# exceeded. Maximum allowed: 500." Unlike genuinely invalid requests, these are
# org-wide capacity conditions that clear once usage drops, so they belong in
# provider_quota rather than provider_transient.
CAPACITY_LIMIT_MARKERS = ("limit exceeded",)


def message_matches_markers(message: str, markers: tuple[str, ...]) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in markers)


MODEL_NOT_FOUND_MARKERS = (
    "not found for api version",
    "model not found",
    "modelnotfounderror",
    "is not found",
    "unknown model",
    "does not exist",
    "not supported for generatecontent",
)

CREDENTIAL_MARKERS = (
    "api key not valid",
    "invalid api key",
    "api_key_invalid",
    "permission denied",
    "permission_denied",
    "unauthorized",
    "authenticationerror",
)


class ErrorClass(str, Enum):
    CONFIG_FATAL = "config_fatal"
    EGRESS_MISCONFIG = "egress_misconfig"
    PROVIDER_TRANSIENT = "provider_transient"
    PROVIDER_QUOTA = "provider_quota"
    RESOURCE_MISCONFIG = "resource_misconfig"
    AGENT_TASK = "agent_task"
    UNKNOWN = "unknown"


class InvalidModelError(ValueError):
    pass


class MissingCredentialError(ValueError):
    pass


class EgressMisconfigError(ValueError):
    pass


def _matching_classes(
    *classes: type[BaseException] | None,
) -> tuple[type[BaseException], ...]:
    return tuple(cls for cls in classes if cls is not None)


def _candidate_messages(exc: BaseException) -> tuple[str, ...]:
    message = str(exc)
    if type(exc).__name__ != "NonZeroAgentExitCodeError":
        return (message,)

    candidates: list[str] = []
    stdout_match = re.search(
        r"(?ms)^stdout:\s*(.*?)(?:^stderr:\s*|\Z)",
        message,
    )
    if stdout_match:
        stdout = stdout_match.group(1).strip()
        if stdout and stdout.lower() != "none":
            candidates.append(stdout)

    stderr_match = re.search(r"(?ms)^stderr:\s*(.*)$", message)
    if stderr_match:
        stderr = stderr_match.group(1).strip()
        if stderr and stderr.lower() != "none":
            candidates.append(stderr)

    candidates.append(message)
    return tuple(candidates)


def message_matches_markers(message: str, markers: tuple[str, ...]) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in markers)


def classify(exc: BaseException) -> ErrorClass:
    try:
        from daytona import (
            DaytonaError,
            DaytonaNotFoundError,
            DaytonaRateLimitError,
            DaytonaValidationError,
        )
    except ImportError:
        DaytonaError = None
        DaytonaNotFoundError = None
        DaytonaRateLimitError = None
        DaytonaValidationError = None

    # Daytona capacity/quota exhaustion arrives as a DaytonaValidationError
    # whose message reports an org-level limit being exceeded. Match on the
    # capacity condition rather than the class, since DaytonaValidationError
    # also covers genuinely invalid requests (which stay provider_transient via
    # the DaytonaError rule below).
    is_daytona_validation = (
        DaytonaValidationError is not None and isinstance(exc, DaytonaValidationError)
    ) or type(exc).__name__ == "DaytonaValidationError"
    if is_daytona_validation and message_matches_markers(
        str(exc), CAPACITY_LIMIT_MARKERS
    ):
        return ErrorClass.PROVIDER_QUOTA

    from presidio.agents.installed.base import NonZeroAgentExitCodeError
    from presidio.environments.base import HealthcheckError
    from presidio.trial.execution import (
        AgentSetupTimeoutError,
        AgentTimeoutError,
        EnvironmentStartTimeoutError,
    )
    from presidio.trial.trial import VerifierTimeoutError
    from presidio.verifier.verifier import (
        RewardFileEmptyError,
        RewardFileNotFoundError,
        VerifierOutputParseError,
    )

    rules = (
        (
            ErrorClass.PROVIDER_QUOTA,
            _matching_classes(DaytonaRateLimitError),
            {"DaytonaRateLimitError"},
        ),
        (
            ErrorClass.CONFIG_FATAL,
            (InvalidModelError, MissingCredentialError),
            {"InvalidModelError", "MissingCredentialError"},
        ),
        (
            ErrorClass.EGRESS_MISCONFIG,
            (EgressMisconfigError,),
            {"EgressMisconfigError"},
        ),
        (
            ErrorClass.PROVIDER_TRANSIENT,
            _matching_classes(
                DaytonaNotFoundError,
                DaytonaError,
                AgentSetupTimeoutError,
                EnvironmentStartTimeoutError,
                HealthcheckError,
            ),
            {
                "AgentSetupTimeoutError",
                "DaytonaNotFoundError",
                "DaytonaError",
                "DaytonaConnectionError",
                "DaytonaTimeoutError",
                "DaytonaCreateError",
                "CreateSandboxError",
                "DaytonaUploadError",
                "UploadError",
                "DaytonaDownloadError",
                "DownloadError",
                "EnvironmentStartTimeoutError",
                "HealthcheckError",
                "SDKError",
            },
        ),
        (
            ErrorClass.AGENT_TASK,
            (
                AgentTimeoutError,
                VerifierTimeoutError,
                RewardFileNotFoundError,
                RewardFileEmptyError,
                VerifierOutputParseError,
                NonZeroAgentExitCodeError,
            ),
            {
                "AgentTimeoutError",
                "VerifierTimeoutError",
                "RewardFileNotFoundError",
                "RewardFileEmptyError",
                "VerifierOutputParseError",
                "NonZeroAgentExitCodeError",
            },
        ),
    )

    for error_class, classes, names in rules:
        if not (isinstance(exc, classes) or type(exc).__name__ in names):
            continue
        if type(exc).__name__ == "NonZeroAgentExitCodeError":
            for candidate in _candidate_messages(exc):
                if message_matches_markers(
                    candidate,
                    MODEL_NOT_FOUND_MARKERS + CREDENTIAL_MARKERS,
                ):
                    return ErrorClass.CONFIG_FATAL
        return error_class
    return ErrorClass.UNKNOWN
