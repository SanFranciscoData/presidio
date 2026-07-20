from enum import Enum


# Daytona org-level capacity/quota exhaustion. These surface as
# DaytonaValidationError with messages like "Failed to create sandbox: Total
# memory limit exceeded. Maximum allowed: 1000GiB." or "Total CPU limit
# exceeded. Maximum allowed: 500." Unlike genuinely invalid requests, these are
# org-wide capacity conditions that clear once usage drops, so they belong in
# provider_quota rather than provider_transient.
CAPACITY_LIMIT_MARKERS = ("limit exceeded",)


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


def _matching_classes(
    *classes: type[BaseException] | None,
) -> tuple[type[BaseException], ...]:
    return tuple(cls for cls in classes if cls is not None)


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
        if isinstance(exc, classes) or type(exc).__name__ in names:
            return error_class
    return ErrorClass.UNKNOWN
