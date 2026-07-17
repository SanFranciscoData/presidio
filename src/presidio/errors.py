from enum import Enum


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
        )
    except ImportError:
        DaytonaError = None
        DaytonaNotFoundError = None
        DaytonaRateLimitError = None

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
