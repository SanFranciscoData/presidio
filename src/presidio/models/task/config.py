import re
import tomllib
import warnings
from collections.abc import Sequence
from enum import Enum
from typing import Any

import toml
from pydantic import BaseModel, Field, field_validator, model_validator

from presidio.constants import ORG_NAME_PATTERN


class TaskOS(str, Enum):
    """Target operating system for a task's container."""

    LINUX = "linux"
    WINDOWS = "windows"


class VerifierEnvironmentMode(str, Enum):
    """Whether the verifier runs in the agent's environment or its own."""

    SHARED = "shared"
    SEPARATE = "separate"


class NetworkMode(str, Enum):
    """Network policy for an environment phase, mirroring Harbor's model.

    - ``public``: full network access.
    - ``no-network``: no network access.
    - ``allowlist``: egress only to hosts in ``allowed_hosts`` (empty or
      omitted hosts deny all egress).
    """

    PUBLIC = "public"
    NO_NETWORK = "no-network"
    ALLOWLIST = "allowlist"


def validate_bare_hostnames(hosts: list[str]) -> list[str]:
    """Reject URLs/ports/paths; allowlist entries are bare hostnames."""
    cleaned: list[str] = []
    for host in hosts:
        stripped = host.strip()
        if not stripped:
            raise ValueError("allowed_hosts entries must be non-empty hostnames")
        if "://" in stripped or "/" in stripped:
            raise ValueError(
                f"allowed_hosts entries must be bare hostnames, not URLs: {host!r}"
            )
        if ":" in stripped:
            raise ValueError(
                f"allowed_hosts entries must not include a port: {host!r}"
            )
        cleaned.append(stripped)
    return cleaned


def normalize_allowlist_hosts(*host_groups: Sequence[str]) -> list[str]:
    """Normalize hosts to squid dstdomain form ('*.example.com' -> '.example.com')."""
    normalized: set[str] = set()
    for group in host_groups:
        for host in group:
            stripped = host.strip()
            if not stripped:
                continue
            if stripped.startswith("*."):
                stripped = stripped[1:]
            normalized.add(stripped)
    return sorted(normalized)


def resolve_phase_network_override(
    network_mode: "NetworkMode | None", allowed_hosts: Sequence[str]
) -> "tuple[NetworkMode, list[str]] | None":
    """Resolve a per-phase ([agent]/[verifier]) network override.

    Returns ``None`` when the phase declares no override (``network_mode`` is
    unset), otherwise the effective ``(mode, hosts)`` that fully replaces the
    baseline for that phase. Only the ``allowlist`` mode carries hosts.
    """
    if network_mode is None:
        return None
    if network_mode == NetworkMode.ALLOWLIST:
        return NetworkMode.ALLOWLIST, normalize_allowlist_hosts(allowed_hosts)
    return network_mode, []


class Author(BaseModel):
    """Author information for a package or dataset."""

    name: str = Field(..., description="Author name")
    email: str | None = Field(default=None, description="Author email address")


class PackageInfo(BaseModel):
    """Package metadata for the [task] section of task.toml.

    This section identifies the package in the registry with a unique name.
    """

    name: str = Field(
        ...,
        description="Package name in org/name format (e.g., 'presidio/hello-world')",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the task",
    )
    authors: list[Author] = Field(
        default_factory=list,
        description="List of package authors",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Keywords for search and categorization",
    )

    @field_validator("name")
    @classmethod
    def validate_name_format(cls, v: str) -> str:
        """Validate that name follows org/name format."""
        if not re.match(ORG_NAME_PATTERN, v) or ".." in v:
            raise ValueError(
                f"Package name must be in 'org/name' format with alphanumeric characters, "
                f"hyphens, underscores, and dots. Cannot start with a dot or contain '..'. Got: {v}"
            )
        return v

    @property
    def org(self) -> str:
        """Extract organization from package name."""
        return self.name.split("/")[0]

    @property
    def short_name(self) -> str:
        """Extract short name (without org) from package name."""
        return self.name.split("/")[1]


class VerifierConfig(BaseModel):
    timeout_sec: float = 600.0
    env: dict[str, str] = Field(default_factory=dict)
    user: str | int | None = Field(
        default=None,
        description="Username or UID to run the verifier as. None uses the environment's default USER (e.g., root).",
    )
    environment_mode: VerifierEnvironmentMode | None = Field(
        default=None,
        description=(
            "Whether the verifier runs in the agent's environment ('shared') "
            "or in a dedicated container ('separate'). When omitted: defaults "
            "to 'separate' if a verifier 'environment' is set, otherwise "
            "'shared'."
        ),
    )
    environment: "EnvironmentConfig | None" = Field(
        default=None,
        description=(
            "Environment definition for the separate verifier container. "
            "Same schema as the top-level [environment] section. When set "
            "without an explicit environment_mode, implies "
            "environment_mode='separate'. When unset with "
            "environment_mode='separate', a fresh copy of the top-level "
            "[environment] is used. Conflicts with environment_mode='shared'."
        ),
    )
    network_mode: NetworkMode | None = Field(
        default=None,
        description="Per-phase network override for the verifier phase when it "
        "runs in the agent's (shared) environment. When set, it replaces the "
        "[environment] baseline for the duration of verification. Only honored "
        "by environments with the 'dynamic_network_policy' capability (e.g. "
        "e2b); for a separate verifier, configure [verifier.environment] "
        "instead.",
    )
    allowed_hosts: list[str] = Field(
        default_factory=list,
        description="Hostnames reachable during the verifier phase under "
        "network_mode='allowlist'. Same matching rules as "
        "[environment].allowed_hosts.",
    )

    @field_validator("allowed_hosts")
    @classmethod
    def _validate_allowed_hosts(cls, v: list[str]) -> list[str]:
        return validate_bare_hostnames(v)

    def resolve_phase_network(self) -> "tuple[NetworkMode, list[str]] | None":
        """Effective (mode, hosts) override for the verifier phase, or None."""
        return resolve_phase_network_override(self.network_mode, self.allowed_hosts)

    @model_validator(mode="after")
    def _validate_mode_env_consistency(self) -> "VerifierConfig":
        if (
            self.environment_mode == VerifierEnvironmentMode.SHARED
            and self.environment is not None
        ):
            raise ValueError(
                "[verifier].environment_mode='shared' is incompatible with "
                "[verifier.environment]; either omit the environment or set "
                "environment_mode='separate'."
            )
        # A [verifier].network_mode phase override is only applied when the
        # verifier shares the agent's (dynamic) environment. With a separate
        # verifier the override would be silently ignored, so reject it and
        # point at the right knob rather than running with the wrong policy.
        uses_separate_verifier = (
            self.environment_mode == VerifierEnvironmentMode.SEPARATE
            or self.environment is not None
        )
        if self.network_mode is not None and uses_separate_verifier:
            raise ValueError(
                "[verifier].network_mode is a phase override for a shared "
                "verifier and is ignored when the verifier runs in a separate "
                "environment. Set [verifier.environment].network_mode to "
                "configure the separate verifier's baseline instead."
            )
        return self


class SolutionConfig(BaseModel):
    env: dict[str, str] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    timeout_sec: float | None = None
    user: str | int | None = Field(
        default=None,
        description="Username or UID to run the agent as. None uses the environment's default USER (e.g., root).",
    )
    network_mode: NetworkMode | None = Field(
        default=None,
        description="Per-phase network override for the agent phase. When set, "
        "it replaces the [environment] baseline for the duration of the agent "
        "run. Only honored by environments with the 'dynamic_network_policy' "
        "capability (e.g. e2b).",
    )
    allowed_hosts: list[str] = Field(
        default_factory=list,
        description="Hostnames reachable during the agent phase under "
        "network_mode='allowlist'. Same matching rules as "
        "[environment].allowed_hosts.",
    )

    @field_validator("allowed_hosts")
    @classmethod
    def _validate_allowed_hosts(cls, v: list[str]) -> list[str]:
        return validate_bare_hostnames(v)

    def resolve_phase_network(self) -> "tuple[NetworkMode, list[str]] | None":
        """Effective (mode, hosts) override for the agent phase, or None."""
        return resolve_phase_network_override(self.network_mode, self.allowed_hosts)


class HealthcheckConfig(BaseModel):
    """Healthcheck configuration mirroring Docker HEALTHCHECK options.

    Runs a command repeatedly after environment start to verify readiness.
    All retries must pass before agent setup begins.
    """

    command: str = Field(..., description="Shell command to run. Exit 0 means healthy.")
    interval_sec: float = Field(
        default=5.0,
        description="Time in seconds between healthcheck attempts.",
    )
    timeout_sec: float = Field(
        default=30.0,
        description="Maximum time in seconds for a single healthcheck command to run.",
    )
    start_period_sec: float = Field(
        default=0.0,
        description="Grace period in seconds after environment start during which "
        "failures do not count toward retries.",
    )
    start_interval_sec: float = Field(
        default=5.0,
        description="Interval in seconds between checks during the start period.",
    )
    retries: int = Field(
        default=3,
        description="Number of consecutive failures before the healthcheck is considered failed.",
    )


class EnvironmentConfig(BaseModel):
    build_timeout_sec: float = 600.0  # 10 minutes default
    docker_image: str | None = None
    os: TaskOS = Field(
        default=TaskOS.LINUX,
        description="Target operating system for the task's container. "
        "Defaults to 'linux' for back-compat. Set to 'windows' to target "
        "Windows containers (requires Docker Desktop in Windows container "
        "mode on a Windows host).",
    )
    cpus: int | None = None
    memory_mb: int | None = None
    storage_mb: int = 10240
    gpus: int = 0
    gpu_types: list[str] | None = Field(
        default=None,
        description="List of acceptable GPU types (e.g., ['H100', 'A100', 'T4']). None "
        "means any GPU type is acceptable.",
    )
    allow_internet: bool = Field(
        default=True,
        description="Whether to allow internet access in the environment. "
        "Legacy shorthand: when 'network_mode' is unset, True maps to "
        "network_mode='public' and False to 'no-network' (or 'allowlist' "
        "when hosts are present). Ignored when 'network_mode' is set.",
    )
    network_mode: NetworkMode | None = Field(
        default=None,
        description="Network policy for this environment: 'public', "
        "'no-network', or 'allowlist'. When set, it takes precedence over the "
        "legacy 'allow_internet' flag. When omitted, the policy is derived "
        "from 'allow_internet' for back-compat.",
    )
    allowed_hosts: list[str] = Field(
        default_factory=list,
        description="Hostnames reachable under network_mode='allowlist'. "
        "Bare hosts (api.openai.com) match exactly; '*.example.com' (or the "
        "squid-style '.example.com') matches subdomains. Ignored unless the "
        "effective network mode is 'allowlist'.",
    )
    mcp_servers: list["MCPServerConfig"] = Field(default_factory=list)
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables required for the task and resolved from the host at runtime. "
        "Supports ${VAR} and ${VAR:-default} template syntax.",
    )
    skills_dir: str | None = Field(
        default=None,
        description="Path to skills directory in the environment. "
        "Contents are copied to the agent's skills config directory.",
    )
    healthcheck: HealthcheckConfig | None = Field(
        default=None,
        description="Healthcheck to run after environment start to verify readiness. "
        "Mirrors Docker HEALTHCHECK semantics.",
    )
    workdir: str | None = Field(
        default=None,
        description="Default working directory for command execution. "
        "Overrides the container's WORKDIR when set.",
    )

    # Deprecated fields - marked as excluded so they don't appear in serialization by default
    memory: str | None = Field(
        default=None,
        deprecated="Use 'memory_mb' instead. This field will be removed in a future version.",
        exclude=True,
    )
    storage: str | None = Field(
        default=None,
        deprecated="Use 'storage_mb' instead. This field will be removed in a future version.",
        exclude=True,
    )

    @field_validator("os", mode="before")
    @classmethod
    def normalize_os(cls, v: Any) -> Any:
        """Accept case-insensitive string values for the os field."""
        if isinstance(v, str):
            return v.lower()
        return v

    @field_validator("allowed_hosts")
    @classmethod
    def _validate_allowed_hosts(cls, v: list[str]) -> list[str]:
        """Reject URLs/ports/paths; allowed_hosts are bare hostnames."""
        return validate_bare_hostnames(v)

    @model_validator(mode="after")
    def _reconcile_network_mode(self) -> "EnvironmentConfig":
        """Make allow_internet consistent with an explicit network_mode.

        An explicit network_mode is authoritative; allow_internet is kept in
        sync so legacy enforcement paths (which read allow_internet) behave
        correctly. A directly contradictory allow_internet is rejected rather
        than silently overridden.
        """
        if self.network_mode is None:
            return self
        wants_internet = self.network_mode == NetworkMode.PUBLIC
        if (
            "allow_internet" in self.model_fields_set
            and self.allow_internet != wants_internet
        ):
            raise ValueError(
                f"network_mode={self.network_mode.value!r} conflicts with "
                f"allow_internet={self.allow_internet}; remove allow_internet "
                f"or set it to {str(wants_internet).lower()}."
            )
        self.allow_internet = wants_internet
        return self

    def resolve_network(
        self, extra_hosts: Sequence[str] = ()
    ) -> tuple[NetworkMode, list[str]]:
        """Resolve the effective (mode, allowed_hosts) for this environment.

        ``extra_hosts`` are phase-supplied hosts (e.g. an agent's self-declared
        provider hosts) that are unioned with ``allowed_hosts`` only when the
        effective mode is 'allowlist'. Hosts are normalized to squid dstdomain
        form ('*.example.com' -> '.example.com').
        """
        if self.network_mode is None:
            # Legacy allow_internet model.
            if self.allow_internet:
                return NetworkMode.PUBLIC, []
            hosts = self._normalize_hosts(self.allowed_hosts, extra_hosts)
            if hosts:
                return NetworkMode.ALLOWLIST, hosts
            return NetworkMode.NO_NETWORK, []
        if self.network_mode == NetworkMode.ALLOWLIST:
            return NetworkMode.ALLOWLIST, self._normalize_hosts(
                self.allowed_hosts, extra_hosts
            )
        # public / no-network ignore any allowlist.
        return self.network_mode, []

    @staticmethod
    def _normalize_hosts(*host_groups: Sequence[str]) -> list[str]:
        return normalize_allowlist_hosts(*host_groups)

    @staticmethod
    def _parse_size_to_mb(size_str: str) -> int:
        size_str = size_str.strip().upper()

        if size_str.endswith("G"):
            return int(float(size_str[:-1]) * 1024)
        elif size_str.endswith("M"):
            return int(float(size_str[:-1]))
        elif size_str.endswith("K"):
            return int(float(size_str[:-1]) / 1024)
        else:
            raise ValueError(
                f"Invalid size format: {size_str}. Expected format like '1G', "
                "'512M', etc."
            )

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_resource_fields(cls, data: Any) -> Any:
        """Map deprecated memory/storage fields to memory_mb/storage_mb."""
        if not isinstance(data, dict):
            return data

        if "memory" in data:
            warnings.warn(
                "The 'memory' field is deprecated. Use 'memory_mb' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            memory = data.pop("memory")
            if isinstance(memory, str):
                memory_mb = cls._parse_size_to_mb(memory)
                if "memory_mb" in data and data["memory_mb"] != memory_mb:
                    raise ValueError(
                        "Conflicting 'memory' and 'memory_mb' values: "
                        f"memory={memory!r} ({memory_mb} MB) != "
                        f"memory_mb={data['memory_mb']!r}."
                    )
                data.setdefault("memory_mb", memory_mb)

        if "storage" in data:
            warnings.warn(
                "The 'storage' field is deprecated. Use 'storage_mb' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            storage = data.pop("storage")
            if isinstance(storage, str):
                storage_mb = cls._parse_size_to_mb(storage)
                if "storage_mb" in data and data["storage_mb"] != storage_mb:
                    raise ValueError(
                        "Conflicting 'storage' and 'storage_mb' values: "
                        f"storage={storage!r} ({storage_mb} MB) != "
                        f"storage_mb={data['storage_mb']!r}."
                    )
                data.setdefault("storage_mb", storage_mb)

        return data


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server available to the agent."""

    name: str
    transport: str = "sse"  # "sse" | "streamable-http" | "stdio"
    url: str | None = None  # required for sse/streamable-http
    command: str | None = None  # for stdio
    args: list[str] = Field(default_factory=list)  # for stdio

    @model_validator(mode="after")
    def validate_transport_fields(self) -> "MCPServerConfig":
        if self.transport in ("sse", "streamable-http") and not self.url:
            raise ValueError(f"'url' is required for transport '{self.transport}'")
        if self.transport == "stdio" and not self.command:
            raise ValueError("'command' is required for transport 'stdio'")
        return self


class ArtifactConfig(BaseModel):
    source: str
    destination: str | None = None
    exclude: list[str] = Field(
        default_factory=list,
        description="Patterns to exclude when downloading a directory artifact "
        "(passed as tar --exclude flags).",
    )


class StepConfig(BaseModel):
    name: str
    agent: AgentConfig = Field(default_factory=AgentConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    min_reward: float | dict[str, float] | None = Field(
        default=None,
        description="If set, abort remaining steps when this step's rewards do "
        "not meet the threshold(s). A float gates on the 'reward' key (1D "
        "convention); a dict gates on each declared key (aborts if any key is "
        "below its threshold or missing from the rewards dict). A missing "
        "verifier_result (verifier crash) or missing gated key is treated as "
        "-inf. Ignored when verification is globally disabled.",
    )
    healthcheck: HealthcheckConfig | None = Field(
        default=None,
        description="Optional per-step healthcheck run after this step's setup "
        "completes and before the agent runs. Mirrors the semantics of the "
        "top-level environment healthcheck; start_period_sec applies as a grace "
        "period after setup. Supplements rather than replaces the top-level "
        "healthcheck.",
    )
    artifacts: list[str | ArtifactConfig] = Field(
        default_factory=list,
        description="Artifacts to collect after this step's verification into "
        "steps/{name}/artifacts/. Appended to task-level and trial-level "
        "artifacts during this step's collection pass.",
    )


class MultiStepRewardStrategy(str, Enum):
    """Strategy for deriving a trial-level reward from per-step verifier results."""

    MEAN = "mean"
    FINAL = "final"


class TaskConfig(BaseModel):
    schema_version: str = "1.2"
    task: PackageInfo | None = Field(
        default=None,
        description="Package information for the task, parsed from the [task] section of task.toml.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    solution: SolutionConfig = Field(default_factory=SolutionConfig)
    source: str | None = None
    multi_step_reward_strategy: MultiStepRewardStrategy | None = Field(
        default=None,
        description=(
            "How to derive the trial-level reward from per-step verifier "
            "results in a multi-step task. 'mean' computes per-key means "
            "across steps (missing keys treated as 0; steps without a "
            "verifier_result excluded). 'final' uses the last step's "
            "verifier_result verbatim. Only applies to multi-step tasks; "
            "leave unset for single-step tasks. Defaults to 'mean' when "
            "unset on a multi-step task."
        ),
    )
    steps: list[StepConfig] | None = None
    artifacts: list[str | ArtifactConfig] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def handle_version_rename(cls, data: Any) -> Any:
        if isinstance(data, dict) and "version" in data:
            data.setdefault("schema_version", data.pop("version"))
        return data

    @classmethod
    def model_validate_toml(cls, toml_data: str) -> "TaskConfig":
        toml_dict = tomllib.loads(toml_data)
        return cls.model_validate(toml_dict)

    def model_dump_toml(self) -> str:
        data = self._without_none(self.model_dump(mode="json"))

        parts: list[str] = []
        emitted: set[str] = set()
        root_fields = [
            "schema_version",
            "source",
            "multi_step_reward_strategy",
            "artifacts",
        ]
        known_sections = (
            "task",
            "steps",
            "metadata",
            "verifier",
            "agent",
            "environment",
            "solution",
        )
        root_data: dict[str, Any] = {}
        for field in root_fields:
            if field in data and not isinstance(data[field], dict):
                root_data[field] = data[field]
        for field, value in data.items():
            if field in root_fields or field in known_sections:
                continue
            if not self._is_toml_table_like(value):
                root_data[field] = value
        if root_data:
            parts.append(toml.dumps(root_data))
            emitted.update(root_data)

        if "task" in data:
            parts.append(toml.dumps({"task": data["task"]}))
            emitted.add("task")

        if "steps" in data:
            parts.append(toml.dumps({"steps": data["steps"]}))
            emitted.add("steps")

        for section in ("metadata", "verifier", "agent", "environment", "solution"):
            if section in data:
                parts.append(toml.dumps({section: data[section]}))
                emitted.add(section)

        for field, value in data.items():
            if field not in emitted:
                parts.append(toml.dumps({field: value}))
                emitted.add(field)

        return "\n\n".join(part.strip() for part in parts if part.strip()) + "\n"

    @staticmethod
    def _is_toml_table_like(value: Any) -> bool:
        return isinstance(value, dict) or (
            isinstance(value, list) and any(isinstance(item, dict) for item in value)
        )

    @classmethod
    def _without_none(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: cls._without_none(item)
                for key, item in value.items()
                if item is not None
            }
        if isinstance(value, list):
            return [cls._without_none(item) for item in value]
        return value
