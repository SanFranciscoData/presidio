"""E2B sandbox environment.

E2B (https://e2b.dev) runs each task in a Firecracker microVM started from a
prebuilt *template*. Unlike the Docker/Daytona/Modal environments, E2B does not
build an image from a Dockerfile at trial time — the template is built ahead of
time with the E2B CLI (``e2b template build``) and referenced here by name via
``[environment].docker_image`` (treated as the template id; ``None`` uses E2B's
default base template).

Network policy is enforced natively by E2B's firewall, so this environment does
*not* use the squid egress proxy that the container-based environments share:

* ``public``     -> ``allow_internet_access=True``
* ``no-network`` -> ``allow_internet_access=False`` (deny all egress)
* ``allowlist``  -> ``network={"allow_out": [...], "deny_out": ["0.0.0.0/0"]}``;
  E2B matches bare hosts exactly and ``*.example.com`` wildcards by SNI/Host.

Because E2B can also rewrite egress rules on a *running* sandbox via
``update_network``, this environment declares ``dynamic_network_policy`` and
implements :meth:`apply_network_policy`, the seam used for per-phase
(``[agent]``/``[verifier]``) network overrides.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import math
import os
import shlex
import tarfile
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from presidio.environments.agent_setup import install_step_script
from presidio.environments.base import BaseEnvironment, ExecResult
from presidio.environments.build_lock import KeyedBuildLock
from presidio.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from presidio.models.agent.install import InstallStep
from presidio.models.environment_type import EnvironmentType
from presidio.models.task.config import NetworkMode
from presidio.utils.env import resolve_env_vars
from presidio.utils.optional_import import MissingExtraError

try:
    from e2b import AsyncSandbox, AsyncTemplate
    from e2b.sandbox.commands.command_handle import CommandExitException

    _HAS_E2B = True
except ImportError:
    _HAS_E2B = False

if TYPE_CHECKING:
    from e2b import AsyncSandbox

# E2B denies all egress by sending the whole IPv4 space to deny_out; allow rules
# always take precedence, so allow_out entries remain reachable.
_E2B_ALL_TRAFFIC = "0.0.0.0/0"

# E2B's default sandbox user. Commands with no explicit user run as this user,
# so an agent install with no configured user is baked in as it too.
_E2B_DEFAULT_USER = "user"

# Initial sandbox TTL at create time. A background keepalive then refreshes it
# while the sandbox is alive, so the harness (asyncio.wait_for) — not this value
# — decides when a run ends. Acts only as a backstop if the process dies.
E2B_DEFAULT_SANDBOX_TIMEOUT_SEC = 3600

# Hard ceiling E2B imposes on a sandbox's lifetime: 24h for Pro, 1h for Hobby
# accounts. set_timeout cannot push past this. Overridable for Hobby accounts
# (or future tiers) via E2B_MAX_SANDBOX_LIFETIME_SEC so the budget fail-fast
# check matches the account's real cap.
E2B_MAX_SANDBOX_LIFETIME_SEC = 24 * 60 * 60

# The keepalive extends the TTL to this rolling window every interval, so the
# sandbox always has comfortably more than `interval` seconds of life ahead.
_KEEPALIVE_WINDOW_SEC = 300
_KEEPALIVE_INTERVAL_SEC = 60

# Default per-command exec timeout when neither the caller nor the harness
# budget bounds it. Independent of the sandbox TTL so a long-lived sandbox does
# not turn every command into a multi-hour hang.
E2B_DEFAULT_EXEC_TIMEOUT_SEC = 3600


def _e2b_max_lifetime_sec() -> int:
    raw = os.environ.get("E2B_MAX_SANDBOX_LIFETIME_SEC")
    if not raw:
        return E2B_MAX_SANDBOX_LIFETIME_SEC
    try:
        value = int(float(raw))
    except ValueError:
        return E2B_MAX_SANDBOX_LIFETIME_SEC
    return value if value > 0 else E2B_MAX_SANDBOX_LIFETIME_SEC


def _e2b_preflight() -> None:
    if not os.environ.get("E2B_API_KEY"):
        raise SystemExit(
            "E2B requires E2B_API_KEY to be set. Create a key at "
            "https://e2b.dev/dashboard and export E2B_API_KEY, then try again."
        )


def network_allowlist_to_e2b_hosts(domains: list[str]) -> list[str]:
    """Convert resolved allowlist domains to E2B ``allow_out`` entries.

    ``resolve_network`` normalizes subdomain wildcards to the squid-style
    leading-dot form (``.example.com``). Squid's ``dstdomain .example.com``
    matches the apex *and* every subdomain, so to keep reachability identical
    across environments each leading-dot entry expands to both the apex
    (``example.com``) and the E2B subdomain glob (``*.example.com``). Bare
    hosts pass through unchanged.
    """
    hosts: set[str] = set()
    for domain in domains:
        stripped = domain.strip()
        if not stripped:
            continue
        if stripped.startswith("."):
            apex = stripped.lstrip(".")
            if apex:
                hosts.add(apex)
                hosts.add(f"*.{apex}")
            continue
        hosts.add(stripped)
    return sorted(hosts)


def _is_template_alias_conflict(exc: BaseException) -> bool:
    """True iff an E2B template build failed *only* because the template alias
    already exists -- i.e. a concurrent or previous build already registered it.

    Matched on the provider's unique-constraint signature rather than an
    exception class so it stays robust to E2B SDK error-type churn. The observed
    message is ``Error when inserting alias '...': ERROR: duplicate key value
    violates unique constraint "idx_env_aliases_alias_namespace_unique"
    (SQLSTATE 23505)``."""
    msg = str(exc).lower()
    if "duplicate key" in msg or "sqlstate 23505" in msg:
        return True
    return "alias" in msg and "already exists" in msg


class E2bEnvironment(BaseEnvironment):
    # Per-template-name build serialization, shared primitive with the other
    # build-from-scratch backends (see ``environments/build_lock.py``).
    _build_locks = KeyedBuildLock()

    def __init__(
        self,
        *args: Any,
        sandbox_timeout_sec: int = E2B_DEFAULT_SANDBOX_TIMEOUT_SEC,
        **kwargs: Any,
    ) -> None:
        self._sandbox_timeout_sec = sandbox_timeout_sec
        self._sandbox: AsyncSandbox | None = None
        self._keepalive_task: asyncio.Task[None] | None = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.E2B

    def provider_max_lifetime_sec(self) -> float | None:
        return float(_e2b_max_lifetime_sec())

    def _initial_timeout_sec(self) -> int:
        """TTL to request at create time.

        The keepalive maintains the sandbox afterwards, so this only needs to
        comfortably exceed the keepalive window. When a finite harness budget is
        known, start at least that high (capped at the provider max) so even a
        keepalive hiccup cannot reap a run early.
        """
        floor = max(self._sandbox_timeout_sec, _KEEPALIVE_WINDOW_SEC)
        if self._min_lifetime_sec is None:
            return floor
        budget = math.ceil(self._min_lifetime_sec) + _KEEPALIVE_WINDOW_SEC
        return min(max(floor, budget), _e2b_max_lifetime_sec())

    async def _keepalive_loop(self) -> None:
        """Periodically extend the sandbox TTL while it is alive.

        Keeps the rolling deadline ahead of the keepalive interval so the
        provider never reaps a sandbox under an active phase. The harness's
        asyncio.wait_for remains the only thing that aborts a run; stop()
        cancels this loop.
        """
        while self._sandbox is not None:
            try:
                await asyncio.sleep(_KEEPALIVE_INTERVAL_SEC)
                if self._sandbox is None:
                    return
                await self._sandbox.set_timeout(_KEEPALIVE_WINDOW_SEC)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - network/SDK errors
                self.logger.warning("E2B keepalive set_timeout failed: %s", exc)

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(
            disable_internet=True,
            network_allowlist=True,
            dynamic_network_policy=True,
            # The agent install is baked into the ephemeral template at build
            # time (full network), mirroring the docker/modal/daytona build-time
            # install path. E2B has no runtime install-network phase, so this is
            # the only fair place to install -- see ``_resolve_template``.
            preinstall_agents=True,
        )

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities | None:
        # CPU/memory are fixed by the E2B template, not requested per-sandbox.
        return None

    @classmethod
    def preflight(cls) -> None:
        _e2b_preflight()

    def _validate_definition(self) -> None:
        # E2B runs from a prebuilt template referenced by name; there is no
        # local definition file to validate.
        return

    # --- network ----------------------------------------------------------

    def _create_network_kwargs(self) -> dict[str, Any]:
        """E2B ``AsyncSandbox.create`` kwargs for the resolved baseline policy."""
        if self._network_mode == NetworkMode.PUBLIC:
            return {"allow_internet_access": True}
        if self._network_mode == NetworkMode.NO_NETWORK:
            return {"allow_internet_access": False}
        return {
            "network": {
                "allow_out": network_allowlist_to_e2b_hosts(
                    self.network_allowlist.domains
                ),
                "deny_out": [_E2B_ALL_TRAFFIC],
            }
        }

    def _network_update_payload(
        self, network_mode: NetworkMode, allowed_hosts: list[str]
    ) -> dict[str, Any]:
        """Build the ``update_network`` payload for a phase override."""
        if network_mode == NetworkMode.PUBLIC:
            return {"allow_internet_access": True}
        if network_mode == NetworkMode.NO_NETWORK:
            return {"allow_internet_access": False}
        return {
            "allow_out": network_allowlist_to_e2b_hosts(allowed_hosts),
            "deny_out": [_E2B_ALL_TRAFFIC],
        }

    async def apply_network_policy(
        self, network_mode: NetworkMode, allowed_hosts: list[str]
    ) -> None:
        """Rewrite egress rules on the running sandbox (phase override seam).

        E2B replaces the egress configuration wholesale, so the caller passes
        the fully resolved policy for the phase.
        """
        if self._sandbox is None:
            raise RuntimeError(
                "Cannot update network policy before the sandbox is started."
            )
        payload = self._network_update_payload(network_mode, allowed_hosts)
        self.logger.info(
            "Applying E2B phase network policy: mode=%s hosts=%s",
            network_mode.value,
            allowed_hosts if network_mode == NetworkMode.ALLOWLIST else [],
        )
        await self._sandbox.update_network(payload)

    # --- lifecycle --------------------------------------------------------

    def _sandbox_env(self) -> dict[str, str]:
        env: dict[str, str] = dict(self._persistent_env)
        if self.task_env_config.env:
            env = {**resolve_env_vars(self.task_env_config.env), **env}
        return env

    def _dockerfile_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    def _ephemeral_template_name(self) -> str:
        """A deterministic E2B template name keyed to the build context, so repeated trials of
        the same task reuse one template instead of rebuilding. Hashes the Dockerfile plus the
        names+sizes of the context files it may COPY (cheap signature; E2B's own layer cache
        catches finer content changes)."""
        h = hashlib.sha256()
        for p in sorted(self.environment_dir.rglob("*")):
            if p.is_file():
                h.update(p.relative_to(self.environment_dir).as_posix().encode())
                h.update(str(p.stat().st_size).encode())
                if p.name == "Dockerfile":
                    h.update(p.read_bytes())
        # A baked-in agent (or a prebuilt base image) changes the built layers,
        # so they must key the template name too -- otherwise two agents would
        # collide on one cached template.
        if self.agent_install_spec is not None:
            h.update(b"agent:")
            h.update(self.agent_install_spec.fingerprint().encode())
        if self.task_env_config.docker_image:
            h.update(b"image:")
            h.update(self.task_env_config.docker_image.encode())
        return f"presidio-{h.hexdigest()[:24]}"

    def _base_template_dockerfile(self) -> str:
        """The base Dockerfile text the ephemeral template builds from.

        A prebuilt ``docker_image`` is extended via ``FROM <image>`` (matching
        the docker-family precedence: a declared image wins over the on-disk
        Dockerfile); otherwise the task's ``environment/Dockerfile`` is the
        base. Agent install layers are applied separately via
        ``_apply_agent_install`` so they go through E2B's builder API rather
        than Dockerfile text."""
        image = self.task_env_config.docker_image
        if image:
            return f"FROM {image}\n"
        return self._dockerfile_path().read_text()

    def _agent_install_user(self, step: InstallStep) -> str:
        """The build user for an install step, matching E2B's *runtime* user.

        A ``root`` step always runs as root. Other steps run as the same user
        the agent itself runs as on E2B: the task's configured agent user, or
        -- when none is configured -- E2B's default sandbox user (so e.g. a
        ``npm install -g`` lands on that user's PATH). Note this differs from
        the docker family, where an unset user means root."""
        if step.user == "root":
            return "root"
        configured = self._resolve_user(None)
        return _E2B_DEFAULT_USER if configured is None else str(configured)

    def _apply_agent_install(self, builder: Any) -> Any:
        """Append the agent's install steps to the template builder.

        Each step's script is run through E2B's ``run_cmd`` builder API rather
        than emitted as Dockerfile text, because E2B's ``from_dockerfile``
        parser does not support exec-form ``RUN [...]`` instructions. The
        script is base64-piped into ``bash`` so arbitrary quoting survives
        intact and bash (not the build shell's ``sh``) interprets it. This is
        agent-agnostic: it replays whatever ``agent.install_spec()`` declares,
        with nothing hard-coded to a particular agent or model."""
        install = self.agent_install_spec
        if install is None:
            return builder
        for step in install.steps:
            encoded = base64.b64encode(install_step_script(step).encode()).decode()
            builder = builder.run_cmd(
                f"echo {encoded} | base64 -d | bash",
                user=self._agent_install_user(step),
            )
        return builder

    async def _resolve_template(self, force_build: bool) -> str | None:
        """The E2B template to boot.

        A task-declared prebuilt image wins outright when there is no agent to
        install. Otherwise, when the task ships an ``environment/Dockerfile``
        (or a prebuilt image to extend), build an ephemeral template from it so
        Dockerfile-based tasks run on E2B with no pre-declared template (parity
        with the container backends that build at trial time).

        When an ``agent_install_spec`` is present its steps are *baked into the
        built template* -- the template build runs with full network, exactly
        like the docker/modal/daytona build-time install path. Harbor models
        network as an ``[environment]`` baseline plus ``[agent]``/``[verifier]``
        phase overrides only -- there is no install-network phase -- so a
        build-time bake is the one fair place to install. A task that ships
        neither a Dockerfile nor a prebuilt image cannot bake an agent and
        fails fast.

        With neither a Dockerfile nor an agent to install, fall back to E2B's
        bare default template (legacy behavior)."""
        install = self.agent_install_spec
        docker_image = self.task_env_config.docker_image
        has_dockerfile = self._dockerfile_path().is_file()

        # A prebuilt image wins outright when there is no agent layer to add.
        if docker_image and install is None:
            return docker_image
        if not has_dockerfile and not docker_image:
            if install is not None:
                raise ValueError(
                    f"E2B cannot install agent '{install.agent_name}': the task "
                    "ships neither an environment/Dockerfile nor a prebuilt "
                    "[environment].docker_image to bake it into."
                )
            return None

        builder = AsyncTemplate(file_context_path=self.environment_dir).from_dockerfile(
            self._base_template_dockerfile()
        )
        builder = self._apply_agent_install(builder)
        name = self._ephemeral_template_name()
        self.logger.info(
            "Building ephemeral E2B template %s%s",
            name,
            f" with agent '{install.agent_name}' baked in" if install else "",
        )
        # Serialize builds of the same template within this process so N
        # concurrent trials of one task don't all try to register the same
        # alias at once (see ``environments/build_lock.py``); the cross-process
        # race is handled by idempotent reuse in ``_build_template``.
        async with self._build_locks(name):
            await self._build_template(builder, name, force_build)
        return name

    async def _build_template(self, builder: Any, name: str, force_build: bool) -> None:
        """Build the ephemeral template, tolerating a concurrent/previous winner.

        The per-name lock serializes builds in this process, but two separate
        ``presidio run`` invocations can still race to register the same template
        alias. E2B's build is idempotent once the alias exists, so a
        duplicate-alias failure means another build already created it -- treat
        that as success and reuse the named template rather than failing the
        trial."""
        try:
            await AsyncTemplate.build(
                builder,
                name=name,
                skip_cache=force_build,
                on_build_logs=lambda entry: self.logger.debug("e2b build: %s", entry),
            )
        except Exception as exc:
            if not _is_template_alias_conflict(exc):
                raise
            self.logger.info(
                "E2B template %s was already registered by a concurrent build; "
                "reusing it.",
                name,
            )

    async def start(self, force_build: bool) -> None:
        if not _HAS_E2B:
            raise MissingExtraError(package="e2b", extra="e2b")

        create_kwargs: dict[str, Any] = {
            "timeout": self._initial_timeout_sec(),
            **self._create_network_kwargs(),
        }
        template = await self._resolve_template(force_build)
        if template:
            create_kwargs["template"] = template
        env = self._sandbox_env()
        if env:
            create_kwargs["envs"] = env

        self._sandbox = await AsyncSandbox.create(**create_kwargs)

        # Keep the sandbox alive for as long as the harness runs; task.toml
        # timeouts (enforced by the harness) decide when a run actually ends.
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

        # Record the baseline as the active policy so a phase that declares no
        # override (or repeats the baseline) does not trigger a redundant
        # update_network call.
        mode, hosts = self.baseline_network_policy()
        self._active_network_policy = (mode, tuple(sorted(hosts)))

        paths = self.env_paths
        await self.reset_dirs(
            remove_dirs=[],
            create_dirs=[paths.agent_dir, paths.verifier_dir],
            chmod_dirs=[paths.agent_dir, paths.verifier_dir],
        )

    async def stop(self, delete: bool) -> None:
        await self._cancel_keepalive()
        if self._sandbox is None:
            self.logger.warning("No E2B sandbox to stop.")
            return
        try:
            await self._sandbox.kill()
        finally:
            self._sandbox = None

    async def _cancel_keepalive(self) -> None:
        task = self._keepalive_task
        self._keepalive_task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    def _require_sandbox(self) -> AsyncSandbox:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        return self._sandbox

    # --- exec & file transfer --------------------------------------------

    def _default_exec_timeout_sec(self) -> int:
        """Per-command timeout when the caller passes none.

        Bounded by the harness budget when known (so a command can never
        outlast the phase the harness will abort anyway), otherwise a fixed
        default. Decoupled from the sandbox TTL.
        """
        if self._min_lifetime_sec is not None:
            return max(1, math.ceil(self._min_lifetime_sec))
        return E2B_DEFAULT_EXEC_TIMEOUT_SEC

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        sandbox = self._require_sandbox()
        resolved_user = self._resolve_user(user)
        run_kwargs: dict[str, Any] = {
            "timeout": timeout_sec
            if timeout_sec is not None
            else self._default_exec_timeout_sec(),
        }
        if cwd is not None:
            run_kwargs["cwd"] = cwd
        merged_env = self._merge_env(env)
        if merged_env:
            run_kwargs["envs"] = merged_env
        if resolved_user is not None:
            run_kwargs["user"] = str(resolved_user)

        try:
            result = await sandbox.commands.run(command, **run_kwargs)
        except CommandExitException as exc:
            return ExecResult(
                stdout=exc.stdout, stderr=exc.stderr, return_code=exc.exit_code
            )
        return ExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.exit_code,
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        sandbox = self._require_sandbox()
        parent = str(Path(target_path).parent)
        if parent and parent not in (".", "/"):
            await self.exec(f"mkdir -p {shlex.quote(parent)}", user="root")
        # Hand the SDK a file object so the body is streamed from disk rather
        # than buffered whole into memory.
        with Path(source_path).open("rb") as handle:
            await sandbox.files.write(target_path, handle)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        source = Path(source_dir)
        if not source.is_dir():
            raise NotADirectoryError(f"{source} is not a directory")

        tar_name = f".hb-upload-{uuid4().hex}.tar.gz"
        remote_tar = f"/tmp/{tar_name}"
        await self.exec(f"mkdir -p {shlex.quote(target_dir)}", user="root")
        with tempfile.TemporaryDirectory() as host_tmp_dir:
            local_tar = Path(host_tmp_dir) / tar_name
            with tarfile.open(local_tar, "w:gz") as tf:
                for item in source.iterdir():
                    tf.add(item, arcname=item.name)
            await self.upload_file(local_tar, remote_tar)

        result = await self.exec(
            f"tar xzf {shlex.quote(remote_tar)} -C {shlex.quote(target_dir)}",
            timeout_sec=120,
            user="root",
        )
        await self.exec(f"rm -f {shlex.quote(remote_tar)}", user="root")
        if result.return_code != 0:
            output = result.stderr or result.stdout or "no output"
            raise RuntimeError(
                f"Failed to extract uploaded archive to {target_dir!r} "
                f"with code {result.return_code}: {output}"
            )

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        sandbox = self._require_sandbox()
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Stream chunks straight to disk instead of materializing the whole
        # file in memory before writing.
        stream = await sandbox.files.read(source_path, format="stream")
        with target.open("wb") as handle:
            async for chunk in stream:
                handle.write(chunk)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await self.download_dir_with_exclusions(
            source_dir=source_dir,
            target_dir=target_dir,
            exclude=[],
        )
