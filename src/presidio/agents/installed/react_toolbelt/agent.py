"""ReAct Toolbelt agent — presidio wrapper.

Runs the vendored ``react_toolbelt_agent`` package (see ``vendor/``, ported
from Mercor-Intelligence/archipelago) inside the task container: a LiteLLM
ReAct loop with dynamic toolbelt management over the task's MCP servers,
todo-gated ``final_answer`` termination, and ReSum context summarization.

Unlike the CLI agents (claude-code, codex, ...), the agent's only interface
to the environment is MCP — a task must declare at least one MCP server in
``environment.mcp_servers`` (e.g. a Playwright server for operator-profile
web tasks).
"""

import json
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from presidio.agents.installed.base import BaseInstalledAgent, with_prompt_template
from presidio.agents.installed.mini_swe_agent import convert_mini_swe_agent_to_atif
from presidio.agents.network import allowlist_from_urls
from presidio.agents.utils import get_api_key_var_names_from_model_name
from presidio.environments.base import BaseEnvironment
from presidio.models.agent.context import AgentContext
from presidio.models.agent.install import AgentInstallSpec, InstallStep
from presidio.models.agent.name import AgentName
from presidio.models.agent.network import NetworkAllowlist
from presidio.utils.trajectory_metrics import populate_context_from_final_metrics

_INSTALL_ROOT = PurePosixPath("/opt/react-toolbelt")
_VENV_PYTHON = _INSTALL_ROOT / "venv/bin/python"
_SRC_DIR = _INSTALL_ROOT / "src"
_VENDOR_DIR = Path(__file__).parent / "vendor"

_TRAJECTORY_FILE = "react-toolbelt.trajectory.json"
_RESULT_FILE = "react-toolbelt.result.json"

# Container-side deps for the vendored agent. The python-dotenv override
# mirrors upstream archipelago: litellm 1.83.4+ pins python-dotenv==1.0.1
# while fastmcp>=3.2 requires >=1.1.0; runtime works fine with 1.1.x.
# litellm[proxy]: newer litellm patches import litellm.responses.mcp.* ->
# litellm.proxy.* on the ordinary completion path, which transitively needs the
# proxy extras (fastapi, orjson, uvicorn, ...). Bare litellm dies with
# ModuleNotFoundError per-missing-module (fastapi, then orjson, ...) on every LLM
# call; the [proxy] extra pulls them all in one shot.
_VENV_PACKAGES = '"litellm[proxy]>=1.83.10,<2" "fastmcp>=3.2.0,<4" "loguru>=0.7.3"'
_VENV_OVERRIDES = "python-dotenv>=1.1.0"


class ReactToolbelt(BaseInstalledAgent):
    """ReAct loop with dynamic MCP toolbelt, ported from archipelago."""

    SUPPORTS_ATIF: bool = True

    # Tools an operator-profile agent has no business using — they let it bypass
    # the UI it is supposed to drive. Matches the claude-code operator's
    # --disallowedTools block. Escape hatches (arbitrary code exec / raw HTTP)
    # and read-only recon (console + network-request listing, which leak the
    # backend API shape) are always blocked. browser_evaluate (arbitrary in-page
    # JS) is blocked by default but re-enablable via allow_browser_evaluate.
    _ALWAYS_BLOCKED_TOOLS: tuple[str, ...] = (
        "browser_run_code_unsafe",
        "browser_network_request",
        "browser_network_requests",
        "browser_console_messages",
    )
    _EVALUATE_TOOL: str = "browser_evaluate"

    _DEFAULT_PROVIDER_DOMAINS: dict[str, list[str]] = {
        "anthropic": ["api.anthropic.com"],
        "bedrock": [".amazonaws.com"],
        "deepseek": ["api.deepseek.com"],
        "gemini": [".googleapis.com"],
        "google": [".googleapis.com"],
        "groq": ["api.groq.com"],
        "mistral": ["api.mistral.ai"],
        "openai": ["api.openai.com"],
        "openrouter": ["openrouter.ai"],
        "vertex_ai": [".googleapis.com"],
        "xai": ["api.x.ai"],
    }

    def __init__(
        self,
        system_prompt: str | None = None,
        agent_timeout: int | None = None,
        max_steps: int | None = None,
        llm_response_timeout: int | None = None,
        tool_call_timeout: int | None = None,
        max_toolbelt_size: int | None = None,
        blocked_tools: list[str] | None = None,
        allow_browser_evaluate: bool = False,
        extra_args: dict[str, Any] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._system_prompt = system_prompt
        self._extra_args = dict(extra_args) if extra_args else {}
        self._agent_config: dict[str, Any] = {}
        for key, value in (
            ("timeout", agent_timeout),
            ("max_steps", max_steps),
            ("llm_response_timeout", llm_response_timeout),
            ("tool_call_timeout", tool_call_timeout),
            ("max_toolbelt_size", max_toolbelt_size),
        ):
            if value is not None:
                self._agent_config[key] = value
        # Tool gating. An explicit blocked_tools list is an absolute override;
        # otherwise block the always-unsafe escape hatches plus browser_evaluate
        # unless the caller opts back in with allow_browser_evaluate=True.
        if blocked_tools is not None:
            effective_blocked = list(blocked_tools)
        else:
            effective_blocked = list(self._ALWAYS_BLOCKED_TOOLS)
            if not allow_browser_evaluate:
                effective_blocked.append(self._EVALUATE_TOOL)
        self._agent_config["blocked_tools"] = effective_blocked

    @staticmethod
    def name() -> str:
        return AgentName.REACT_TOOLBELT.value

    def get_version_command(self) -> str | None:
        return (
            f"PYTHONPATH={_SRC_DIR} {_VENV_PYTHON} -c "
            f'"import react_toolbelt_agent; print(react_toolbelt_agent.VERSION)"'
        )

    def install_spec(self) -> AgentInstallSpec:
        root_run = (
            "if command -v curl &>/dev/null; then true;"
            " elif command -v apt-get &>/dev/null; then"
            "  apt-get update && apt-get install -y curl;"
            " elif command -v apk &>/dev/null; then"
            "  apk add --no-cache curl;"
            " elif command -v yum &>/dev/null; then"
            "  yum install -y curl;"
            " elif command -v dnf &>/dev/null; then"
            "  dnf install -y curl;"
            " else"
            '  echo "Warning: no package manager found, assuming curl exists" >&2;'
            " fi"
        )
        venv_run = f"""
set -euo pipefail
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/0.7.13/install.sh \\
    | env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh
fi
mkdir -p {_INSTALL_ROOT}
uv venv {_INSTALL_ROOT}/venv --python 3.12
printf '{_VENV_OVERRIDES}\\n' > /tmp/react-toolbelt-overrides.txt
uv pip install --python {_VENV_PYTHON} \\
  --override /tmp/react-toolbelt-overrides.txt {_VENV_PACKAGES}
chmod -R a+rX {_INSTALL_ROOT}
"""
        return AgentInstallSpec(
            agent_name=self.name(),
            version=self._version,
            steps=[
                InstallStep(
                    user="root",
                    env={"DEBIAN_FRONTEND": "noninteractive"},
                    run=root_run,
                ),
                InstallStep(user="root", run=venv_run),
            ],
            verification_command=None,
        )

    async def setup(self, environment: BaseEnvironment) -> None:
        await super().setup(environment)
        # The vendored agent source is uploaded at setup (not baked at image
        # build) so the container always runs the code shipped with this
        # presidio version.
        await environment.upload_dir(
            source_dir=_VENDOR_DIR,
            target_dir=str(_SRC_DIR),
        )
        await environment.exec(command=f"chmod -R a+rX {_INSTALL_ROOT}", user="root")

    def network_allowlist(self) -> NetworkAllowlist:
        urls: list[str] = []
        for key in (
            "OPENAI_BASE_URL",
            "OPENAI_API_BASE",
            "ANTHROPIC_BASE_URL",
            "GEMINI_API_BASE",
            "OPENROUTER_API_BASE",
        ):
            if value := self._get_env(key):
                urls.append(value)

        provider = None
        if self.model_name and "/" in self.model_name:
            provider = self.model_name.split("/", 1)[0]

        return allowlist_from_urls(
            urls,
            default_domains=self._DEFAULT_PROVIDER_DOMAINS.get(provider or "", []),
        )

    def _build_env(self) -> dict[str, str]:
        env = self.build_process_env(
            {
                # Keep litellm from fetching its model-cost map over the
                # network at import time (blocked by the egress allowlist).
                "LITELLM_LOCAL_MODEL_COST_MAP": "true",
                "PYTHONPATH": str(_SRC_DIR),
                "PYTHONUNBUFFERED": "1",
                # Playwright/MCP browser servers this agent spawns over stdio
                # (e.g. @playwright/mcp) inherit THIS env. Without the browsers
                # path, `--browser chromium` can't locate the image's bundled
                # Chromium and falls back to the (unavailable) chrome-for-testing
                # channel. The Microsoft Playwright base image installs to
                # /ms-playwright; propagate it so operator-profile browser tasks
                # work the same way claude-code's MCP client already does.
                "PLAYWRIGHT_BROWSERS_PATH": "/ms-playwright",
            }
        )

        api_key_vars = get_api_key_var_names_from_model_name(self.model_name or "")
        for api_key_var in api_key_vars:
            value = self._get_env(api_key_var)
            if value:
                env[api_key_var] = value
            else:
                raise ValueError(
                    f"Unset API variable for model {self.model_name}. "
                    f"Please set {api_key_var}."
                )

        for key in (
            "OPENAI_API_BASE",
            "OPENAI_BASE_URL",
            "ANTHROPIC_BASE_URL",
            "GEMINI_API_BASE",
            "OPENROUTER_API_BASE",
        ):
            if value := self._get_env(key):
                env[key] = value

        return env

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("Model name must be in the format provider/model_name")
        if not self.mcp_servers:
            raise ValueError(
                "react-toolbelt requires at least one MCP server in the task's "
                "environment.mcp_servers — MCP is the agent's only interface "
                "to the environment."
            )

        env_paths = environment.env_paths
        run_config = {
            "model": self.model_name,
            "instruction": instruction,
            "system_prompt": self._system_prompt,
            "mcp_servers": [server.model_dump() for server in self.mcp_servers],
            "trajectory_path": str(env_paths.agent_dir / _TRAJECTORY_FILE),
            "result_path": str(env_paths.agent_dir / _RESULT_FILE),
            "config": self._agent_config,
            "extra_args": self._extra_args,
        }

        config_path = "/tmp/react-toolbelt-run/config.json"
        marker = f"RT_CONFIG_EOF_{uuid.uuid4().hex[:8]}"
        write_config_cmd = (
            f"mkdir -p /tmp/react-toolbelt-run\n"
            f"cat > '{config_path}' << '{marker}'\n"
            f"{json.dumps(run_config, indent=2)}\n"
            f"{marker}\n"
        )
        await self.exec_as_agent(environment, command=write_config_cmd)

        await self.exec_as_agent(
            environment,
            command=(
                f"{_VENV_PYTHON} -m react_toolbelt_agent --config {config_path} "
                f"2>&1 </dev/null | tee {env_paths.agent_dir}/react-toolbelt.txt"
            ),
            env=self._build_env(),
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        trajectory_path = self.logs_dir / _TRAJECTORY_FILE
        if not trajectory_path.exists():
            self.logger.debug(f"Trajectory file {trajectory_path} does not exist")
            return

        try:
            data = json.loads(trajectory_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            self.logger.debug(f"Failed to read trajectory: {e}")
            return

        try:
            # The vendored agent writes its trajectory in the mini-swe v2
            # message layout, so the existing ATIF converter applies; only
            # the agent identity needs correcting afterwards.
            atif_trajectory = convert_mini_swe_agent_to_atif(data, str(uuid.uuid4()))
            atif_trajectory.agent.name = self.name()
            atif_trajectory.agent.version = self.version() or (
                data.get("info") or {}
            ).get("mini_version", "unknown")
            atif_trajectory.notes = (
                "Converted from react-toolbelt trajectory format to ATIF"
            )
            (self.logs_dir / "trajectory.json").write_text(
                json.dumps(atif_trajectory.to_json_dict(), indent=2)
            )
            if atif_trajectory.final_metrics:
                populate_context_from_final_metrics(
                    context, atif_trajectory.final_metrics
                )
        except Exception as e:
            self.logger.debug(f"Failed to convert trajectory to ATIF format: {e}")

        usage = (data.get("info") or {}).get("usage") or {}
        compactions = usage.get("compactions")
        if isinstance(compactions, int) and compactions > 0:
            context.summarization_count = compactions

        result = data.get("result") or {}
        if result:
            metadata = dict(context.metadata or {})
            metadata["react_toolbelt"] = {
                "status": result.get("status"),
                "final_status": result.get("final_status"),
                "final_answer": result.get("final_answer"),
            }
            context.metadata = metadata
