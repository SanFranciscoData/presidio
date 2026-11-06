import json
import shlex
import tempfile
from pathlib import Path
from typing import Any, Literal

from presidio.agents.installed.base import BaseInstalledAgent, with_prompt_template
from presidio.agents.network import allowlist_from_urls
from presidio.errors import InvalidModelError, MissingCredentialError
from presidio.environments.base import BaseEnvironment
from presidio.models.agent.context import AgentContext
from presidio.models.agent.install import AgentInstallSpec, InstallStep
from presidio.models.agent.name import AgentName
from presidio.models.agent.network import NetworkAllowlist
from presidio.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from presidio.utils.trajectory_metrics import (
    extra_with_context_metrics,
    peak_context_tokens_from_steps,
    populate_context_from_final_metrics,
)

_ReasoningEffort = Literal["low", "medium", "high", "xhigh", "max"]


class AntigravityCli(BaseInstalledAgent):
    """Run Google's native agy CLI using headless Gemini API-key authentication."""

    SUPPORTS_ATIF = True
    _OUTPUT_FILENAME = "antigravity-cli.jsonl"
    _STDERR_FILENAME = "antigravity-cli.stderr.txt"
    _PROMPT_PATH = "/tmp/presidio-antigravity-instruction.txt"

    @staticmethod
    def name() -> str:
        return AgentName.ANTIGRAVITY_CLI.value

    def __init__(
        self,
        *args,
        reasoning_effort: _ReasoningEffort | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._reasoning_effort = reasoning_effort
        if self._version not in (None, "latest"):
            raise ValueError("Antigravity's installer only supports the latest release")
        if reasoning_effort not in (None, "low", "medium", "high", "xhigh", "max"):
            raise ValueError(
                "reasoning_effort must be low, medium, high, xhigh, or max"
            )

    def get_version_command(self) -> str:
        return '"$HOME/.local/bin/agy" --version'

    def install_spec(self) -> AgentInstallSpec:
        return AgentInstallSpec(
            agent_name=self.name(),
            version=self._version,
            steps=[
                InstallStep(
                    user="root",
                    env={"DEBIAN_FRONTEND": "noninteractive"},
                    run="apt-get update && apt-get install -y curl ca-certificates coreutils",
                ),
                InstallStep(
                    user="agent",
                    run=(
                        "set -euo pipefail; "
                        "curl -fsSL https://antigravity.google/cli/install.sh | bash && "
                        + self.get_version_command()
                    ),
                ),
            ],
            verification_command=self.get_version_command(),
        )

    def network_allowlist(self) -> NetworkAllowlist:
        return allowlist_from_urls(
            [self._get_env("GOOGLE_GEMINI_BASE_URL")],
            default_domains=[
                ".googleapis.com",
                "generativelanguage.googleapis.com",
                "oauth2.googleapis.com",
            ],
        )

    def _runtime_env(self) -> dict[str, str]:
        if (self._get_env("AGY_ADC_AUTH") or "").lower() not in ("", "false", "0"):
            raise ValueError(
                "antigravity-cli currently supports Gemini API-key auth only"
            )
        key = (
            self._extra_env.get("GEMINI_API_KEY")
            or self._extra_env.get("GOOGLE_API_KEY")
            or self._get_env("GEMINI_API_KEY")
            or self._get_env("GOOGLE_API_KEY")
        )
        if not key:
            raise MissingCredentialError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) is required for antigravity-cli"
            )
        env = {"GEMINI_API_KEY": key, "GEMINI_CLI_TRUST_WORKSPACE": "true"}
        base_url = self._get_env("GOOGLE_GEMINI_BASE_URL")
        if base_url:
            env["GOOGLE_GEMINI_BASE_URL"] = base_url
        env.update(
            {f"AGY_CLI_{k}": v for k, v in self._get_env_prefixed("AGY_CLI_").items()}
        )
        return env

    @staticmethod
    def _parse_known_models(stdout: str) -> set[str]:
        known = set()
        for line in stdout.splitlines():
            parts = [part.strip() for part in line.split("\t")]
            if len(parts) >= 2 and parts[0] and " " not in parts[0]:
                known.add(parts[0])
        return known

    def _settings_command(
        self, model: str | None = None, *, activate_mcp: bool = True
    ) -> str:
        config: dict[str, Any] = {
            "modelProvider": "gemini",
            "experimental": {"skills": True},
        }
        if model:
            config["customModelsConfig"] = {
                "customModels": {model: {"modelName": model}}
            }
        servers = {}
        for server in self.mcp_servers if activate_mcp else []:
            if server.transport == "stdio":
                servers[server.name] = {
                    "command": server.command,
                    "args": server.args,
                    "disabled": False,
                }
            else:
                servers[server.name] = {"serverUrl": server.url, "disabled": False}
        return (
            "mkdir -p ~/.gemini/antigravity-cli ~/.gemini/config && "
            f"printf %s {shlex.quote(json.dumps(config))} "
            "> ~/.gemini/antigravity-cli/settings.json && "
            f"printf %s {shlex.quote(json.dumps({'mcpServers': servers}))} "
            "> ~/.gemini/config/mcp_config.json"
        )

    def _build_run_command(self, model: str) -> str:
        log_dir = "/logs/agent"
        effort = (
            f"--effort {shlex.quote(self._reasoning_effort)} "
            if self._reasoning_effort
            else ""
        )
        return (
            f'"$HOME/.local/bin/agy" --dangerously-skip-permissions '
            '--print-timeout 24h --output-format stream-json --add-dir "$PWD" '
            f"--model {shlex.quote(model)} {effort}"
            f'--prompt="$(cat {shlex.quote(self._PROMPT_PATH)})" '
            f"2>{log_dir}/{self._STDERR_FILENAME} </dev/null "
            f"| stdbuf -oL tee {log_dir}/{self._OUTPUT_FILENAME}; "
            "rc=$?; if [ $rc -ne 0 ]; then "
            f"tail -c 262144 {log_dir}/{self._STDERR_FILENAME} >&2; fi; exit $rc"
        )

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        if not self.model_name:
            raise InvalidModelError("Model name is required for antigravity-cli")
        provider, sep, model = self.model_name.partition("/")
        if not sep:
            model = provider
        elif provider not in ("gemini", "google"):
            raise InvalidModelError("antigravity-cli requires a Gemini API model")
        env = self._runtime_env()
        self._instruction = instruction
        # Native model slugs must not be re-registered on agy's custom-model path.
        await self.exec_as_agent(
            environment, command=self._settings_command(activate_mcp=False)
        )
        known = set()
        try:
            result = await self.exec_as_agent(
                environment,
                command='"$HOME/.local/bin/agy" models </dev/null',
                env=env,
                timeout_sec=15,
            )
            known = self._parse_known_models(result.stdout or "")
        except Exception:
            self.logger.debug("agy model discovery failed; registering requested model")
        model_is_known = any(
            candidate in known
            for candidate in (
                model,
                *(f"{model}-{e}" for e in ("low", "medium", "high", "xhigh", "max")),
            )
        )
        await self.exec_as_agent(
            environment,
            command=self._settings_command(None if model_is_known else model),
        )
        if self.skills_dir:
            await self.exec_as_agent(
                environment,
                command=(
                    "mkdir -p ~/.gemini/antigravity-cli/skills && "
                    f"cp -r {shlex.quote(self.skills_dir)}/. "
                    "~/.gemini/antigravity-cli/skills/"
                ),
            )
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as prompt:
            prompt.write(instruction)
            prompt.flush()
            await environment.upload_file(prompt.name, self._PROMPT_PATH)
        await self.exec_as_root(environment, command=f"chmod 644 {self._PROMPT_PATH}")
        await self.exec_as_agent(
            environment, command=self._build_run_command(model), env=env
        )

    @staticmethod
    def _usage(usage: dict[str, Any]) -> Metrics:
        return Metrics(
            prompt_tokens=(
                usage.get("input_tokens", 0) + usage.get("cache_read_tokens", 0)
            )
            if "input_tokens" in usage
            else None,
            completion_tokens=usage.get("output_tokens"),
            cached_tokens=usage.get("cache_read_tokens"),
            extra={"thinking_tokens": usage["thinking_tokens"]}
            if "thinking_tokens" in usage
            else None,
        )

    def _cost(self, metrics: Metrics) -> float | None:
        if metrics.prompt_tokens is None or metrics.completion_tokens is None:
            return None
        import litellm

        try:
            input_cost, output_cost = litellm.cost_per_token(
                model=self.model_name,
                prompt_tokens=metrics.prompt_tokens,
                completion_tokens=metrics.completion_tokens,
                cache_read_input_tokens=metrics.cached_tokens or 0,
            )
        except Exception:
            self.logger.debug("No token pricing available for %s", self.model_name)
            return None
        return input_cost + output_cost

    def _convert_events(self, events: list[dict[str, Any]]) -> Trajectory | None:
        updates: dict[int, dict[str, Any]] = {}
        session_id = None
        result = {}
        for event in events:
            session_id = event.get("conversation_id") or session_id
            if event.get("event") == "result":
                result = event.get("result") or {}
                session_id = result.get("conversation_id") or session_id
            if event.get("event") != "step_update":
                continue
            update = event.get("step_update")
            if not isinstance(update, dict) or not isinstance(
                update.get("step_index"), int
            ):
                continue
            index = update["step_index"]
            prior = updates.setdefault(index, {})
            text = prior.get("text_delta", "") + (update.get("text_delta") or "")
            prior.update(update)
            prior["text_delta"] = text
        steps = []
        instruction = getattr(self, "_instruction", None)
        if instruction:
            steps.append(Step(step_id=1, source="user", message=instruction))
        for index, update in sorted(updates.items()):
            kind = update.get("step_type")
            if kind not in ("user_input", "agent_response", "tool"):
                continue
            if kind == "user_input" and instruction:
                continue
            message = update.get("text_delta") or ""
            if kind == "user_input" and not message:
                continue
            step = Step(
                step_id=len(steps) + 1,
                source="user" if kind == "user_input" else "agent",
                message=message,
                model_name=self.model_name if kind != "user_input" else None,
                metrics=self._usage(update["usage"]) if update.get("usage") else None,
            )
            tool = update.get("tool_info")
            if kind == "tool" and isinstance(tool, dict):
                call_id = f"step-{index}"
                step.tool_calls = [
                    ToolCall(
                        tool_call_id=call_id,
                        function_name=tool.get("name")
                        or update.get("tool_name")
                        or "unknown",
                        arguments=tool.get("parameters") or {},
                    )
                ]
                if "output" in tool:
                    output = tool["output"]
                    step.observation = Observation(
                        results=[
                            ObservationResult(
                                source_call_id=call_id,
                                content=output
                                if isinstance(output, str)
                                else json.dumps(output),
                            )
                        ]
                    )
            steps.append(step)
        if not steps and result.get("response"):
            steps.append(Step(step_id=1, source="agent", message=result["response"]))
        if not steps:
            return None
        usage = result.get("usage")
        if not isinstance(usage, dict):
            usages = [u["usage"] for u in updates.values() if u.get("usage")]
            usage = (
                {
                    key: sum(u.get(key, 0) for u in usages)
                    for key in ("input_tokens", "output_tokens", "cache_read_tokens")
                }
                if usages
                else {}
            )
        metrics = self._usage(usage)
        return Trajectory(
            session_id=session_id,
            agent=Agent(
                name=self.name(),
                version=self.version() or "unknown",
                model_name=self.model_name,
            ),
            steps=steps,
            final_metrics=FinalMetrics(
                total_prompt_tokens=metrics.prompt_tokens,
                total_completion_tokens=metrics.completion_tokens,
                total_cached_tokens=metrics.cached_tokens,
                total_steps=len(steps),
                total_cost_usd=self._cost(metrics),
                extra=extra_with_context_metrics(
                    {"status": result["status"]} if "status" in result else None,
                    peak_context_tokens=peak_context_tokens_from_steps(steps),
                    summarization_count=None,
                ),
            ),
        )

    def convert_trajectory(self, logs_dir: Path) -> Trajectory | None:
        path = logs_dir / self._OUTPUT_FILENAME
        if not path.exists():
            return None
        events = []
        for line in path.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return self._convert_events(events)

    def populate_context_post_run(self, context: AgentContext) -> None:
        trajectory = self.convert_trajectory(self.logs_dir)
        if trajectory is None:
            return
        (self.logs_dir / "trajectory.json").write_text(
            trajectory.model_dump_json(indent=2, exclude_none=True)
        )
        if trajectory.final_metrics:
            populate_context_from_final_metrics(context, trajectory.final_metrics)
