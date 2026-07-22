import asyncio
import importlib.metadata
import json
import os
import re
import shlex
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from presidio.agents.base import BaseAgent
from presidio.environments.base import BaseEnvironment
from presidio.models.agent.context import AgentContext
from presidio.models.agent.install import AgentInstallSpec, InstallStep
from presidio.models.agent.name import AgentName
from presidio.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Step,
    ToolCall,
    Trajectory,
)
from presidio.utils.templating import render_prompt_template


class _EnvExecResult:
    def __init__(self, exit_code: int, output: bytes):
        self.exit_code = exit_code
        self.output = output


class PresidioTmuxSession:
    _ENTER_KEYS = {"Enter", "C-m", "KPEnter", "C-j", "^M", "^J"}
    _TMUX_COMPLETION_COMMAND = "; tmux wait -S done"
    _ENDS_WITH_NEWLINE_PATTERN = r"[\r\n]$"
    _NEWLINE_CHARS = "\r\n"

    def __init__(
        self,
        environment: BaseEnvironment,
        loop: asyncio.AbstractEventLoop,
        session_name: str,
        exec_timeout_sec: int,
        logger,
    ):
        self._environment = environment
        self._loop = loop
        self._session_name = session_name
        self._exec_timeout_sec = exec_timeout_sec
        self._logger = logger
        self._previous_buffer: str | None = None

    @property
    def logging_path(self) -> Path:
        return Path("/tmp/presidio-terminus") / f"{self._session_name}.log"

    def _exec(
        self, command: list[str], timeout_sec: int | float | None = None
    ) -> _EnvExecResult:
        command_str = shlex.join(command)
        future = asyncio.run_coroutine_threadsafe(
            self._environment.exec(
                command=command_str,
                user=None,
                timeout_sec=timeout_sec or self._exec_timeout_sec,
            ),
            self._loop,
        )
        result = future.result()
        return _EnvExecResult(
            exit_code=result.return_code,
            output=(result.stdout or "").encode(),
        )

    def start(self) -> None:
        command = (
            f"tmux new-session -x 160 -y 40 -d -s {shlex.quote(self._session_name)} \\; "
            f"set-option -t {shlex.quote(self._session_name)} history-limit 50000 \\; "
            f"pipe-pane -t {shlex.quote(self._session_name)} "
            f'"cat > {shlex.quote(str(self.logging_path))}"'
        )
        result = self._exec(["bash", "-c", command])
        if result.exit_code != 0:
            raise RuntimeError("Failed to start tmux session")

    def stop(self) -> None:
        try:
            self._exec(["tmux", "kill-session", "-t", self._session_name])
        except Exception:
            self._logger.debug(
                f"Failed to stop tmux session {self._session_name}", exc_info=True
            )

    def _is_enter_key(self, key: str) -> bool:
        return key in self._ENTER_KEYS

    def _ends_with_newline(self, key: str) -> bool:
        return re.search(self._ENDS_WITH_NEWLINE_PATTERN, key) is not None

    def _is_executing_command(self, key: str) -> bool:
        return self._is_enter_key(key) or self._ends_with_newline(key)

    def _prevent_execution(self, keys: list[str]) -> list[str]:
        keys = keys.copy()
        while keys and self._is_executing_command(keys[-1]):
            if self._is_enter_key(keys[-1]):
                keys.pop()
            else:
                stripped_key = keys[-1].rstrip(self._NEWLINE_CHARS)
                if stripped_key:
                    keys[-1] = stripped_key
                else:
                    keys.pop()
        return keys

    def _prepare_keys(
        self, keys: str | list[str], block: bool
    ) -> tuple[list[str], bool]:
        if isinstance(keys, str):
            keys = [keys]
        if not block or not keys or not self._is_executing_command(keys[-1]):
            return keys, False
        keys = self._prevent_execution(keys)
        keys.extend([self._TMUX_COMPLETION_COMMAND, "Enter"])
        return keys, True

    def _send_blocking_keys(self, keys: list[str], max_timeout_sec: float) -> None:
        start_time_sec = time.time()
        self._exec(["tmux", "send-keys", "-t", self._session_name, *keys])
        result = self._exec(
            ["timeout", f"{max_timeout_sec}s", "tmux", "wait", "done"],
            timeout_sec=int(max_timeout_sec) + 15,
        )
        if result.exit_code != 0:
            raise TimeoutError(f"Command timed out after {max_timeout_sec} seconds")
        self._logger.debug(
            f"Blocking command completed in {time.time() - start_time_sec:.2f}s."
        )

    def _send_non_blocking_keys(
        self, keys: list[str], min_timeout_sec: float
    ) -> None:
        start_time_sec = time.time()
        self._exec(["tmux", "send-keys", "-t", self._session_name, *keys])
        elapsed_time_sec = time.time() - start_time_sec
        if elapsed_time_sec < min_timeout_sec:
            time.sleep(min_timeout_sec - elapsed_time_sec)

    def send_keys(
        self,
        keys: str | list[str],
        block: bool = False,
        min_timeout_sec: float = 0.0,
        max_timeout_sec: float = 180.0,
    ) -> None:
        if block and min_timeout_sec > 0.0:
            self._logger.debug("min_timeout_sec will be ignored because block is True.")
        prepared_keys, is_blocking = self._prepare_keys(keys, block)
        self._logger.debug(
            f"Sending keys: {prepared_keys}"
            f" min_timeout_sec: {min_timeout_sec} max_timeout_sec: {max_timeout_sec}"
        )
        if is_blocking:
            self._send_blocking_keys(prepared_keys, max_timeout_sec)
        else:
            self._send_non_blocking_keys(prepared_keys, min_timeout_sec)

    def get_asciinema_timestamp(self) -> float:
        return 0.0

    def is_session_alive(self) -> bool:
        result = self._exec(["tmux", "has-session", "-t", self._session_name])
        return result.exit_code == 0

    def capture_pane(self, capture_entire: bool = False) -> str:
        command = ["tmux", "capture-pane", "-p"]
        if capture_entire:
            command.extend(["-S", "-"])
        command.extend(["-t", self._session_name])
        return self._exec(command).output.decode(errors="replace")

    def get_incremental_output(self) -> str:
        current_buffer = self.capture_pane(capture_entire=True)
        if self._previous_buffer is None:
            self._previous_buffer = current_buffer
            return f"Current Terminal Screen:\n{self._get_visible_screen()}"
        new_content = self._find_new_content(current_buffer)
        self._previous_buffer = current_buffer
        if new_content is not None:
            if new_content.strip():
                return f"New Terminal Output:\n{new_content}"
            return f"Current Terminal Screen:\n{self._get_visible_screen()}"
        return f"Current Terminal Screen:\n{self._get_visible_screen()}"

    def _find_new_content(self, current_buffer: str) -> str | None:
        if self._previous_buffer is None:
            return None
        previous_buffer = self._previous_buffer.strip()
        if previous_buffer in current_buffer:
            index = current_buffer.index(previous_buffer)
            if "\n" in previous_buffer:
                index = previous_buffer.rfind("\n")
            return current_buffer[index:]
        return None

    def _get_visible_screen(self) -> str:
        return self.capture_pane(capture_entire=False)

    def clear_history(self) -> None:
        try:
            result = self._exec(["tmux", "clear-history", "-t", self._session_name])
        except Exception as exc:
            self._logger.warning(
                f"Failed to clear tmux history for session {self._session_name}: {exc}"
            )
            return
        if result.exit_code != 0:
            self._logger.warning(
                f"Failed to clear tmux history for session {self._session_name}. "
                f"Exit code: {result.exit_code}"
            )


class _BaseTerminusAgent(BaseAgent):
    SUPPORTS_ATIF = True
    SUPPORTS_WINDOWS = False

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        extra_env: dict[str, str] | None = None,
        prompt_template_path: Path | str | None = None,
        max_episodes: int | None = None,
        api_base: str | None = None,
        temperature: float = 0.7,
        exec_timeout_sec: int = 1200,
        version: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._extra_env = dict(extra_env or {})
        self._prompt_template_path = (
            Path(prompt_template_path) if prompt_template_path else None
        )
        self._max_episodes = max_episodes
        self._api_base = api_base
        self._temperature = temperature
        self._exec_timeout_sec = exec_timeout_sec
        self._version_override = version

    def version(self) -> str:
        if self._version_override:
            return self._version_override
        try:
            return importlib.metadata.version("terminal-bench")
        except importlib.metadata.PackageNotFoundError:
            return "unknown"

    def install_spec(self) -> AgentInstallSpec:
        install_tmux = (
            "set -e; "
            "if command -v apt-get >/dev/null 2>&1; then "
            "export DEBIAN_FRONTEND=noninteractive; "
            "apt-get update -qq && apt-get install -y -qq tmux; "
            "elif command -v apk >/dev/null 2>&1; then "
            "apk add --no-cache tmux; "
            "elif command -v dnf >/dev/null 2>&1; then "
            "dnf install -y tmux; "
            "elif command -v yum >/dev/null 2>&1; then "
            "yum install -y tmux; "
            "else echo 'No supported package manager found' >&2; exit 1; fi"
        )
        return AgentInstallSpec(
            agent_name=self.name(),
            version=self.version(),
            steps=[InstallStep(run=install_tmux, user="root")],
            verification_command="tmux -V",
        )

    def _render_instruction(self, instruction: str) -> str:
        if self._prompt_template_path:
            return render_prompt_template(self._prompt_template_path, instruction)
        return instruction

    def _build_trajectory(self, result: Any) -> Trajectory | None:
        episode_dirs = []
        for episode_dir in self.logs_dir.glob("episode-*"):
            if not episode_dir.is_dir():
                continue
            match = re.fullmatch(r"episode-(\d+)", episode_dir.name)
            if match:
                episode_dirs.append((int(match.group(1)), episode_dir))
        episode_dirs.sort(key=lambda item: item[0])
        if not episode_dirs:
            return None

        steps = []
        for step_id, (episode_number, episode_dir) in enumerate(
            episode_dirs, start=1
        ):
            response = self._read_json(episode_dir / "response.json")
            debug = self._read_json(episode_dir / "debug.json")
            message_parts = [
                value
                for key in ("state_analysis", "explanation")
                if isinstance(value := response.get(key), str) and value
            ]
            tool_calls = []
            commands = response.get("commands")
            if isinstance(commands, list):
                for index, command in enumerate(commands):
                    if not isinstance(command, dict):
                        continue
                    tool_calls.append(
                        ToolCall(
                            tool_call_id=f"ep{episode_number}-{index}",
                            function_name="terminal",
                            arguments=command,
                        )
                    )

            step_kwargs: dict[str, Any] = {
                "step_id": step_id,
                "source": "agent",
                "model_name": self.model_name,
                "message": "\n".join(message_parts),
            }
            if tool_calls:
                step_kwargs["tool_calls"] = tool_calls
            metrics = self._episode_metrics(debug)
            if metrics is not None:
                step_kwargs["metrics"] = metrics
            timestamp = self._episode_timestamp(debug)
            if timestamp is not None:
                step_kwargs["timestamp"] = timestamp
            steps.append(Step(**step_kwargs))

        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = 0
        has_metrics = False
        for step in steps:
            metrics = step.metrics
            if metrics is None:
                continue
            has_metrics = True
            if metrics.prompt_tokens is not None:
                prompt_tokens += metrics.prompt_tokens
            if metrics.completion_tokens is not None:
                completion_tokens += metrics.completion_tokens
            if metrics.cached_tokens is not None:
                cached_tokens += metrics.cached_tokens

        fallback_prompt_tokens = self._optional_int(
            getattr(result, "total_input_tokens", None)
        )
        fallback_completion_tokens = self._optional_int(
            getattr(result, "total_output_tokens", None)
        )
        total_prompt_tokens = prompt_tokens if has_metrics else fallback_prompt_tokens
        total_completion_tokens = (
            completion_tokens if has_metrics else fallback_completion_tokens
        )
        total_cached_tokens = cached_tokens if has_metrics else None

        return Trajectory(
            schema_version="ATIF-v1.7",
            agent=Agent(
                name=self.name(),
                version=self.version(),
                model_name=self.model_name,
            ),
            steps=steps,
            final_metrics=FinalMetrics(
                total_prompt_tokens=total_prompt_tokens,
                total_completion_tokens=total_completion_tokens,
                total_cached_tokens=total_cached_tokens,
                total_steps=len(steps),
            ),
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    @classmethod
    def _optional_int(cls, value: Any) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    @classmethod
    def _episode_metrics(cls, debug: dict[str, Any]) -> Metrics | None:
        original_response = debug.get("original_response")
        if not isinstance(original_response, str):
            return None
        try:
            response = json.loads(original_response)
        except json.JSONDecodeError:
            return None
        if not isinstance(response, dict):
            return None
        usage = response.get("usage")
        if not isinstance(usage, dict):
            return None
        cached_tokens = None
        if "prompt_tokens" in usage or "completion_tokens" in usage:
            prompt_tokens = cls._optional_int(usage.get("prompt_tokens"))
            completion_tokens = cls._optional_int(usage.get("completion_tokens"))
            prompt_details = usage.get("prompt_tokens_details")
            if isinstance(prompt_details, dict):
                cached_tokens = cls._optional_int(prompt_details.get("cached_tokens"))
        else:
            input_tokens = cls._optional_int(usage.get("input_tokens"))
            cache_creation_tokens = cls._optional_int(
                usage.get("cache_creation_input_tokens")
            )
            cache_read_tokens = cls._optional_int(
                usage.get("cache_read_input_tokens")
            )
            if (
                input_tokens is None
                and cache_creation_tokens is None
                and cache_read_tokens is None
            ):
                prompt_tokens = None
            else:
                prompt_tokens = sum(
                    value or 0
                    for value in (
                        input_tokens,
                        cache_creation_tokens,
                        cache_read_tokens,
                    )
                )
            completion_tokens = cls._optional_int(usage.get("output_tokens"))
            cached_tokens = cache_read_tokens
        if prompt_tokens is None and completion_tokens is None:
            return None
        return Metrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
        )

    @staticmethod
    def _episode_timestamp(debug: dict[str, Any]) -> str | None:
        timestamp = debug.get("start_time")
        if not isinstance(timestamp, str):
            return None
        try:
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            return None
        return timestamp

    def _write_trajectory(self, result: Any) -> Trajectory | None:
        trajectory = self._build_trajectory(result)
        if trajectory is None:
            return None
        try:
            trajectory_path = self.logs_dir / "trajectory.json"
            trajectory_path.write_text(
                json.dumps(trajectory.to_json_dict(), indent=2)
            )
        except (OSError, TypeError, ValueError):
            self.logger.warning(
                "Failed to write Terminus ATIF trajectory", exc_info=True
            )
        return trajectory

    async def setup(self, environment: BaseEnvironment) -> None:
        command = (
            "set -e; "
            "command -v tmux >/dev/null 2>&1 || { "
            "echo 'tmux is missing; build-time agent installation was unavailable' "
            ">&2; exit 1; "
            "}; "
            "mkdir -p /tmp/presidio-terminus && chmod 777 /tmp/presidio-terminus"
        )
        result = await environment.exec(command=command)
        if result.return_code != 0:
            raise RuntimeError(
                "tmux must be present in the image; build-time installation was "
                "unavailable"
            )

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        if not self.model_name or "/" not in self.model_name:
            raise ValueError(
                "Terminus requires a model name in 'provider/model' format."
            )
        instruction = self._render_instruction(instruction)
        loop = asyncio.get_running_loop()

        def _run_sync():
            session = PresidioTmuxSession(
                environment,
                loop,
                session_name="agent",
                exec_timeout_sec=self._exec_timeout_sec,
                logger=self.logger,
            )
            session.start()
            try:
                tb_agent = self._make_tb_agent()
                return tb_agent.perform_task(
                    instruction, session, logging_dir=self.logs_dir
                )
            finally:
                session.stop()

        old_env = {key: os.environ.get(key) for key in self._extra_env}
        try:
            os.environ.update(self._extra_env)
            result = await asyncio.to_thread(_run_sync)
        except Exception:
            self.logger.exception("Terminus agent execution failed")
            raise
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        if result.total_input_tokens is not None:
            context.n_input_tokens = result.total_input_tokens
        if result.total_output_tokens is not None:
            context.n_output_tokens = result.total_output_tokens
        trajectory = self._write_trajectory(result)
        if trajectory is not None:
            context.n_agent_steps = len(trajectory.steps)

    def _make_tb_agent(self):
        raise NotImplementedError


class TerminusAgent(_BaseTerminusAgent):
    @staticmethod
    def name() -> str:
        return AgentName.TERMINUS.value

    def _make_tb_agent(self):
        from terminal_bench.agents.terminus_1 import Terminus

        kwargs = {
            "model_name": self.model_name,
            "api_base": self._api_base,
            "temperature": self._temperature,
        }
        if self._max_episodes is not None:
            kwargs["max_episodes"] = self._max_episodes
        return Terminus(**kwargs)


class Terminus2Agent(_BaseTerminusAgent):
    def __init__(
        self, *args: Any, parser_name: str = "json", **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self._parser_name = parser_name

    @staticmethod
    def name() -> str:
        return AgentName.TERMINUS_2.value

    def _make_tb_agent(self):
        from terminal_bench.agents.terminus_2.terminus_2 import Terminus2

        return Terminus2(
            model_name=self.model_name,
            max_episodes=self._max_episodes,
            parser_name=self._parser_name,
            api_base=self._api_base,
            temperature=self._temperature,
        )
