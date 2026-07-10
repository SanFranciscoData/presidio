"""CLI entry point: ``python -m react_toolbelt_agent --config run-config.json``.

The run config is written into the container by the presidio ``ReactToolbelt``
wrapper:

{
  "model": "anthropic/claude-...",
  "instruction": "...",
  "system_prompt": "..." | null,
  "mcp_servers": [{"name", "transport", "url"?, "command"?, "args"?}, ...],
  "trajectory_path": "/logs/agent/react-toolbelt.trajectory.json",
  "result_path": "/logs/agent/react-toolbelt.result.json",
  "config": {"timeout"?, "max_steps"?, "llm_response_timeout"?,
             "tool_call_timeout"?, "max_toolbelt_size"?},
  "extra_args": {...}
}

Exit codes: 0 = run finished (task completed or agent gave up — both are
valid trials), 3 = system error (infra), 4 = cancelled.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from loguru import logger

from .agent import ReActAgent, RunConfig
from .messages import AgentStatus
from .trajectory import build_trajectory_dict, write_json_atomic


def _load_run_config(path: str) -> tuple[RunConfig, str, str]:
    raw = json.loads(Path(path).read_text())
    values = raw.get("config") or {}
    config = RunConfig(
        model=raw["model"],
        instruction=raw["instruction"],
        mcp_servers=raw.get("mcp_servers") or [],
        system_prompt=raw.get("system_prompt"),
        timeout=int(values.get("timeout", 10800)),
        max_steps=int(values.get("max_steps", 250)),
        tool_call_timeout=int(values.get("tool_call_timeout", 60)),
        llm_response_timeout=int(values.get("llm_response_timeout", 600)),
        max_toolbelt_size=int(values.get("max_toolbelt_size", 80)),
        extra_args=raw.get("extra_args") or {},
    )
    return config, raw["trajectory_path"], raw["result_path"]


def _agent_config_dict(config: RunConfig) -> dict:
    return {
        "timeout": config.timeout,
        "max_steps": config.max_steps,
        "tool_call_timeout": config.tool_call_timeout,
        "llm_response_timeout": config.llm_response_timeout,
        "max_toolbelt_size": config.max_toolbelt_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="react_toolbelt_agent")
    parser.add_argument("--config", required=True, help="Path to run config JSON")
    args = parser.parse_args()

    config, trajectory_path, result_path = _load_run_config(args.config)
    agent_config = _agent_config_dict(config)

    def checkpoint(agent: ReActAgent) -> None:
        write_json_atomic(
            trajectory_path,
            build_trajectory_dict(
                model=config.model,
                messages=agent.resum.get_full_history(agent.messages),
                msg_meta=agent.msg_meta,
                usage=agent.usage_tracker.to_dict(),
                agent_config=agent_config,
            ),
        )

    agent = ReActAgent(config, on_step=checkpoint)
    result = asyncio.run(agent.run())

    result_dict = {
        "status": result.status.value,
        "final_answer": result.final_answer,
        "final_status": result.final_status,
        "time_elapsed": result.time_elapsed,
        "usage": result.usage,
    }
    write_json_atomic(
        trajectory_path,
        build_trajectory_dict(
            model=config.model,
            messages=result.messages,
            msg_meta=result.msg_meta,
            usage=result.usage,
            agent_config=agent_config,
            result=result_dict,
        ),
    )
    write_json_atomic(result_path, result_dict)

    logger.info(
        f"Run finished: status={result.status.value} "
        f"final_status={result.final_status} "
        f"steps_elapsed={result.time_elapsed:.0f}s"
    )

    if result.status == AgentStatus.ERROR:
        return 3
    if result.status == AgentStatus.CANCELLED:
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
