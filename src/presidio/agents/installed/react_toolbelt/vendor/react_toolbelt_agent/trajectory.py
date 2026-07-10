"""Trajectory serialization.

Writes the run as ``{"info": ..., "messages": [...]}`` — the same shape as a
mini-swe-agent v2 trajectory (assistant messages carry ``tool_calls``; tool
results use ``role: "tool"``; per-message ``usage``/``timestamp``), so
presidio's existing mini-swe → ATIF converter can consume it host-side.
"""

import json
import os
from pathlib import Path
from typing import Any

from . import VERSION
from .messages import AnyMessage, message_to_dict

TRAJECTORY_FORMAT = "react-toolbelt-v1"


def build_trajectory_dict(
    model: str,
    messages: list[AnyMessage],
    msg_meta: dict[int, dict[str, Any]],
    usage: dict[str, Any],
    agent_config: dict[str, Any],
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    serialized: list[dict[str, Any]] = []
    for msg in messages:
        data = message_to_dict(msg)
        meta = msg_meta.get(id(msg))
        if meta:
            if meta.get("usage"):
                data["usage"] = meta["usage"]
            if meta.get("timestamp"):
                data["timestamp"] = meta["timestamp"]
        serialized.append(data)

    return {
        "trajectory_format": TRAJECTORY_FORMAT,
        "info": {
            # Key names follow the mini-swe v2 layout the host-side ATIF
            # converter reads (config.model.model_name, config.agent,
            # mini_version, model_stats.instance_cost).
            "mini_version": VERSION,
            "config": {
                "model": {"model_name": model},
                "agent": agent_config,
            },
            "model_stats": {"instance_cost": usage.get("cost_usd") or 0.0},
            "usage": usage,
        },
        "messages": serialized,
        "result": result or {},
    }


def write_json_atomic(path: str | Path, data: dict[str, Any]) -> None:
    """Write JSON via a temp file + rename so a killed run never leaves a
    half-written trajectory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)
