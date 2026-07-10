"""Usage accounting for the agent loop.

Adapted from archipelago ``runner/utils/usage.py``: keeps run totals
(prompt/completion/cached tokens, cost, compactions) plus the per-call usage
dicts needed for per-step trajectory metrics; drops upstream's synthetic
output-token breakdown (tool-output/image/final-answer token attribution).
"""

from typing import Any

from litellm.files.main import ModelResponse


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def usage_to_dict(usage: Any) -> dict[str, Any]:
    """Normalize a litellm usage object to a plain dict."""
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    return {}


class UsageTracker:
    """Accumulates token/cost totals across LLM calls in a run."""

    def __init__(self, model: str | None = None):
        self.model = model
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_tokens = 0
        self.reasoning_tokens = 0
        self.cost_usd = 0.0
        self.llm_calls = 0
        self.compactions = 0

    def track_compaction(self) -> None:
        self.compactions += 1

    def track(self, response: ModelResponse) -> dict[str, Any]:
        """Record a response's usage; returns the normalized usage dict.

        The returned dict is also suitable for embedding in the serialized
        assistant message so the trajectory converter can compute per-step
        metrics.
        """
        self.llm_calls += 1
        usage = usage_to_dict(getattr(response, "usage", None))
        if not usage:
            return {}

        self.prompt_tokens += _coerce_int(usage.get("prompt_tokens"))
        self.completion_tokens += _coerce_int(usage.get("completion_tokens"))

        prompt_details = usage.get("prompt_tokens_details") or {}
        if not isinstance(prompt_details, dict):
            prompt_details = {}
        self.cached_tokens += _coerce_int(
            prompt_details.get("cached_tokens") or usage.get("cache_read_input_tokens")
        )

        completion_details = usage.get("completion_tokens_details") or {}
        if isinstance(completion_details, dict):
            self.reasoning_tokens += _coerce_int(
                completion_details.get("reasoning_tokens")
            )

        # LiteLLM computes response cost for known models and stashes it in
        # hidden params; absent/unknown models simply contribute 0.
        hidden = getattr(response, "_hidden_params", None) or {}
        cost = hidden.get("response_cost")
        if isinstance(cost, (int, float)) and cost > 0:
            self.cost_usd += float(cost)
            usage = {**usage, "cost": float(cost)}

        return usage

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost_usd": self.cost_usd,
            "llm_calls": self.llm_calls,
            "compactions": self.compactions,
        }
