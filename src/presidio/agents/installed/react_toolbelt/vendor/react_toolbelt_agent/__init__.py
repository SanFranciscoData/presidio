"""ReAct Toolbelt agent — container-side program.

Ported from Mercor-Intelligence/archipelago @ 2af3e972
(agents/runner/agents/react_toolbelt_agent/). This copy is self-contained:
the archipelago runner's settings/metrics/Datadog/Redis plumbing is removed,
and only the pieces the agent loop actually needs (LLM call hardening, MCP
content conversion, error classification, usage totals) are vendored.

Runtime dependencies: litellm, fastmcp, loguru (installed into the task
container by the presidio ReactToolbelt wrapper's install spec).

Intentional omissions vs. upstream:
- No remote-image fetching / Anthropic image downscaling (upstream
  ``image_fetch.py``): MCP tool images arrive as base64 data URIs and are
  passed through unchanged.
- No per-token output breakdown (upstream ``UsageTracker`` breakdown mode);
  totals and per-call usage are kept so trajectory metrics still work.
- No Responses API path; Chat Completions only.
"""

VERSION = "0.1.0"
