# Presidio

Presidio is a [Harbor](https://www.harborframework.com/docs)-compatible harness for evaluating coding agents in sandboxed environments. It reads Harbor's task format and runs trials against it, adding a native per-phase network-policy model, support for agents that install and run inside the sandbox, and task-defined sandbox lifetimes.

## Install

Presidio is not published to PyPI; install it from source.

```bash
uv tool install "git+https://github.com/SanFranciscoData/presidio.git"
# or from a local clone:
git clone https://github.com/SanFranciscoData/presidio.git
uv tool install ./presidio
```

## Run

Point Presidio at a task (or a directory of tasks), pick an agent and a sandbox environment, and load your secrets from an env file:

```bash
presidio run -p path/to/task --agent <agent> --env <environment> --env-file .env
```

Run a dataset, optionally a deterministic random subset:

```bash
presidio run -p path/to/dataset --agent <agent> --env <environment>
presidio run -p path/to/dataset --n-tasks 10 --sample-seed 0
```

Results are written under `jobs/<name>/<trial_id>/`.

### Gemini models (Antigravity CLI by default)

Google's native [Antigravity CLI](https://antigravity.google/product/antigravity-cli/)
is the default when a Gemini model is supplied without an agent:

```bash
presidio run -p path/to/task \
  -m gemini/gemini-3.8-flash --env-file .env
```

The equivalent job configuration is:

```json
{"agents": [{"model_name": "gemini/gemini-3.8-flash"}]}
```

This applies to `gemini/*`, `google/*`, and bare `gemini-*` model names.
Explicit agent names and custom import paths are never replaced: use
`-a gemini-cli` (or `"name": "gemini-cli"` in job config) for the legacy CLI.
You can also explicitly select `-a antigravity-cli`. Other models and runs
without a model retain the `oracle` fallback.

Set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) for headless authentication.
`GOOGLE_GEMINI_BASE_URL` can select a Gemini-compatible endpoint. The native
`agy` binary is installed using Google's official latest-release installer;
version pinning, interactive OAuth, ADC, and native session resume are not supported
by this integration. Use `--ak reasoning_effort=high` to pass an explicit effort
(`low`, `medium`, `high`, `xhigh`, or `max`, subject to model support).
Task-provided skills and MCP servers are configured in Antigravity's native paths.
Separate streaming JSON and stderr logs are retained; ATIF trajectories include
tool calls, responses, and cache-inclusive token counts. Costs use LiteLLM pricing
when available and remain unset for unknown models.

Run `presidio run --help` for the full flag list and the available agents and environments. Other commands: `presidio job`, `presidio critique`, `presidio check`, `presidio analyze`.

## Tasks

Tasks are standard Harbor `task.toml` packages. Network access is declared per task and per phase:

```toml
[environment]
network_mode = "no-network"          # "public" | "no-network" | "allowlist"

[verifier.environment]
network_mode = "allowlist"
allowed_hosts = ["api.example.com"]
```

`allowlist` is what lets a verifier reach an external service under deny-by-default egress. On sandbox backends that support it, `[agent]`/`[verifier]` may override the policy per phase on a running sandbox.

## Environments

Pick a sandbox backend with `--env` and pass backend-specific options with `--ek key=value`. For Daytona, sandboxes auto-stop after 60 idle minutes and are deleted on stop by default (override with `--ek auto_stop_interval_mins=... --ek auto_delete_interval_mins=...`). Custom labels can be attached with `--ek sandbox_labels={"k":"v"}`.

## Agents

Select an agent with `--agent` and configure it with a YAML agent config — `model_name` for trial metadata, `env` for runtime variables, and agent-specific `kwargs`. See `presidio run --help`.

## License

Apache 2.0. Presidio includes code derived from [Harbor](https://github.com/harbor-framework/harbor), which is licensed under Apache 2.0; see `LICENSE` and `NOTICE`.
