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

## Daytona environment

`--env daytona` runs each trial in a [Daytona](https://www.daytona.io/) sandbox. Environment kwargs (`--ek key=value`) specific to Daytona:

- `sandbox_labels` (or `labels`): dict of custom labels applied to every sandbox Presidio creates (all create paths: direct and Docker-in-Docker, image- and snapshot-based). Presidio always adds its own `presidio.owner-token` ownership label, which cannot be overridden; it is used to find and delete sandboxes leaked by interrupted create calls.
- `auto_stop_interval_mins` / `auto_delete_interval_mins`: sandbox lifecycle intervals in minutes, passed through to Daytona. Defaults are safe: auto-stop after 60 idle minutes and auto-delete immediately upon stopping. Passing `auto_stop_interval_mins=0` disables auto-stop; combining it with a negative `auto_delete_interval_mins` (auto-delete disabled) is honored but logs a warning, since leaked sandboxes would then persist until deleted manually.

Task `memory_mb`/`storage_mb` are converted to Daytona's whole-GiB resource units by rounding up, so a sandbox is never provisioned below what the task requests. Daytona HTTP 429 responses are retried natively, honoring the provider's `Retry-After` header when present instead of a generic short backoff.

## Agents

Select an agent with `--agent` and configure it with a YAML agent config — `model_name` for trial metadata, `env` for runtime variables, and agent-specific `kwargs`. See `presidio run --help`.

## License

Apache 2.0. Presidio includes code derived from [Harbor](https://github.com/harbor-framework/harbor), which is licensed under Apache 2.0; see `LICENSE` and `NOTICE`.
