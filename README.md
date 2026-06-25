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

## Agents

Select an agent with `--agent` and configure it with a YAML agent config — `model_name` for trial metadata, `env` for runtime variables, and agent-specific `kwargs`. See `presidio run --help`.

## License

Apache 2.0. Presidio includes code derived from [Harbor](https://github.com/harbor-framework/harbor), which is licensed under Apache 2.0; see `LICENSE` and `NOTICE`.
