from pathlib import Path
from typing import Annotated

import yaml
from typer import Option

from presidio.cli.jobs import console
from presidio.cli.utils import run_async
from presidio.models.job.config import JobConfig
from presidio.preflight import run_preflight


def preflight_command(
    config_path: Annotated[
        Path,
        Option(
            "--config",
            "-c",
            help="Path to a job configuration in YAML or JSON format.",
        ),
    ],
) -> None:
    """Check model credentials and egress access before running a job."""
    if config_path.suffix == ".yaml":
        config = JobConfig.model_validate(yaml.safe_load(config_path.read_text()))
    elif config_path.suffix == ".json":
        config = JobConfig.model_validate_json(config_path.read_text())
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    run_async(
        run_preflight(
            config,
            include_egress=True,
            output=console.print,
        )
    )
