from pathlib import Path

from presidio.job import Job
from presidio.models.job.config import JobConfig


def test_resume_ignores_attempts_directory(tmp_path: Path):
    config = JobConfig(job_name="job", jobs_dir=tmp_path)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "config.json").write_text(config.model_dump_json())

    attempts_dir = job_dir / "attempts"
    attempts_dir.mkdir()
    (attempts_dir / "result.json").write_text("not a trial result")

    job = object.__new__(Job)
    job.config = config
    job._maybe_init_existing_job()

    assert job._existing_trial_configs == []
    assert job._existing_trial_results == []
