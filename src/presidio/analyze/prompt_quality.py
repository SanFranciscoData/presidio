"""Prompt quality gates and agentic prompt repair."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from presidio.analyze.backend import query_agent
from presidio.analyze.models import (
    QualityCheckModel,
    QualityCheckResult,
    build_check_response_model,
    build_criteria_guidance,
    load_rubric,
)
from presidio.models.task.paths import TaskPaths

import presidio.analyze

PROMPTS_DIR = Path(presidio.analyze.__file__).parent / "prompts"
PROMPT_QUALITY_RUBRIC = PROMPTS_DIR / "prompt_quality_rubric.toml"

PROMPT_QUALITY_GATE_NAMES = {
    "prompt_length",
    "no_markdown_headers",
    "no_template_labels",
    "no_canary_markers",
    "no_relative_paths",
    "no_persona_opener",
    "no_signoff",
    "no_suggestive_language",
    "no_eval_leakage",
    "no_verifier_description",
    "no_environment_narration",
    "outcome_orientation",
    "absolute_paths",
    "preserve_concrete_requirements",
    "no_pleasantries",
    "minimal_styling",
    "natural_not_procedural",
}
PROMPT_QUALITY_FLAG_NAMES = {
    "not_llm_synthesized",
    "realistic_scenario",
    "avoid_bulleted_lists",
    "word_count_advisory",
    "excessive_emphasis",
}


def _result(ok: bool, explanation: str) -> QualityCheckModel:
    return QualityCheckModel(outcome="pass" if ok else "fail", explanation=explanation)


def _in_fenced_code(text: str, index: int) -> bool:
    before = text[:index]
    return (
        sum(1 for line in before.splitlines() if line.lstrip().startswith("```")) % 2
        == 1
    )


def _no_headers(text: str) -> QualityCheckModel:
    lines = text.splitlines()
    in_fence = False
    for index, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if re.match(r"^\s*#{1,6}\s", line):
            return _result(False, "Markdown ATX header found.")
        if (
            line.strip()
            and index + 1 < len(lines)
            and re.match(r"^\s*(?:=+|-+)\s*$", lines[index + 1])
            and not in_fence
        ):
            return _result(False, "Markdown setext header found.")
    return _result(True, "No Markdown headers found.")


def _no_template_labels(text: str) -> QualityCheckModel:
    labels = (
        "Objective:",
        "Requirements:",
        "Deliverables:",
        "Steps:",
        "Acceptance Criteria:",
        "Verification:",
    )
    pattern = re.compile(
        r"^\s*(?:\*\*)?(?:" + "|".join(map(re.escape, labels)) + ")", re.I
    )
    line = next((line for line in text.splitlines() if pattern.match(line)), None)
    return _result(
        line is None,
        f"Template label found: {line.strip()}"
        if line
        else "No template labels found.",
    )


def _no_canary_markers(text: str) -> QualityCheckModel:
    comments = re.findall(r"<!--.*?-->", text, re.I | re.S)
    marker = re.search(r"harbor-canary|canary:", text, re.I)
    comment_canary = re.search(r"canary", " ".join(comments), re.I)
    guid = re.search(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
        " ".join(comments),
        re.I,
    )
    found = marker or comment_canary or guid
    return _result(
        not found,
        "No canary markers found." if not found else "Canary marker found.",
    )


def _no_relative_paths(text: str) -> QualityCheckModel:
    url_ranges = [match.span() for match in re.finditer(r"https?://\S+", text)]
    for match in re.finditer(r"(?<!\w)(?:\.\.?/)", text):
        if _in_fenced_code(text, match.start()):
            continue
        # URLs and protocol-relative URLs are not task paths.
        if any(start <= match.start() < end for start, end in url_ranges):
            continue
        return _result(False, f"Relative path token found: {match.group()}")
    return _result(True, "No relative path tokens found.")


def _bullet_flag(text: str) -> QualityCheckModel:
    count = sum(bool(re.match(r"^\s*[-*+]\s", line)) for line in text.splitlines())
    return _result(
        count <= 2, f"Found {count} markdown bullet lines (advisory limit is 2)."
    )


def _word_count(text: str) -> QualityCheckModel:
    count = len(re.findall(r"\b[\w'-]+\b", text))
    return _result(count <= 350, f"Word count: {count} (advisory limit is 350).")


def _emphasis_flag(text: str) -> QualityCheckModel:
    visible_lines: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            visible_lines.append(line)
    visible_text = "\n".join(visible_lines)
    spans = re.findall(
        r"\*\*[^*\n]+\*\*|__[^_\n]+__|(?<!\*)\*(?!\*)[^*\n]+\*(?!\*)|"
        r"(?<!\w)_(?!_)[^_\n]+(?<!\w)_(?!_)",
        visible_text,
    )
    count = len(spans)
    return _result(
        count <= 4,
        f"Found {count} bold/italic emphasis spans (advisory limit is 4).",
    )


@dataclass(frozen=True)
class GateSpec:
    name: str
    is_gate: bool
    fn: Callable[[str], QualityCheckModel]


DETERMINISTIC_GATES = [
    GateSpec(
        "prompt_length",
        True,
        lambda text: _result(
            len(text) <= 2000, f"Character count: {len(text)} (hard limit is 2000)."
        ),
    ),
    GateSpec("no_markdown_headers", True, _no_headers),
    GateSpec("no_template_labels", True, _no_template_labels),
    GateSpec("no_canary_markers", True, _no_canary_markers),
    GateSpec("no_relative_paths", True, _no_relative_paths),
    GateSpec("avoid_bulleted_lists", False, _bullet_flag),
    GateSpec("word_count_advisory", False, _word_count),
    GateSpec("excessive_emphasis", False, _emphasis_flag),
]


def run_deterministic_gates(instruction_text: str) -> dict[str, QualityCheckModel]:
    return {spec.name: spec.fn(instruction_text) for spec in DETERMINISTIC_GATES}


def _instruction_files(task_paths: TaskPaths) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    if task_paths.instruction_path.exists():
        files.append(("instruction.md", task_paths.instruction_path))
    if task_paths.has_configured_steps():
        try:
            import tomllib

            config = tomllib.loads(task_paths.config_path.read_text())
            for step in config.get("steps", []):
                name = step.get("name")
                if name:
                    path = task_paths.step_instruction_path(name)
                    if path.exists():
                        files.append((f"step {name}", path))
        except (OSError, tomllib.TOMLDecodeError):
            pass
    return files


def _merge_deterministic(
    all_results: list[tuple[str, dict[str, QualityCheckModel]]],
) -> dict[str, QualityCheckModel]:
    merged: dict[str, QualityCheckModel] = {}
    for spec in DETERMINISTIC_GATES:
        failures = [
            f"{label}: {result[spec.name].explanation}"
            for label, result in all_results
            if result[spec.name].outcome.value == "fail"
        ]
        merged[spec.name] = _result(
            not failures,
            "; ".join(failures)
            if failures
            else all_results[0][1][spec.name].explanation,
        )
    return merged


async def run_prompt_quality_check(
    task_dir: Path, model: str = "sonnet", verbose: bool = False
) -> QualityCheckResult:
    task_dir = Path(task_dir)
    if not task_dir.is_dir():
        raise FileNotFoundError(
            f"Task directory '{task_dir}' not found or is not a directory"
        )
    paths = TaskPaths(task_dir)
    files = _instruction_files(paths)
    if not files:
        raise ValueError(
            f"No instruction.md or configured step instructions found in '{task_dir}'"
        )
    deterministic = _merge_deterministic(
        [(label, run_deterministic_gates(path.read_text())) for label, path in files]
    )
    rubric = load_rubric(PROMPT_QUALITY_RUBRIC)
    response_model = build_check_response_model(rubric)
    template = (PROMPTS_DIR / "prompt_quality_check.txt").read_text()
    prompt = template.format(
        file_tree=_build_file_tree(task_dir),
        criteria_guidance=build_criteria_guidance(rubric),
    )
    response = await query_agent(
        prompt=prompt,
        model=model,
        cwd=str(task_dir),
        tools=["Read", "Glob", "Grep"],
        output_schema=response_model.model_json_schema(),
        verbose=verbose,
    )
    parsed = response_model.model_validate(response)
    return QualityCheckResult(checks={**deterministic, **parsed.model_dump()})


def _build_file_tree(task_dir: Path) -> str:
    files = [
        path.relative_to(task_dir).as_posix()
        for path in sorted(task_dir.rglob("*"))
        if path.is_file()
    ]
    return "\n".join(files) if files else "No files found"


def _snapshot(paths: list[Path]) -> dict[str, tuple[int, bytes]]:
    snapshot = {}
    for root in paths:
        if root.is_file():
            snapshot[str(root)] = (root.stat().st_mtime_ns, root.read_bytes())
        elif root.is_dir():
            for path in root.rglob("*"):
                if path.is_file():
                    snapshot[str(path)] = (path.stat().st_mtime_ns, path.read_bytes())
    return snapshot


def _gate_failures(result: QualityCheckResult) -> dict[str, QualityCheckModel]:
    return {
        name: check
        for name, check in result.checks.items()
        if name in PROMPT_QUALITY_GATE_NAMES and check.outcome.value == "fail"
    }


async def clean_prompt(
    task_dir: Path,
    model: str = "sonnet",
    max_rounds: int = 3,
    verbose: bool = False,
) -> tuple[QualityCheckResult, int, bool]:
    task_dir = Path(task_dir)
    paths = TaskPaths(task_dir)
    before = {str(path): path.read_bytes() for _, path in _instruction_files(paths)}
    protected = [
        paths.tests_dir,
        paths.solution_dir,
        paths.config_path,
        paths.environment_dir,
    ]
    result = await run_prompt_quality_check(task_dir, model, verbose)
    rounds = 0
    while rounds < max_rounds:
        failures = _gate_failures(result)
        if not failures:
            break
        rounds += 1
        failure_text = "\n".join(
            f"- {name}: {check.explanation}" for name, check in failures.items()
        )
        prompt = (
            (PROMPTS_DIR / "prompt_quality_fix.txt")
            .read_text()
            .format(
                failing_gates=failure_text,
            )
        )
        protected_snapshot = _snapshot(protected)
        await query_agent(
            prompt=prompt,
            model=model,
            cwd=str(task_dir),
            tools=["Read", "Glob", "Grep", "Edit"],
            verbose=verbose,
        )
        if _snapshot(protected) != protected_snapshot:
            raise RuntimeError(
                "Prompt fixer modified tests/, solution/, task.toml, or environment/"
            )
        result = await run_prompt_quality_check(task_dir, model, verbose)
    after = {str(path): path.read_bytes() for _, path in _instruction_files(paths)}
    return result, rounds, before != after
