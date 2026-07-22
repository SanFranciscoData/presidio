from presidio.analyze.prompt_quality import (
    DETERMINISTIC_GATES,
    PROMPT_QUALITY_FLAG_NAMES,
    PROMPT_QUALITY_GATE_NAMES,
    run_deterministic_gates,
)
from presidio.cli.quality_checker.models import load_rubric


def outcome(text: str, name: str) -> str:
    return run_deterministic_gates(text)[name].outcome.value


def test_prompt_length_gate():
    assert outcome("a" * 2000, "prompt_length") == "pass"
    assert outcome("a" * 2001, "prompt_length") == "fail"


def test_no_markdown_headers_gate():
    assert (
        outcome("Implement the service.\n\nUse /app/config.", "no_markdown_headers")
        == "pass"
    )
    assert (
        outcome("# Overview\nImplement the service.", "no_markdown_headers") == "fail"
    )
    assert (
        outcome("Overview\n========\nImplement the service.", "no_markdown_headers")
        == "fail"
    )
    assert outcome("```text\n# not a header\n---\n```", "no_markdown_headers") == "pass"


def test_no_template_labels_gate():
    assert (
        outcome("Implement the service using /app/config.", "no_template_labels")
        == "pass"
    )
    assert (
        outcome("**Requirements:** implement the service.", "no_template_labels")
        == "fail"
    )
    assert (
        outcome("The Requirements: field is part of the input.", "no_template_labels")
        == "pass"
    )


def test_no_canary_markers_gate():
    assert outcome("Implement the service.", "no_canary_markers") == "pass"
    assert outcome("harbor-canary must not appear.", "no_canary_markers") == "fail"
    assert outcome("<!-- canary: 123 -->", "no_canary_markers") == "fail"
    assert outcome("<!-- CANARY -->", "no_canary_markers") == "fail"
    assert (
        outcome("<!-- 123e4567-e89b-12d3-a456-426614174000 -->", "no_canary_markers")
        == "fail"
    )


def test_no_relative_paths_gate():
    assert outcome("Write to /app/data/output.json.", "no_relative_paths") == "pass"
    assert outcome("Write to ./data/output.json.", "no_relative_paths") == "fail"
    assert outcome("Read ../config/settings.toml.", "no_relative_paths") == "fail"
    assert outcome("Avoid /app/../tmp/output.", "no_relative_paths") == "fail"
    assert outcome("```sh\ncat ./example.txt\n```", "no_relative_paths") == "pass"
    assert (
        outcome("See https://example.test/../docs for context.", "no_relative_paths")
        == "pass"
    )


def test_advisory_flags():
    bullets = "- one\n- two\n- three"
    assert outcome(bullets, "avoid_bulleted_lists") == "fail"
    assert outcome("one two three", "avoid_bulleted_lists") == "pass"
    assert outcome(" ".join(["word"] * 351), "word_count_advisory") == "fail"
    assert outcome(" ".join(["word"] * 350), "word_count_advisory") == "pass"


def test_excessive_emphasis_flag():
    assert outcome("**one** and *two* and __three__", "excessive_emphasis") == "pass"
    assert (
        outcome(
            "**one** **two** **three** **four** **five**",
            "excessive_emphasis",
        )
        == "fail"
    )
    assert (
        outcome(
            "```md\n**one** **two** **three** **four** **five**\n```",
            "excessive_emphasis",
        )
        == "pass"
    )


def test_gate_classification_matches_rubric():
    prompt_rubric = load_rubric(
        __import__("pathlib").Path(
            "src/presidio/analyze/prompts/prompt_quality_rubric.toml"
        )
    )
    llm_names = {criterion.name for criterion in prompt_rubric.criteria}
    deterministic_names = {spec.name for spec in DETERMINISTIC_GATES}
    assert PROMPT_QUALITY_GATE_NAMES | PROMPT_QUALITY_FLAG_NAMES == (
        deterministic_names | llm_names
    )
    assert PROMPT_QUALITY_GATE_NAMES.isdisjoint(PROMPT_QUALITY_FLAG_NAMES)
