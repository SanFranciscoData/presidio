import json
from types import SimpleNamespace

from presidio.agents.terminus import PresidioTmuxSession, Terminus2Agent, TerminusAgent
from presidio.models.trajectories import Trajectory


def _session() -> PresidioTmuxSession:
    return PresidioTmuxSession(
        environment=None,
        loop=None,
        session_name="test",
        exec_timeout_sec=1,
        logger=__import__("logging").getLogger(__name__),
    )


def test_prepare_keys():
    session = _session()
    keys, blocking = session._prepare_keys("ls\n", block=True)
    assert keys == ["ls", "; tmux wait -S done", "Enter"]
    assert blocking is True

    keys, blocking = session._prepare_keys("C-c", block=True)
    assert keys == ["C-c"]
    assert blocking is False


def test_find_new_content():
    session = _session()
    session._previous_buffer = "first\nsecond"
    assert session._find_new_content("first\nsecond\nthird") == "\nsecond\nthird"
    assert session._find_new_content("unrelated") is None


def test_agent_names_and_install_spec(tmp_path):
    assert TerminusAgent.name() == "terminus"
    assert Terminus2Agent.name() == "terminus-2"

    spec = TerminusAgent(
        logs_dir=tmp_path,
        model_name="anthropic/x",
    ).install_spec()
    assert spec.agent_name == "terminus"
    assert "tmux" in spec.steps[0].run


def test_render_instruction(tmp_path):
    template_path = tmp_path / "prompt_template.txt"
    template_path.write_text("Marker text\n{{ instruction }}")
    agent = TerminusAgent(
        logs_dir=tmp_path,
        model_name="anthropic/x",
        prompt_template_path=template_path,
    )

    rendered = agent._render_instruction("Original instruction")

    assert "Marker text" in rendered
    assert "Original instruction" in rendered


def test_build_trajectory_from_episode_logs(tmp_path):
    agent = TerminusAgent(
        logs_dir=tmp_path,
        model_name="anthropic/x",
    )
    usage_by_episode = (
        (
            7,
            {
                "input_tokens": 11,
                "cache_creation_input_tokens": 2,
                "cache_read_input_tokens": 3,
                "output_tokens": 3,
            },
        ),
        (
            2,
            {
                "prompt_tokens": 17,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 4},
            },
        ),
    )
    for episode_number, usage in usage_by_episode:
        episode_dir = tmp_path / f"episode-{episode_number}"
        episode_dir.mkdir()
        (episode_dir / "response.json").write_text(
            json.dumps(
                {
                    "state_analysis": f"State {episode_number}",
                    "explanation": f"Explain {episode_number}",
                    "commands": [
                        {
                            "keystrokes": "pwd\n",
                            "is_blocking": True,
                            "timeout_sec": 5.0,
                        }
                    ],
                }
            )
        )
        (episode_dir / "debug.json").write_text(
            json.dumps(
                {
                    "start_time": "2026-01-02T03:04:05",
                    "original_response": json.dumps({"usage": usage}),
                }
            )
        )

    result = SimpleNamespace(total_input_tokens=999, total_output_tokens=888)
    trajectory = agent._write_trajectory(result)

    assert trajectory is not None
    assert [step.step_id for step in trajectory.steps] == [1, 2]
    assert [step.message for step in trajectory.steps] == [
        "State 2\nExplain 2",
        "State 7\nExplain 7",
    ]
    assert trajectory.final_metrics.total_prompt_tokens == 33
    assert trajectory.final_metrics.total_completion_tokens == 8
    assert trajectory.final_metrics.total_cached_tokens == 7
    assert [
        (
            step.metrics.prompt_tokens,
            step.metrics.completion_tokens,
            step.metrics.cached_tokens,
        )
        for step in trajectory.steps
        if step.metrics is not None
    ] == [(17, 5, 4), (16, 3, 3)]
    serialized = trajectory.to_json_dict()
    round_tripped = Trajectory.model_validate_json(json.dumps(serialized))
    assert len(round_tripped.steps) == 2
    assert (tmp_path / "trajectory.json").exists()


def test_build_trajectory_falls_back_to_aggregate_tokens_when_metrics_missing(
    tmp_path,
):
    agent = TerminusAgent(
        logs_dir=tmp_path,
        model_name="anthropic/x",
    )
    episode_dir = tmp_path / "episode-1"
    episode_dir.mkdir()
    (episode_dir / "response.json").write_text(
        json.dumps(
            {
                "state_analysis": "State 1",
                "explanation": "Explain 1",
                "commands": [
                    {
                        "keystrokes": "pwd\n",
                        "is_blocking": True,
                        "timeout_sec": 5.0,
                    }
                ],
            }
        )
    )
    (episode_dir / "debug.json").write_text(
        json.dumps(
            {
                "start_time": "2026-01-02T03:04:05",
                "original_response": json.dumps({"usage": {}}),
            }
        )
    )

    result = SimpleNamespace(total_input_tokens=12, total_output_tokens=4)
    trajectory = agent._write_trajectory(result)

    assert trajectory is not None
    assert trajectory.steps[0].metrics is None
    assert trajectory.final_metrics.total_prompt_tokens == 12
    assert trajectory.final_metrics.total_completion_tokens == 4
    assert trajectory.final_metrics.total_cached_tokens is None
