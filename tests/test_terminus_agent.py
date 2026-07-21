from presidio.agents.terminus import PresidioTmuxSession, Terminus2Agent, TerminusAgent


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
