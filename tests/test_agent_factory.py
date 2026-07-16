from presidio.agents.factory import AgentFactory
from presidio.models.agent.name import AgentName


def test_agent_names_exactly_match_registered_agents():
    """Advertised AgentName values must match the registered agents 1:1.

    Guards against the enum advertising a name (e.g. the removed swe-agent /
    taiga) that AgentFactory cannot construct: such names pass the
    ``config.name in AgentName.values()`` gate and then fail late at
    construction with a confusing "Unknown agent type" error.
    """
    registered_names = {agent.name() for agent in AgentFactory._AGENTS}

    assert AgentName.values() == registered_names


def test_every_agent_name_is_in_the_factory_map():
    for name in AgentName:
        assert name in AgentFactory._AGENT_MAP
