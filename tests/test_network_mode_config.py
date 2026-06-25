"""Tests for Harbor-native network_mode / allowed_hosts on EnvironmentConfig.

Covers legacy allow_internet back-compat, explicit network_mode resolution,
allowed_hosts validation/normalization, and the BaseEnvironment capability
gate plus the resolved allowlist (the seam that lets a separate-verifier
environment reach an LLM judge's provider under deny-by-default egress).
"""

from pathlib import Path

import pytest

from presidio.environments.base import BaseEnvironment
from presidio.environments.capabilities import EnvironmentCapabilities
from presidio.models.agent.network import NetworkAllowlist
from presidio.models.environment_type import EnvironmentType
from presidio.models.task.config import EnvironmentConfig, NetworkMode, TaskOS
from presidio.models.trial.paths import TrialPaths


# --- resolve_network: legacy allow_internet back-compat ---------------------


def test_legacy_allow_internet_true_is_public():
    cfg = EnvironmentConfig(allow_internet=True)
    assert cfg.resolve_network() == (NetworkMode.PUBLIC, [])


def test_legacy_allow_internet_false_no_hosts_is_no_network():
    cfg = EnvironmentConfig(allow_internet=False)
    assert cfg.resolve_network() == (NetworkMode.NO_NETWORK, [])


def test_legacy_allow_internet_false_with_agent_hosts_is_allowlist():
    cfg = EnvironmentConfig(allow_internet=False)
    mode, hosts = cfg.resolve_network(["api.anthropic.com"])
    assert mode == NetworkMode.ALLOWLIST
    assert hosts == ["api.anthropic.com"]


# --- resolve_network: explicit network_mode --------------------------------


def test_explicit_public_ignores_hosts():
    cfg = EnvironmentConfig(network_mode=NetworkMode.PUBLIC)
    assert cfg.resolve_network(["api.anthropic.com"]) == (NetworkMode.PUBLIC, [])
    assert cfg.allow_internet is True


def test_explicit_no_network_ignores_agent_hosts():
    cfg = EnvironmentConfig(network_mode=NetworkMode.NO_NETWORK)
    # An agent's self-declared host must NOT re-open egress under no-network.
    assert cfg.resolve_network(["api.anthropic.com"]) == (NetworkMode.NO_NETWORK, [])
    assert cfg.allow_internet is False


def test_explicit_allowlist_unions_config_and_agent_hosts():
    cfg = EnvironmentConfig(
        network_mode=NetworkMode.ALLOWLIST,
        allowed_hosts=["api.anthropic.com"],
    )
    mode, hosts = cfg.resolve_network(["api.openai.com"])
    assert mode == NetworkMode.ALLOWLIST
    assert hosts == ["api.anthropic.com", "api.openai.com"]
    assert cfg.allow_internet is False


def test_allowlist_wildcard_normalized_to_squid_dstdomain():
    cfg = EnvironmentConfig(
        network_mode=NetworkMode.ALLOWLIST,
        allowed_hosts=["*.example.com"],
    )
    _, hosts = cfg.resolve_network()
    assert hosts == [".example.com"]


# --- allow_internet reconciliation -----------------------------------------


def test_network_mode_overrides_unset_allow_internet():
    cfg = EnvironmentConfig(network_mode=NetworkMode.NO_NETWORK)
    assert cfg.allow_internet is False


def test_contradictory_allow_internet_and_network_mode_rejected():
    with pytest.raises(ValueError, match="conflicts with allow_internet"):
        EnvironmentConfig(network_mode=NetworkMode.NO_NETWORK, allow_internet=True)
    with pytest.raises(ValueError, match="conflicts with allow_internet"):
        EnvironmentConfig(network_mode=NetworkMode.PUBLIC, allow_internet=False)


# --- allowed_hosts validation ----------------------------------------------


@pytest.mark.parametrize(
    "bad",
    ["https://api.openai.com", "api.openai.com/v1", "api.openai.com:443", "  "],
)
def test_allowed_hosts_rejects_non_hostnames(bad: str):
    with pytest.raises(ValueError):
        EnvironmentConfig(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=[bad])


# --- BaseEnvironment capability gate + resolved allowlist -------------------


class _StubEnv(BaseEnvironment):
    """Minimal environment whose capabilities are injected per-test."""

    _caps = EnvironmentCapabilities()

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.DOCKER

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return self._caps

    def _validate_definition(self):
        pass

    async def start(self, force_build: bool) -> None:  # pragma: no cover
        pass

    async def stop(self, delete: bool):  # pragma: no cover
        pass

    async def upload_file(self, source_path, target_path):  # pragma: no cover
        pass

    async def upload_dir(self, source_dir, target_dir):  # pragma: no cover
        pass

    async def download_file(self, source_path, target_path):  # pragma: no cover
        pass

    async def download_dir(self, source_dir, target_dir):  # pragma: no cover
        pass

    async def exec(self, command, **kwargs):  # pragma: no cover
        raise NotImplementedError


def _make_env(
    tmp_path: Path,
    cfg: EnvironmentConfig,
    caps: EnvironmentCapabilities,
    network_allowlist: NetworkAllowlist | None = None,
) -> _StubEnv:
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    _StubEnv._caps = caps
    return _StubEnv(
        environment_dir=tmp_path,
        environment_name="t",
        session_id="s",
        trial_paths=trial_paths,
        task_env_config=cfg,
        network_allowlist=network_allowlist,
    )


def test_allowlist_rejected_without_capability(tmp_path: Path):
    cfg = EnvironmentConfig(
        os=TaskOS.LINUX,
        network_mode=NetworkMode.ALLOWLIST,
        allowed_hosts=["api.anthropic.com"],
    )
    with pytest.raises(ValueError, match="network_mode='allowlist'"):
        _make_env(tmp_path, cfg, EnvironmentCapabilities(disable_internet=True))


def test_no_network_rejected_without_capability(tmp_path: Path):
    cfg = EnvironmentConfig(os=TaskOS.LINUX, network_mode=NetworkMode.NO_NETWORK)
    with pytest.raises(ValueError, match="network_mode='no-network'"):
        _make_env(tmp_path, cfg, EnvironmentCapabilities())


def test_separate_verifier_allowlist_comes_from_config(tmp_path: Path):
    # No agent (network_allowlist=None), mirroring a separate verifier env.
    # The judge's provider host must come from [verifier.environment].allowed_hosts.
    cfg = EnvironmentConfig(
        os=TaskOS.LINUX,
        network_mode=NetworkMode.ALLOWLIST,
        allowed_hosts=["api.anthropic.com"],
    )
    env = _make_env(
        tmp_path,
        cfg,
        EnvironmentCapabilities(disable_internet=True, network_allowlist=True),
    )
    assert env.network_allowlist.domains == ["api.anthropic.com"]


def test_agent_hosts_union_with_config_hosts(tmp_path: Path):
    cfg = EnvironmentConfig(
        os=TaskOS.LINUX,
        network_mode=NetworkMode.ALLOWLIST,
        allowed_hosts=["api.anthropic.com"],
    )
    env = _make_env(
        tmp_path,
        cfg,
        EnvironmentCapabilities(disable_internet=True, network_allowlist=True),
        network_allowlist=NetworkAllowlist(domains=["api.openai.com"]),
    )
    assert env.network_allowlist.domains == ["api.anthropic.com", "api.openai.com"]
