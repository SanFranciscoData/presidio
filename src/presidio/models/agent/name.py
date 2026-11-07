from enum import Enum


class AgentName(str, Enum):
    ORACLE = "oracle"
    NOP = "nop"
    CLAUDE_CODE = "claude-code"
    CODEX = "codex"
    CURSOR_CLI = "cursor-cli"
    GEMINI_CLI = "gemini-cli"
    ANTIGRAVITY_CLI = "antigravity-cli"
    MINI_SWE_AGENT = "mini-swe-agent"
    OPENCODE = "opencode"
    TERMINUS = "terminus"
    TERMINUS_2 = "terminus-2"

    @classmethod
    def default_for_model(cls, model_name: str | None) -> "AgentName":
        if model_name and model_name.startswith(("gemini/", "google/", "gemini-")):
            return cls.ANTIGRAVITY_CLI
        return cls.ORACLE

    @classmethod
    def values(cls) -> set[str]:
        return {member.value for member in cls}
