"""API-only Codex and Gemini CLI adapters derived from the experiment runners.

Harbor provides each CLI's execution and trajectory parser. This module enforces
the benchmark's isolated HOME, pinned installation and single-session settings.
"""
import shlex
from pathlib import PurePosixPath

from harbor.agents.installed.codex import Codex
from harbor.agents.installed.gemini_cli import GeminiCli


class Isolated:
    cli = ""

    async def exec_as_agent(self, environment, command, env=None, **kwargs):
        isolated = dict(env or {}, HOME="/opt/sld-agent-home",
                        XDG_CONFIG_HOME="/opt/sld-agent-home/.config")
        command = "export HOME=/opt/sld-agent-home; " + command
        if "| tee " in command or "| stdbuf " in command:
            command = "set -o pipefail; " + command
        return await super().exec_as_agent(environment, command, env=isolated, **kwargs)

    async def install(self, environment):
        user = environment.default_user or "root"
        await self.exec_as_root(environment,
            "test ! -e /opt/sld-agent-home && mkdir -m 700 /opt/sld-agent-home && chown "
            + shlex.quote(str(user)) + " /opt/sld-agent-home")
        result = await self.exec_as_agent(environment, self.cli + " --version")
        actual = self.parse_version(result.stdout or "")
        (self.logs_dir / "pinned-version.txt").write_text(result.stdout or "", encoding="utf-8")
        if result.return_code != 0 or actual != self._version:
            raise RuntimeError(f"{self.cli} version mismatch: expected {self._version}, got {actual!r}; rebuild the runtime image")


class ApiCodex(Isolated, Codex):
    cli = "codex"
    _REMOTE_CODEX_HOME = PurePosixPath("/opt/sld-agent-home/.codex")

    def _resolve_auth_json_path(self):
        if self._get_env("CODEX_AUTH_JSON_PATH") or self._get_env("CODEX_FORCE_AUTH_JSON"):
            raise ValueError("Benchmark runs use API keys, not personal Codex login files")
        return None


class ApiGemini(Isolated, GeminiCli):
    cli = "gemini"

    def __init__(self, *args, max_turns=None, benchmark_instruction="", **kwargs):
        self._benchmark_max_turns = int(max_turns) if max_turns is not None else None
        self._benchmark_instruction = benchmark_instruction
        super().__init__(*args, **kwargs)

    def _resolve_oauth_creds_path(self):
        if self._get_env("GEMINI_OAUTH_CREDS_PATH") or self._get_env("GEMINI_FORCE_OAUTH"):
            raise ValueError("Benchmark runs use API keys, not personal Gemini login files")
        return None

    def _build_settings_config(self, model=None, auth_type=None):
        config, alias = super()._build_settings_config(model, auth_type)
        config = config or {}
        config.setdefault("experimental", {}).update(
            enableAgents=False, skills=False, dynamicModelConfiguration=True)
        config["tools"] = {"exclude": ["google_web_search", "web_fetch", "delegate_to_agent"]}
        config["advanced"] = {"ignoreLocalEnv": True}
        config["telemetry"] = {"enabled": False}
        config["security"] = {"auth": {"selectedType": "gemini-api-key"}}
        if self._benchmark_max_turns is not None:
            config.setdefault("model", {})["maxSessionTurns"] = self._benchmark_max_turns
        return config, alias

    async def run(self, instruction, environment, context):
        if self._benchmark_instruction:
            instruction += "\n\n" + self._benchmark_instruction
        return await super().run(instruction, environment, context)
