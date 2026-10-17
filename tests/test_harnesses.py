"""Verify framework routing, protocol selection and credential isolation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark.harnesses import PROFILES, provider_environment


class HarnessTests(unittest.TestCase):
    def dry_run(self, agent, *args):
        return subprocess.run([sys.executable, str(ROOT / "run.py"),
                               "--agent", agent, "--dry-run", *args],
                              text=True, capture_output=True)

    def test_all_frameworks_route_to_correct_protocol_and_version(self):
        for agent, profile in PROFILES.items():
            with self.subTest(agent=agent):
                result = self.dry_run(agent, "--all", "--repeats", "3")
                self.assertEqual(result.returncode, 0, result.stderr)
                config = json.loads(result.stdout)
                self.assertEqual(config["agent_version"], profile["version"])
                self.assertEqual(config["protocol"], profile["protocol"])
                command = config["command"]
                self.assertEqual(command[command.index("--agent") + 1], profile["adapter"])
                self.assertEqual(len(config["tasks"]) * config["repeats"], 180)

    def test_credentials_do_not_cross_framework_boundaries(self):
        inherited = {"PATH": "/bin", "OPENAI_API_KEY": "old-openai",
                     "GEMINI_API_KEY": "old-google", "ANTHROPIC_API_KEY": "old-anthropic",
                     "CODEX_AUTH_JSON_PATH": "/private/auth.json",
                     "GOOGLE_APPLICATION_CREDENTIALS": "/private/oauth.json",
                     "GEMINI_FORCE_OAUTH": "true", "CUSTOM_KEY": "chosen-key",
                     "HTTP_PROXY": "http://proxy"}
        for agent, profile in PROFILES.items():
            env = provider_environment(inherited, agent, "chosen-key", "https://example.org",
                                       "CUSTOM_KEY", no_proxy=True)
            self.assertEqual(env[profile["runtime_key"]], "chosen-key")
            self.assertEqual(env[profile["runtime_url"]], "https://example.org")
            self.assertEqual(set(env), {"PATH", profile["runtime_key"], profile["runtime_url"],
                                        "LITELLM_LOCAL_MODEL_COST_MAP"})

    def test_dry_run_does_not_read_or_print_credentials(self):
        marker = "private-test-value-do-not-print"
        result = subprocess.run([sys.executable, str(ROOT / "run.py"), "--agent", "codex", "--dry-run"],
                                env={**os.environ, "OPENAI_API_KEY": marker}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertNotIn(marker, result.stdout + result.stderr)

    def test_codex_custom_provider_uses_responses(self):
        result = self.dry_run("codex", "--base-url", "https://example.org/v1", "--model", "custom-model")
        config = json.loads(result.stdout)
        raw = next(x.removeprefix("config=") for x in config["command"] if x.startswith("config="))
        provider = json.loads(raw)["model_providers"]["benchmark"]
        self.assertEqual(provider["wire_api"], "responses")
        self.assertEqual(provider["base_url"], "https://example.org/v1")
        self.assertFalse(provider["requires_openai_auth"])

    def test_turn_limits_are_not_silently_ignored(self):
        self.assertNotEqual(self.dry_run("codex", "--max-turns", "5").returncode, 0)
        for agent in PROFILES:
            config = json.loads(self.dry_run(agent, "--smoke").stdout)
            self.assertEqual(config["max_turns"], None if agent == "codex" else 12)
            self.assertEqual(config["agent_seconds"], 900)

    def test_unsafe_model_and_credential_in_url_rejected(self):
        for agent in PROFILES:
            for args in [("--model", "foo;env"), ("--base-url", "https://user:secret@example.org")]:
                self.assertNotEqual(self.dry_run(agent, *args).returncode, 0)


if __name__ == "__main__":
    unittest.main()
