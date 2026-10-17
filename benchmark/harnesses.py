"""Provider routing and pinned CLI versions shared by the runner and adapters."""
import json

PROFILES = {
    "claude-code": {
        "version": "2.1.258", "adapter": "claude-code",
        "model": "qwen3.8-flash", "prefix": "",
        "base_url": "https://dashscope.aliyuncs.com/apps/anthropic",
        "key_env": "DASHSCOPE_API_KEY", "runtime_key": "ANTHROPIC_API_KEY",
        "runtime_url": "ANTHROPIC_BASE_URL", "protocol": "Anthropic Messages",
    },
    "codex": {
        "version": "0.153.4", "adapter": "agents:ApiCodex",
        "model": "gpt-5.6-sol", "prefix": "openai/",
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY", "runtime_key": "OPENAI_API_KEY",
        "runtime_url": "OPENAI_BASE_URL", "protocol": "OpenAI Responses",
    },
    "gemini-cli": {
        "version": "0.59.0", "adapter": "agents:ApiGemini",
        "model": "gemini-3.8-flash", "prefix": "google/",
        "base_url": "https://generativelanguage.googleapis.com",
        "key_env": "GEMINI_API_KEY", "runtime_key": "GEMINI_API_KEY",
        "runtime_url": "GOOGLE_GEMINI_BASE_URL", "protocol": "Gemini generateContent",
    },
}

SYSTEM_PROMPT = (
    "Work in this single agent session. Do not launch subagents or nested model clients. "
    "Follow the provided task instruction and complete law.py and explain.md. "
    "Do not access online task solutions."
)
SMOKE_PROMPT = (
    " This is a short end-to-end smoke test. Complete law.py and explain.md "
    "within eight tool calls. Fit a simple formula and submit promptly; "
    "do not run extended model selection or optimization loops."
)


def provider_environment(environ, agent, key, base_url, key_env, no_proxy=False):
    """Forward only the selected credential; do not inherit another CLI's auth."""
    env = {k: v for k, v in environ.items()
           if not k.startswith(("ANTHROPIC_", "OPENAI_", "CLAUDE_", "HARBOR_",
                                "CODEX_", "GEMINI_", "GOOGLE_"))
           and k not in ("DASHSCOPE_API_KEY", key_env)}
    profile = PROFILES[agent]
    env.update({profile["runtime_key"]: key, profile["runtime_url"]: base_url,
                "LITELLM_LOCAL_MODEL_COST_MAP": "True"})
    if no_proxy:
        env = {k: v for k, v in env.items() if not k.lower().endswith("_proxy")}
    return env


def agent_arguments(agent, root, base_url, effort, max_turns, smoke):
    profile = PROFILES[agent]
    args = ["--agent", profile["adapter"],
            "--agent-kwarg", "version=" + profile["version"],
            "--agent-kwarg", "reasoning_effort=" + effort]
    prompt = SYSTEM_PROMPT + (SMOKE_PROMPT if smoke else "")
    if agent == "claude-code":
        settings = json.loads((root / "settings.json").read_text(encoding="utf-8"))
        args += ["--agent-kwarg", f"config={root / 'settings.json'}",
                 "--agent-kwarg", "append_system_prompt=" + prompt,
                 "--agent-kwarg", "disallowed_tools=" + ",".join(settings["permissions"]["deny"])]
        if max_turns is not None:
            args += ["--agent-kwarg", f"max_turns={max_turns}"]
    elif agent == "codex":
        config = {
            "web_search": "disabled", "model_provider": "benchmark",
            "developer_instructions": prompt, "features": {"multi_agent": False},
            "model_providers": {"benchmark": {
                "name": "Benchmark API", "base_url": base_url,
                "env_key": "OPENAI_API_KEY", "wire_api": "responses",
                "requires_openai_auth": False,
                "request_max_retries": 0, "stream_max_retries": 0,
            }},
        }
        args += ["--agent-kwarg", "config=" + json.dumps(config)]
    else:
        args += ["--agent-kwarg", "benchmark_instruction=" + prompt]
        if max_turns is not None:
            args += ["--agent-kwarg", f"max_turns={max_turns}"]
    return args
