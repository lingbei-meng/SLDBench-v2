"""Run SLDBench through Harbor with Claude Code, Codex or Gemini CLI."""
import argparse
import csv
import getpass
import json
import math
import os
import platform
import re
import subprocess
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from .prepare import IMAGE, ROOT, RUNS, TASKS, materialize
from .harnesses import PROFILES, agent_arguments, provider_environment

DEFAULT_TASK = "colpret_ibm_model_token_openbookqa_accuracy_scaling"


def summarize(folder):
    names = {}
    for definition in TASKS.glob("*/task.toml"):
        name = tomllib.loads(definition.read_text(encoding="utf-8"))["task"]["name"]
        names[name] = definition.parent.name
    rows = []
    for path in sorted(folder.glob("*/result.json")):
        result = json.loads(path.read_text())
        rewards = (result.get("verifier_result") or {}).get("rewards") or {}
        metrics_path = path.parent / "verifier" / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        reward_path = path.parent / "verifier" / "reward.json"
        raw_reward = json.loads(reward_path.read_text()) if reward_path.exists() else {}
        agent = result.get("agent_result") or {}
        task = names.get(result.get("task_name"), result.get("task_name", path.parent.name))
        rows.append({
            "task": task.removesuffix("_costaware"),
            "setting": "active" if task.endswith("_costaware") else "passive",
            "trial": path.parent.name,
            "error": (result.get("exception_info") or {}).get("exception_type", ""),
            **rewards,
            "r2": metrics.get("r2", rewards.get("r2", rewards.get("reward", rewards.get("oracle_reward")))),
            "grader_error": metrics.get("error") or raw_reward.get("error", ""),
            "input_tokens": agent.get("n_input_tokens"),
            "cache_tokens": agent.get("n_cache_tokens"),
            "output_tokens": agent.get("n_output_tokens"),
        })
    (folder / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    if keys:
        with (folder / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Results: {folder}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=PROFILES, default="claude-code")
    parser.add_argument("--model", help="Model ID; default depends on --agent")
    parser.add_argument("--base-url", help="Compatible HTTPS API endpoint; default depends on --agent")
    parser.add_argument("--api-key-env", help="Environment variable containing the provider key; otherwise prompt without echo")
    parser.add_argument("--no-proxy", action="store_true", help="Ignore inherited proxy environment variables for this run")
    parser.add_argument("--task", action="append", help="Base task name; repeat for multiple tasks")
    parser.add_argument("--all", action="store_true", help="Run all 30 task pairs")
    parser.add_argument("--setting", choices=["passive", "active", "both"], default="both")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--smoke", action="store_true", help="Short end-to-end plumbing test, not a benchmark score run")
    parser.add_argument("--effort", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--max-turns", type=int, help="Optional smoke-test turn limit; unset for full runs")
    parser.add_argument("--agent-seconds", type=int, default=3600)
    parser.add_argument("--name", help="Unique output folder name")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print non-secret configuration; no API call")
    args = parser.parse_args()
    profile = PROFILES[args.agent]
    args.model = args.model or profile["model"]
    args.base_url = args.base_url or profile["base_url"]
    args.api_key_env = args.api_key_env or profile["key_env"]
    if not re.fullmatch(r"[A-Za-z0-9._:/\[\]-]+", args.model):
        parser.error("model must be a model identifier without spaces or shell characters")
    if args.agent == "codex" and args.max_turns is not None:
        parser.error("Codex CLI has no max-turns option; use --agent-seconds")
    if args.smoke:
        if args.all or args.repeats != 1:
            parser.error("--smoke runs one repetition of selected tasks; do not combine with --all")
        args.effort = "low"
        args.max_turns = None if args.agent == "codex" else 12
        args.agent_seconds = 900
    manifest = json.loads((ROOT / "tasks-manifest.json").read_text())
    bases = sorted(t for t in manifest["tasks"] if not t.endswith("_costaware"))
    if args.list:
        print("\n".join(bases))
        return
    if args.all and args.task:
        parser.error("Use --all or --task, not both")
    if min(args.repeats, args.concurrency, args.agent_seconds) < 1:
        parser.error("Repeats, concurrency and agent-seconds must be positive")
    if args.max_turns is not None and args.max_turns < 1:
        parser.error("max-turns must be positive")
    chosen = bases if args.all else (args.task or [DEFAULT_TASK])
    unknown = set(chosen) - set(bases)
    if unknown:
        parser.error(f"Unknown task(s): {sorted(unknown)}; use --list")
    tasks = [base + suffix for base in dict.fromkeys(chosen)
             for suffix in ([""] if args.setting == "passive" else
                            ["_costaware"] if args.setting == "active" else ["", "_costaware"])]
    job_name = args.name or datetime.now().strftime("%Y%m%d_%H%M%S")
    if Path(job_name).name != job_name or job_name in (".", ".."):
        parser.error("name must be a simple folder name")
    endpoint = urlparse(args.base_url)
    if endpoint.scheme != "https" or not endpoint.hostname or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        parser.error(f"base-url must be an HTTPS {profile['protocol']}-compatible endpoint without credentials")
    folder = RUNS / job_name
    if folder.exists():
        parser.error(f"Output already exists: {folder}; choose another --name")
    command = [sys.executable, str(ROOT / "runtime.py"), "run",
               "--path", str(TASKS), "--jobs-dir", str(RUNS),
               "--job-name", job_name, "--env", "runtime:BenchmarkDockerEnvironment",
               "--model", (args.model if not profile["prefix"] or args.model.startswith(profile["prefix"])
                            else profile["prefix"] + args.model),
               "--n-concurrent", str(args.concurrency), "--n-attempts", str(args.repeats),
               "--max-retries", "0", "--quiet", "--yes",
               "--allow-agent-host", endpoint.hostname,
               "--agent-timeout-multiplier", str(args.agent_seconds / 3600)]
    command.extend(agent_arguments(args.agent, ROOT, args.base_url, args.effort, args.max_turns, args.smoke))
    for task in tasks:
        command.extend(["--include-task-name", task])
    configuration = {"agent": args.agent, "agent_version": profile["version"],
                     "protocol": profile["protocol"],
                     "model": args.model, "base_url": args.base_url, "tasks": tasks,
                     "repeats": args.repeats, "concurrency": args.concurrency,
                     "smoke": args.smoke, "effort": args.effort,
                     "max_turns": args.max_turns, "agent_seconds": args.agent_seconds,
                     "command": command}
    if args.dry_run:
        print(json.dumps(configuration, indent=2))
        return
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        parser.error("Run on Linux x86_64 / WSL2 with Docker")
    if sys.version_info < (3, 12):
        parser.error("The benchmark runtime requires Python 3.12 or later")
    materialize()
    check = subprocess.run(["docker", "image", "inspect", IMAGE],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if check.returncode:
        parser.error("First run: python -m benchmark.prepare")
    key = os.environ.get(args.api_key_env) or getpass.getpass("Provider API key (not saved): ")
    if not key.strip():
        parser.error("API key is empty")
    env = provider_environment(os.environ, args.agent, key, args.base_url, args.api_key_env, args.no_proxy)
    status = subprocess.run(command, env=env, cwd=ROOT).returncode
    if folder.exists():
        (folder / "run-settings.json").write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
        rows = summarize(folder)
        if len(rows) != len(tasks) * args.repeats or any(
            r["error"] or r["grader_error"] or not isinstance(r["r2"], (int, float))
            or not math.isfinite(r["r2"]) for r in rows
        ):
            status = status or 1
    else:
        status = status or 1
    raise SystemExit(status)


if __name__ == "__main__":
    main()
