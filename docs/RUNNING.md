# Running SLDBench v2

The executable benchmark contains the 30 evaluation tasks, each in Passive and Active settings. The full data library contains 127 datasets, 60 of which include acquisition catalogues. The runner uses Harbor 0.22.0 and supports three agent frameworks through the same task and scoring pipeline.

| `--agent` | Pinned CLI | API protocol | Default credential variable |
|---|---|---|---|
| `claude-code` (default) | Claude Code 2.1.258 | Anthropic Messages | `DASHSCOPE_API_KEY` |
| `codex` | Codex 0.153.4 | OpenAI Responses | `OPENAI_API_KEY` |
| `gemini-cli` | Gemini CLI 0.59.0 | Gemini generateContent | `GEMINI_API_KEY` |

## Installation

Use Linux x86_64 with Python 3.12, Docker Engine, and Docker Compose v2. Docker must be accessible to the current user. Windows users can run inside WSL2. No GPU is required; experiments are replayed from recorded measurements.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-benchmark.txt
python tools/validate.py
python -m benchmark.prepare
```

Preparation verifies task inputs against checksums, assembles isolated build contexts, and builds the runtime image. Official CLI packages are downloaded with integrity verification; Python dependencies and base-image digests are pinned. Preparation checks all three CLI versions without network access. The first build needs network access and several GB of Docker storage. Allow approximately 4 GB of memory per concurrent agent, with additional capacity for the verifier and host.

For data-only validation on any platform:

```bash
python -m benchmark.prepare --data-only
```

## Credentials and provider

By default, the runner uses `qwen3.8-flash` at `https://dashscope.aliyuncs.com/apps/anthropic`. It reads `DASHSCOPE_API_KEY`, or prompts without echo if the variable is unset. Keys are passed through the process environment, not command arguments or configuration files.

For another Anthropic-compatible provider, set its credential in an environment variable and pass the endpoint and model:

```bash
python run.py --model YOUR_MODEL --base-url https://YOUR_PROVIDER/anthropic \
  --api-key-env YOUR_PROVIDER_API_KEY --name provider_run
```

Each endpoint must support the selected framework's native protocol, including tool calls. A Chat Completions-only endpoint is insufficient for Codex or Gemini CLI. API credentials are read from the selected variable or an interactive prompt; personal Codex or Gemini login files are not imported.

```bash
# Codex: default model gpt-5.6-sol, endpoint https://api.openai.com/v1.
python run.py --agent codex --smoke --name codex_smoke

# Gemini CLI: default model gemini-3.8-flash, Google Gemini API endpoint.
python run.py --agent gemini-cli --smoke --name gemini_smoke

# A provider offering the native Gemini API; keys stay in the environment.
python run.py --agent gemini-cli --model YOUR_MODEL \
  --base-url https://YOUR_PROVIDER --api-key-env YOUR_PROVIDER_API_KEY \
  --smoke --name gemini_provider_smoke
```

Codex custom endpoints normally end in `/v1`; Gemini endpoints are the root before `/v1beta/models/...`. Model access depends on the API account. All frameworks support both settings, repeated runs, concurrency, result summaries and best@3 aggregation. These commands run new experiments; published result tables remain unchanged.

If inherited proxy variables refer to an unavailable proxy, add `--no-proxy` to preparation and execution. This affects only the launched process environment.

## Commands

```bash
# One task in both settings: check API access, acquisition, submission and grading.
python run.py --smoke --name smoke

# A chosen task, Active only, three repeats.
python run.py --task colpret_ibm_model_token_openbookqa_accuracy_scaling \
  --setting active --repeats 3 --name openbookqa_active

# All 30 evaluation tasks, both settings, three repeats each.
python run.py --all --setting both --repeats 3 --concurrency 2 --name full

# Inspect tasks and the complete command without model calls.
python run.py --list
python run.py --all --repeats 3 --dry-run

# Aggregate a completed run; best@3 is included when repeats = 3.
python tools/summarize_runs.py runs/full
```

`--task` can be repeated. The default without `--all` or `--task` is the OpenBookQA task shown above, in both settings. Output names must be unique; existing results are never overwritten.

| Option | Default |
|---|---|
| `--setting` | `both` |
| `--repeats` | `1` |
| `--concurrency` | `1` |
| `--effort` | `high` |
| `--agent-seconds` | `3600` |
| `--max-turns` | No explicit limit |
| Verifier timeout | `600` seconds |

`--smoke` uses low effort, a short submission prompt and a 900-second limit. Claude Code and Gemini CLI also have a 12-turn limit. Codex has no native turn-limit flag; its smoke run uses the prompt and time limit, and explicitly rejects `--max-turns`. Smoke runs are installation tests, not configurations for reporting benchmark scores. Ordinary runs use the task instructions and the standard single-session prompt. Each run records its framework, CLI version, protocol, model, endpoint, task list, repeats, limits, and executed command in `run-settings.json`.

## Outputs

```text
runs/<name>/
  run-settings.json       # Run configuration, excluding credentials
  summary.json           # Per-trial scores, costs and token usage
  summary.csv            # Same per-trial results in table form
  aggregate.json         # Generated by tools/summarize_runs.py
  <trial>/
    result.json          # Harbor trial result
    agent/               # Agent logs and trajectory
    verifier/            # Held-out metrics and rewards
    artifacts/           # Submitted files and Active acquisition ledger
```

R² is read from the original verifier metrics. Active summaries also include `flops_spent`, `pool_flops_total`, `flops_fraction_of_pool`, and `observations_bought`. Aggregation checks that every configured task has the requested number of distinct completed trials. It does not update the published leaderboard automatically.

Raw trajectories and generated artifacts remain local under the ignored `runs/` directory. Review them before sharing. To rebuild task inputs after intentional source changes, move `benchmark/tasks/` aside, update the manifest, and prepare again.
