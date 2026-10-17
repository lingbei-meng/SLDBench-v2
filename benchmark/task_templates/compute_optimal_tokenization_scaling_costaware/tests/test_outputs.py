#!/usr/bin/env python3
"""Cost-aware verifier: grade the discovered law, then report what it cost.

Runs in a dedicated container (``environment_mode = "separate"``) built from
this directory, so the hidden split and the grader never exist inside the
environment the agent controls. The agent's ``/app`` and the spend ledger
arrive as collected artifacts; the ledger is written only by the root-owned
oracle.

Three things happen in a fixed order, and the order is the point:

1. The ledger is read before anything else, so nothing that runs later can
   change what the trial is recorded as having spent. Only ``/var/lib/oracle``
   is trusted, and only lines carrying a valid HMAC. Anything the agent can
   write -- ``/logs/artifacts`` and ``/app`` both come straight out of its
   container -- is deliberately not a fallback.
2. The hidden split is loaded into memory and then deleted from disk.
3. Only then is ``law.py`` executed, in a child process, as an unprivileged
   user. An agent that could read ``test.parquet`` at grading time would score
   a perfect R^2 having bought nothing, which is exactly the result this
   setting exists to make impossible.

``reward`` is pure extrapolation accuracy. Compute spend is *reported*, never
subtracted, because the population is compared as an accuracy/FLOPs Pareto
frontier -- folding cost into the scalar would destroy the axis we want to plot.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import math
import os
import pwd
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ['compute_flops', 'model_parameter_count', 'global_parameter_count', 'training_data_bytes', 'patch_size']
TARGET = 'c4_loss'
GROUPED = True
POOL_FLOPS_TOTAL = 4.9e+22
POOL_ROWS = 402
# Same key as /opt/oracle/private/ledger.key inside the environment image. It
# lives here rather than being read from that file because this container is
# built from tests/, not from environment/. It never enters the agent's
# container: tests/ is only ever the verifier image's build context.
LEDGER_KEY = bytes.fromhex('71d64f679622bb1a1055f0d8a4d42fdfd4aafbbbe162941fdce227906bb63ada')

TESTS_DIR = Path(__file__).resolve().parent
TEST_CANDIDATES = [
    TESTS_DIR / "data" / "test.parquet",
    Path("/tests/data/test.parquet"),
    Path("data/test.parquet"),
]
LAW_RUNNER = TESTS_DIR / "law_runner.py"
APP_DIR = Path("/app")
LEDGER_PATH = Path("/var/lib/oracle/ledger.jsonl")
REWARD_JSON = Path("/logs/verifier/reward.json")
REWARD_TEXT = Path("/logs/verifier/reward.txt")
METRICS_PATH = Path("/logs/verifier/metrics.json")
LAW_TIMEOUT_SEC = 420
LAW_USER = "grader"


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


# --------------------------------------------------------------------------
# 1. Spend, read before anything else runs
# --------------------------------------------------------------------------


def signature_of(record: dict) -> str:
    body = {key: record[key] for key in ("obs_id", "cost_flops", "ts")}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(LEDGER_KEY, canonical, hashlib.sha256).hexdigest()


def read_spend():
    """Summarize the oracle ledger, counting signed lines only.

    The environment image ships an empty ledger, so the file is always present
    and always collected. ``ledger_status`` distinguishes the cases the frontier
    analysis has to tell apart: a clean read, and a ledger that went missing or
    carries lines the oracle did not write. Spend is reported as ``None`` rather
    than 0.0 in that second case, so a collection failure can never be mistaken
    for a free-and-accurate result.
    """
    if not LEDGER_PATH.exists():
        return {
            "observations_bought": None,
            "flops_spent": None,
            "ledger_found": False,
            "ledger_status": "missing",
            "ledger_lines_rejected": 0,
        }
    paid: dict[str, float] = {}
    rejected = 0
    for line in LEDGER_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            obs_id = str(record["obs_id"])
            price = float(record["cost_flops"])
            given = str(record["sig"])
        except (ValueError, KeyError, TypeError):
            rejected += 1
            continue
        if not hmac.compare_digest(given, signature_of(record)):
            rejected += 1
            continue
        paid.setdefault(obs_id, price)
    return {
        "observations_bought": len(paid),
        "flops_spent": float(sum(paid.values())),
        "ledger_found": True,
        "ledger_status": "tampered" if rejected else "ok",
        "ledger_lines_rejected": rejected,
    }


# --------------------------------------------------------------------------
# 2. The hidden split: load, then destroy
# --------------------------------------------------------------------------


def purge(test_path: Path) -> bool:
    """Delete every on-disk copy of the answers this container holds."""
    for target in {test_path.parent, *(p.parent for p in TEST_CANDIDATES)}:
        shutil.rmtree(target, ignore_errors=True)
    return not any(path.exists() for path in TEST_CANDIDATES)


# --------------------------------------------------------------------------
# 3. The candidate law, at arm's length
# --------------------------------------------------------------------------


def jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def group_blocks(frame: pd.DataFrame):
    """Feature rows per group, plus the matching truth, in one row order."""
    blocks = []
    truth: list[float] = []
    if GROUPED:
        for label, group_frame in frame.groupby("group", sort=False):
            blocks.append(
                {
                    "group": str(label),
                    "rows": group_frame[FEATURE_COLUMNS].to_dict(orient="records"),
                }
            )
            truth.extend(group_frame[TARGET].astype(float).tolist())
    else:
        blocks.append(
            {
                "group": str(frame["group"].iloc[0]),
                "rows": frame[FEATURE_COLUMNS].to_dict(orient="records"),
            }
        )
        truth.extend(frame[TARGET].astype(float).tolist())
    return blocks, np.asarray(truth, dtype=float)


def drop_privileges():
    """Become ``grader`` in the child, so a law cannot write the reward files.

    /logs/verifier is a world-writable bind mount; a law running as root could
    otherwise plant a reward and then kill the grader, leaving the planted file
    as the trial's result. A no-op when the grader is already unprivileged --
    failing here would turn a hardening step into a broken verifier.
    """
    if os.geteuid() != 0:
        return
    entry = pwd.getpwnam(LAW_USER)
    os.setgroups([])
    os.setgid(entry.pw_gid)
    os.setuid(entry.pw_uid)


def run_law(blocks) -> tuple[list[float] | None, str | None]:
    if not LAW_RUNNER.exists():
        return None, f"law runner missing at {LAW_RUNNER}"
    if not (APP_DIR / "law.py").exists():
        return None, "no law.py found at /app/law.py"

    # The law may need files it left in /app; make sure the unprivileged child
    # can actually read them, whatever mode the agent created them with.
    subprocess.run(["chmod", "-R", "a+rX", str(APP_DIR)], check=False)

    payload = json.dumps({"groups": blocks}, default=jsonable)
    # start_new_session so the whole process group can be reaped afterwards: a
    # law that forks a background process must not outlive this call and race
    # the reward file.
    process = subprocess.Popen(
        [sys.executable, str(LAW_RUNNER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd="/tmp",
        start_new_session=True,
        preexec_fn=drop_privileges,  # noqa: PLW1509 - deliberate, see docstring
    )
    try:
        out, err = process.communicate(payload, timeout=LAW_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        out, err = "", f"law did not finish within {LAW_TIMEOUT_SEC}s"
    finally:
        with contextlib.suppress(Exception):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)

    if not out.strip():
        return None, f"law produced no output: {err.strip()[-400:]}"
    try:
        result = json.loads(out)
    except ValueError:
        return None, f"law output was not JSON: {out.strip()[-400:]}"
    if not result.get("ok"):
        return None, str(result.get("error", "law failed"))
    return [float(v) for v in result["predictions"]], None


def write(metrics: dict) -> None:
    try:
        REWARD_JSON.parent.mkdir(parents=True, exist_ok=True)
        REWARD_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        REWARD_TEXT.write_text(str(metrics["reward"]), encoding="utf-8")
        METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    except OSError:
        Path("metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    spend = read_spend()
    metrics = {
        "reward": 0.0,
        "r2": 0.0,
        "target_r2_clipped": -1.0,
        "nmse": float("inf"),
        "pool_flops_total": POOL_FLOPS_TOTAL,
        "pool_rows": POOL_ROWS,
        **spend,
    }
    spent = spend["flops_spent"]
    bought = spend["observations_bought"]
    metrics["flops_fraction_of_pool"] = (
        spent / POOL_FLOPS_TOTAL if spent is not None and POOL_FLOPS_TOTAL > 0 else None
    )
    metrics["observed_fraction_of_pool"] = (
        bought / POOL_ROWS if bought is not None and POOL_ROWS > 0 else None
    )

    test_path = first_existing(TEST_CANDIDATES)
    if test_path is None:
        metrics["error"] = "hidden test split not found"
        write(metrics)
        return

    blocks = None
    load_error = None
    try:
        test = pd.read_parquet(test_path)
        blocks, true = group_blocks(test)
        del test
    except Exception as exc:  # noqa: BLE001
        load_error = f"{type(exc).__name__}: {exc}"

    # Unconditional, and before any candidate code runs: nothing else in this
    # container should be able to reach a readable copy of the answers.
    metrics["hidden_split_purged"] = purge(test_path)

    if load_error is not None:
        metrics["error"] = load_error
        write(metrics)
        return
    if not metrics["hidden_split_purged"]:
        metrics["error"] = "refusing to run the candidate law: hidden split still on disk"
        write(metrics)
        return

    pred_list, error = run_law(blocks)
    if error is not None:
        metrics["error"] = error
        write(metrics)
        return

    pred = np.asarray(pred_list, dtype=float)
    if len(pred) != len(true):
        metrics["error"] = "prediction count does not match the held-out rows"
        write(metrics)
        return

    squared_error = float(np.mean((true - pred) ** 2))
    variance = float(np.mean((true - np.mean(true)) ** 2))
    nmse = squared_error / variance if variance > 1e-12 else float("inf")
    r2 = 1.0 - nmse
    metrics["nmse"] = nmse
    metrics["r2"] = r2
    # Reward mirrors the passive tasks: raw R^2, unclipped, 0.0 if degenerate.
    metrics["reward"] = r2 if math.isfinite(r2) else 0.0
    # Also report the clipped target-region R^2 used by the budget-aware
    # sequential-design literature, so results are comparable to published
    # numbers without changing this repo's unclipped reward convention.
    metrics["target_r2_clipped"] = max(-1.0, min(1.0, r2)) if math.isfinite(r2) else -1.0
    write(metrics)


if __name__ == "__main__":
    main()
