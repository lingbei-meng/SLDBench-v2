"""Evaluate the reference law under four acquisition policies.

Run: python -m baselines.acquisition --workers 8
No API key or model calls are needed. See docs/BASELINES.md.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures as cf
import importlib.util
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# Single-threaded numerics in this process and in every worker.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASELINE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = BASELINE_ROOT.parent
DEFAULT_OUT = DEFAULT_DATASET_ROOT / "runs" / "reference_baselines"

RHOS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
RULES = ["cheapest", "random", "group_cheapest", "price_spread"]
RANDOM_SEEDS = [0, 1, 2]
FULL_POOL_RULE = "full_pool"
BUDGET_REL_TOL = 1e-12       # FP slack when testing "price fits in remaining budget"
ERROR_FLOOR = 1e-12          # floor on 1 - R^2 for log error ratios
RESULT_MARKER = "@@ACQ_RESULT@@"

FIT_COLUMNS = [
    "task", "task_id", "source", "rule", "seed", "rho",
    "budget_fraction", "realised_cost_fraction", "n_points", "pool_rows",
    "points_fraction", "groups_covered", "groups_total", "test_rows",
    "test_frac_group_unbought", "r2", "r2_clipped", "nmse", "status", "error",
    "fit_seconds", "predict_seconds", "worker_seconds", "hit_fit_deadline",
    "fit_time_budget_seconds", "n_groups_power", "n_groups_linear",
    "default_kind",
]


# ---------------------------------------------------------------------------
# Acquisition rules (pure functions of the catalogue: obs_id, cost_flops, group)
# ---------------------------------------------------------------------------


def coarse_to_fine_ranks(n: int) -> list[int]:
    """Bit-reversal visiting order of ranks 0..n-1: 0, n-1, mid, 1/4, 3/4, 1/8, ..."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    seen: set[int] = set()
    order: list[int] = []

    def add(rank: int) -> None:
        if rank not in seen:
            seen.add(rank)
            order.append(rank)

    add(0)
    add(n - 1)
    denom = 2
    # Once the dyadic spacing (n-1)/denom drops below 1/2 every rank has been hit.
    while len(order) < n and denom <= 4 * (n - 1):
        for k in range(1, denom, 2):
            # floor(k*(n-1)/denom + 1/2) in exact integer arithmetic
            add((2 * k * (n - 1) + denom) // (2 * denom))
        denom *= 2
    for rank in range(n):
        add(rank)
    return order


def _groups_by_cheapest(cat: pd.DataFrame) -> list[tuple[str, list[int]]]:
    """Groups ordered by (cheapest price, obs_id); each with its rows sorted by (price, obs_id)."""
    groups = []
    for label, frame in cat.groupby("_group_key", sort=False):
        ordered = frame.sort_values(["cost_flops", "obs_id"], kind="mergesort")
        first = ordered.iloc[0]
        groups.append(((float(first["cost_flops"]), str(first["obs_id"])), str(label),
                       ordered.index.tolist()))
    groups.sort(key=lambda item: item[0])
    return [(label, idx) for _, label, idx in groups]


def candidate_order(cat: pd.DataFrame, rule: str, seed: int | None) -> list[int]:
    """Full scan order (catalogue index labels) for a rule."""
    if rule == "cheapest":
        return cat.sort_values(["cost_flops", "obs_id"], kind="mergesort").index.tolist()
    if rule == "random":
        base = cat.sort_values("obs_id", kind="mergesort").index.to_numpy()
        perm = np.random.default_rng(int(seed)).permutation(len(base))
        return base[perm].tolist()
    if rule in ("group_cheapest", "price_spread"):
        groups = _groups_by_cheapest(cat)
        sequences = []
        for _, idx in groups:
            if rule == "group_cheapest":
                sequences.append(idx)
            else:
                sequences.append([idx[r] for r in coarse_to_fine_ranks(len(idx))])
        order: list[int] = []
        depth = max(len(s) for s in sequences)
        for r in range(depth):
            for seq in sequences:
                if r < len(seq):
                    order.append(seq[r])
        return order
    raise ValueError(f"unknown rule {rule!r}")


def greedy_fill(cat: pd.DataFrame, order: list[int], budget: float) -> list[int]:
    """Buy each candidate in ``order`` iff its price fits in the remaining budget."""
    prices = cat["cost_flops"].to_dict()
    spent = 0.0
    slack = BUDGET_REL_TOL * budget
    bought = []
    for index in order:
        price = float(prices[index])
        if spent + price <= budget + slack:
            bought.append(index)
            spent += price
    return bought


def purchase(cat: pd.DataFrame, rule: str, seed: int | None, rho: float) -> list[int]:
    if rule == FULL_POOL_RULE:
        return cat.index.tolist()
    total = math.fsum(cat["cost_flops"].astype(float).tolist())
    return greedy_fill(cat, candidate_order(cat, rule, seed), rho * total)


# ---------------------------------------------------------------------------
# Task metadata helpers
# ---------------------------------------------------------------------------


def _literal_assignments(path: Path, names: set[str]) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                try:
                    out[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    return out


def task_contract(task_dir: Path) -> dict:
    metadata = json.loads((task_dir / "metadata.json").read_text(encoding="utf-8"))["contract"]
    law_path = BASELINE_ROOT / "reference_laws" / (task_dir.name + ".py")
    law = _literal_assignments(law_path, {"TARGET", "FEATURE_COLUMNS"})
    mapping = json.loads((BASELINE_ROOT / "tasks.json").read_text(encoding="utf-8"))
    contract = mapping[task_dir.name + "_costaware"]["contract"]
    if not (contract["target"] == metadata["target"] == law["TARGET"]
            and contract["features"] == metadata["features"] == law["FEATURE_COLUMNS"]):
        raise AssertionError(f"{task_dir.name}: reference and released data contracts differ")
    return contract



def selected_tasks(dataset_root: Path) -> list[str]:
    selection = json.loads((dataset_root / "metadata/evaluation_tasks.json").read_text(encoding="utf-8"))
    return sorted(t + "_costaware" for t in selection["tasks"])



def source_mapping(dataset_root: Path, tasks: list[str]) -> dict[str, dict]:
    mapping = json.loads((BASELINE_ROOT / "tasks.json").read_text(encoding="utf-8"))
    if set(tasks) != set(mapping):
        raise AssertionError("Task list differs from the paper's 30 tasks")
    for task in tasks:
        task_contract(dataset_root / "active" / task.removesuffix("_costaware"))
    return mapping



# ---------------------------------------------------------------------------
# Worker: one fit, in its own process
# ---------------------------------------------------------------------------


def _jsonable(value):
    """Same conversion as tests/test_outputs.py:jsonable."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def run_one(spec: dict) -> dict:
    """Purchase, materialise, fit with the task's law.py, score. Returns bookkeeping only."""
    t_start = time.monotonic()
    task_dir = Path(spec["task_dir"])
    rule, seed, rho = spec["rule"], spec["seed"], float(spec["rho"])
    contract = task_contract(task_dir)
    target, features = contract["target"], contract["features"]

    cat = pd.read_parquet(task_dir / "catalogue.parquet")
    cat["_group_key"] = cat["group"].astype(str)
    total_cost = math.fsum(cat["cost_flops"].astype(float).tolist())
    bought_idx = purchase(cat, rule, seed, rho)
    bought_mask = cat.index.isin(bought_idx)          # keeps catalogue row order
    bought = cat.loc[bought_mask]
    spent = math.fsum(bought["cost_flops"].astype(float).tolist())
    groups_bought = set(bought["_group_key"])

    test = pd.read_parquet(task_dir.parent.parent / "datasets" / task_dir.name / "test.parquet")
    test_groups = test["group"].astype(str)

    record = {
        "rule": rule, "seed": seed, "rho": rho, "budget_fraction": rho,
        "realised_cost_fraction": spent / total_cost if total_cost > 0 else float("nan"),
        "n_points": int(len(bought)), "pool_rows": int(len(cat)),
        "points_fraction": len(bought) / len(cat),
        "groups_covered": int(len(groups_bought)),
        "groups_total": int(cat["_group_key"].nunique()),
        "test_rows": int(len(test)),
        "test_frac_group_unbought": float((~test_groups.isin(groups_bought)).mean()),
        "r2": float("nan"), "r2_clipped": float("nan"), "nmse": float("nan"),
        "status": "", "error": "", "fit_seconds": float("nan"),
        "predict_seconds": float("nan"), "hit_fit_deadline": False,
        "fit_time_budget_seconds": float("nan"), "n_groups_power": -1,
        "n_groups_linear": -1, "default_kind": "",
    }
    if len(bought) == 0:
        record["status"] = "empty_subset"
        record["worker_seconds"] = time.monotonic() - t_start
        return record

    # --- materialise the purchased subset exactly as solution/solve.py does ---
    answers = pd.read_parquet(task_dir / "measurements.parquet",
                              columns=["obs_id", "target"])
    answers = answers[answers["obs_id"].astype(str).isin(set(bought["obs_id"].astype(str)))]
    observed = pd.DataFrame({"obs_id": answers["obs_id"].astype(str).tolist(),
                             "target": [float(v) for v in answers["target"].tolist()]})
    if target != "target":
        observed = observed.rename(columns={"target": target})
    drop_columns = ["obs_id"] + ([] if "cost_flops" in features else ["cost_flops"])
    train = bought.drop(columns=["_group_key"]).merge(observed, on="obs_id", how="left")
    train = train.drop(columns=[c for c in drop_columns if c in train.columns])
    if train[target].isna().any() or len(train) != len(bought):
        raise AssertionError("purchased targets did not merge back onto the catalogue")

    scratch = Path(tempfile.mkdtemp(prefix="acq_fit_", dir=spec["scratch_root"]))
    try:
        (scratch / "data").mkdir()
        train_path = scratch / "data" / "train.parquet"
        train.to_parquet(train_path, index=False)
        del train, answers, observed

        # --- the task's own law.py, unmodified except for where it looks for data ---
        law_copy = scratch / "law.py"
        shutil.copy(BASELINE_ROOT / "reference_laws" / (task_dir.name + ".py"), law_copy)
        mod_spec = importlib.util.spec_from_file_location(
            f"candidate_law_{os.getpid()}", law_copy)
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
        module.TRAIN_CANDIDATES = [train_path]
        budget_s = float(getattr(module, "FIT_TIME_BUDGET_SECONDS", float("nan")))
        record["fit_time_budget_seconds"] = budget_s

        # --- verifier-shaped test blocks (tests/test_outputs.py:group_blocks) ---
        blocks, truth = [], []
        if contract["grouped"]:
            for label, gf in test.groupby("group", sort=False):
                blocks.append({"group": str(label),
                               "rows": gf[features].to_dict(orient="records")})
                truth.extend(gf[target].astype(float).tolist())
        else:
            blocks.append({"group": str(test["group"].iloc[0]),
                           "rows": test[features].to_dict(orient="records")})
            truth.extend(test[target].astype(float).tolist())
        true = np.asarray(truth, dtype=float)
        del test, truth
        payload = json.loads(json.dumps({"groups": blocks}, default=_jsonable))

        t_fit = time.monotonic()
        try:
            fitted = module.fit()
        except Exception as exc:  # noqa: BLE001
            record["fit_seconds"] = time.monotonic() - t_fit
            record["status"] = "fit_error"
            record["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            record["worker_seconds"] = time.monotonic() - t_start
            return record
        record["fit_seconds"] = time.monotonic() - t_fit
        record["hit_fit_deadline"] = bool(record["fit_seconds"] >= budget_s)
        default, fits = fitted[0], fitted[1]
        record["n_groups_power"] = sum(1 for kind, _ in fits.values() if kind == "power")
        record["n_groups_linear"] = sum(1 for kind, _ in fits.values() if kind == "linear")
        record["default_kind"] = default[0] if default is not None else "none"

        # --- tests/law_runner.py:main ---
        t_pred = time.monotonic()
        predictions: list[float] = []
        try:
            for block in payload["groups"]:
                rows = block["rows"]
                output = module.law(rows, str(block["group"]))
                if not isinstance(output, list) or len(output) != len(rows):
                    raise AssertionError("law must return one prediction dict per input row")
                for item in output:
                    if not isinstance(item, dict) or target not in item:
                        raise AssertionError(f"each prediction must be a dict containing {target!r}")
                    value = float(item[target])
                    if not math.isfinite(value):
                        raise AssertionError("predictions must be finite")
                    predictions.append(value)
        except Exception as exc:  # noqa: BLE001
            record["predict_seconds"] = time.monotonic() - t_pred
            record["status"] = "law_error"
            record["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            record["worker_seconds"] = time.monotonic() - t_start
            return record
        record["predict_seconds"] = time.monotonic() - t_pred

        # --- tests/test_outputs.py:main scoring ---
        pred = np.asarray(predictions, dtype=float)
        if len(pred) != len(true):
            record["status"] = "count_mismatch"
        else:
            squared_error = float(np.mean((true - pred) ** 2))
            variance = float(np.mean((true - np.mean(true)) ** 2))
            nmse = squared_error / variance if variance > 1e-12 else float("inf")
            r2 = 1.0 - nmse
            record["nmse"] = nmse
            record["r2"] = r2 if math.isfinite(r2) else float("nan")
            record["r2_clipped"] = max(-1.0, min(1.0, r2)) if math.isfinite(r2) else -1.0
            record["status"] = "ok" if math.isfinite(r2) else "degenerate_variance"
        del pred, true, predictions
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    record["worker_seconds"] = time.monotonic() - t_start
    return record


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_jobs(dataset_root: Path, tasks: list[str], scratch_root: Path) -> list[dict]:
    jobs = []
    for task in tasks:
        task_dir = dataset_root / "active" / task.removesuffix("_costaware")
        for rho in RHOS:
            if rho >= 1.0:
                combos = [(FULL_POOL_RULE, None)]
            else:
                combos = []
                for rule in RULES:
                    seeds = RANDOM_SEEDS if rule == "random" else [None]
                    combos.extend((rule, s) for s in seeds)
            for rule, seed in combos:
                jobs.append({"task": task, "task_dir": str(task_dir), "rule": rule,
                             "seed": seed, "rho": rho, "scratch_root": str(scratch_root)})
    return jobs


def job_key(job: dict) -> tuple:
    seed = -1 if job["seed"] is None or (isinstance(job["seed"], float) and math.isnan(job["seed"])) else int(job["seed"])
    return (job["task"], job["rule"], seed, round(float(job["rho"]), 6))


def launch(job: dict, timeout: float, log_dir: Path) -> dict:
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[var] = "1"
    env["PYTHONHASHSEED"] = "0"
    started = time.monotonic()
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--run-one", json.dumps(job)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        cwd=job["scratch_root"], start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            proc.kill() if os.name == "nt" else os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            pass
        out, err = proc.communicate()
    elapsed = time.monotonic() - started
    base = {"task": job["task"], "rule": job["rule"], "seed": job["seed"], "rho": job["rho"],
            "budget_fraction": job["rho"]}
    if timed_out:
        return {**base, "status": "timeout", "error": f"exceeded {timeout:.0f}s",
                "worker_seconds": elapsed}
    for line in out.splitlines()[::-1]:
        if line.startswith(RESULT_MARKER):
            return {**base, **json.loads(line[len(RESULT_MARKER):])}
    # Keep only the exception tail (no data is printed by the worker itself).
    tail = err.strip().splitlines()[-1][:200] if err.strip() else f"exit code {proc.returncode}"
    (log_dir / "worker_errors.log").open("a").write(
        f"{job_key(job)} rc={proc.returncode}: {tail}\n")
    return {**base, "status": "worker_crash", "error": tail, "worker_seconds": elapsed}


def estimated_weight(job: dict) -> float:
    """Rough scheduling weight: bigger pools and budgets first."""
    return float(job.get("_pool_rows", 1)) * max(job["rho"], 0.005) ** 0.5


def run_all(args) -> None:
    dataset_root = Path(args.dataset_root).resolve()
    out_dir = Path(args.out).resolve()
    if any(out_dir == dataset_root / name or out_dir.is_relative_to(dataset_root / name)
           for name in ("datasets", "active", "metadata", "baselines")):
        raise SystemExit("--out must not overwrite source data or code")
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_root = Path(args.scratch).resolve() if args.scratch else out_dir / "scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    checkpoint = scratch_root / "fits_checkpoint.jsonl"

    tasks = selected_tasks(dataset_root)
    if args.tasks:
        wanted = [t.strip() for t in args.tasks.split(",") if t.strip()]
        unknown = sorted(set(wanted) - set(tasks))
        if unknown:
            raise SystemExit(f"not in the selected set: {unknown}")
        tasks = [t for t in tasks if t in wanted]
    all_tasks = selected_tasks(dataset_root)
    mapping = source_mapping(dataset_root, all_tasks)

    jobs = build_jobs(dataset_root, tasks, scratch_root)
    if args.full_pool_only:
        jobs = [j for j in jobs if j["rule"] == FULL_POOL_RULE]
    input_files = [Path(__file__).resolve(), BASELINE_ROOT / "tasks.json"]
    for task in tasks:
        base = task.removesuffix("_costaware")
        input_files.extend([dataset_root / "active" / base / name
                            for name in ("catalogue.parquet", "measurements.parquet", "metadata.json")])
        input_files.extend([dataset_root / "datasets" / base / "test.parquet",
                            BASELINE_ROOT / "reference_laws" / (base + ".py")])
    fingerprints = {p.relative_to(dataset_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in input_files}
    manifest = {"inputs_sha256": fingerprints, "tasks": tasks, "full_pool_only": args.full_pool_only,
                "rhos": RHOS, "rules": RULES, "random_seeds": RANDOM_SEEDS,
                "expected_fits": len(jobs), "timeout_seconds": args.timeout,
                "python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "scipy": __import__("scipy").__version__}
    manifest_path = out_dir / "run_manifest.json"
    if checkpoint.exists():
        if not args.resume:
            raise SystemExit("A checkpoint exists. Use a new --out directory, or --resume.")
        if not manifest_path.exists() or json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise SystemExit("Checkpoint inputs, implementation, environment or experiment grid changed; use a new --out.")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    pool_rows = {t: len(pd.read_parquet(dataset_root / "active" / t.removesuffix("_costaware") / "catalogue.parquet",
                                        columns=["obs_id"])) for t in tasks}
    for job in jobs:
        job["_pool_rows"] = pool_rows[job["task"]]

    done: dict[tuple, dict] = {}
    if args.resume and checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[job_key(rec)] = rec
    todo = [j for j in jobs if job_key(j) not in done]
    todo.sort(key=lambda j: -estimated_weight(j))
    print(f"[acq] {len(jobs)} fits over {len(tasks)} tasks; {len(done)} already done; "
          f"running {len(todo)} with {args.workers} workers, timeout {args.timeout}s", flush=True)

    lock = threading.Lock()
    t0 = time.monotonic()
    completed = 0
    with checkpoint.open("a") as ckpt, cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(launch, {k: v for k, v in j.items() if not k.startswith("_")},
                               args.timeout, scratch_root): j for j in todo}
        for fut in cf.as_completed(futures):
            rec = fut.result()
            with lock:
                done[job_key(rec)] = rec
                ckpt.write(json.dumps(rec) + "\n")
                ckpt.flush()
                completed += 1
                if completed % 25 == 0 or completed == len(todo):
                    print(f"[acq] {completed}/{len(todo)} done, {time.monotonic() - t0:.0f}s elapsed, "
                          f"last={rec['task'][:40]} {rec['rule']} rho={rec['rho']} "
                          f"status={rec.get('status')}", flush=True)
    wall = time.monotonic() - t0

    records = [done[job_key(j)] for j in jobs]
    frame = pd.DataFrame(records)
    frame["task_id"] = frame["task"].map(lambda t: mapping[t]["task_id"])
    frame["source"] = frame["task"].map(lambda t: mapping[t]["source"])
    for col in FIT_COLUMNS:
        if col not in frame.columns:
            frame[col] = np.nan
    frame["seed"] = frame["seed"].astype("Int64")
    frame = frame[FIT_COLUMNS].sort_values(["task_id", "rho", "rule", "seed"], kind="mergesort")
    frame.to_csv(out_dir / "fits.csv", index=False, float_format="%.12g")
    run_info = {"wall_seconds_this_invocation": wall, "workers": args.workers,
                "timeout_seconds": args.timeout, "fits_run_this_invocation": len(todo)}
    summarize(dataset_root, out_dir, mapping, run_info, scratch_root)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _median_nan_worst(values) -> float:
    """Median over seeds with failures ranked worst (NaN -> -inf); NaN if the median fails."""
    vals = sorted((-math.inf if (v is None or not math.isfinite(v)) else float(v)) for v in values)
    if not vals:
        return float("nan")
    n = len(vals)
    mid = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    return mid if math.isfinite(mid) else float("nan")


def _finite_median(values) -> float | None:
    arr = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    return float(np.median(arr)) if len(arr) else None


def task_level_table(fits: pd.DataFrame) -> pd.DataFrame:
    """One row per (task, rule_label, rho): rules incl. per-seed random and random median."""
    rows = []
    full = fits[fits["rule"] == FULL_POOL_RULE].set_index("task")
    for task, tf in fits.groupby("task"):
        for rho in RHOS:
            if rho >= 1.0:
                if task not in full.index:
                    continue
                f = full.loc[task]
                for label in ["cheapest", "random", "random_seed0", "random_seed1",
                              "random_seed2", "group_cheapest", "price_spread"]:
                    rows.append({"task": task, "rule": label, "rho": rho, "r2": f["r2"],
                                 "realised_cost_fraction": f["realised_cost_fraction"],
                                 "n_points": f["n_points"], "status": f["status"]})
                continue
            sub = tf[np.isclose(tf["rho"], rho)]
            for rule in ["cheapest", "group_cheapest", "price_spread"]:
                r = sub[sub["rule"] == rule]
                if len(r):
                    r = r.iloc[0]
                    rows.append({"task": task, "rule": rule, "rho": rho, "r2": r["r2"],
                                 "realised_cost_fraction": r["realised_cost_fraction"],
                                 "n_points": r["n_points"], "status": r["status"]})
            rnd = sub[sub["rule"] == "random"]
            for _, r in rnd.iterrows():
                rows.append({"task": task, "rule": f"random_seed{int(r['seed'])}", "rho": rho,
                             "r2": r["r2"], "realised_cost_fraction": r["realised_cost_fraction"],
                             "n_points": r["n_points"], "status": r["status"]})
            if len(rnd):
                rows.append({"task": task, "rule": "random", "rho": rho,
                             "r2": _median_nan_worst(rnd["r2"].tolist()),
                             "realised_cost_fraction": float(np.median(rnd["realised_cost_fraction"])),
                             "n_points": float(np.median(rnd["n_points"])),
                             "status": ",".join(sorted(set(rnd["status"])))})
    return pd.DataFrame(rows)


def _clean_json(obj):
    """Replace non-finite floats (and numpy scalars) so summary.json is strict JSON."""
    if isinstance(obj, dict):
        return {str(k): _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    return obj


def summarize(dataset_root: Path, out_dir: Path, mapping: dict, run_info: dict | None,
              scratch_root: Path | None = None) -> None:
    fits = pd.read_csv(out_dir / "fits.csv")
    tl = task_level_table(fits)
    n_tasks_total = fits["task"].nunique()
    tl["source"] = tl["task"].map(lambda t: mapping[t]["source"])

    per_rule_rho = {}
    for rule in ["cheapest", "random", "random_seed0", "random_seed1", "random_seed2",
                 "group_cheapest", "price_spread"]:
        per_rule_rho[rule] = {}
        for rho in RHOS:
            s = tl[(tl["rule"] == rule) & np.isclose(tl["rho"], rho)]
            if not len(s):
                continue
            r2 = s["r2"].astype(float)
            finite = r2[np.isfinite(r2)]
            src_medians = [m for m in (_finite_median(g["r2"].astype(float).tolist())
                                       for _, g in s.groupby("source")) if m is not None]
            per_rule_rho[rule][f"{rho:g}"] = {
                "n_tasks": int(len(s)),
                "n_tasks_finite_r2": int(len(finite)),
                "median_r2": float(np.median(finite)) if len(finite) else None,
                "median_of_source_medians_r2": float(np.median(src_medians)) if src_medians else None,
                "n_sources": int(len(src_medians)),
                "share_r2_gt_0": float((r2 > 0).sum() / len(s)),
                "share_r2_gt_0.5": float((r2 > 0.5).sum() / len(s)),
                "median_realised_cost_fraction": float(np.median(s["realised_cost_fraction"])),
                "median_points_bought": float(np.median(s["n_points"])),
                "shared_full_pool_fit": bool(rho >= 1.0),
            }

    paired = {}
    for rho in RHOS:
        base = tl[(tl["rule"] == "cheapest") & np.isclose(tl["rho"], rho)].set_index("task")["r2"]
        paired[f"{rho:g}"] = {}
        for rule in ["group_cheapest", "price_spread", "random"]:
            other = tl[(tl["rule"] == rule) & np.isclose(tl["rho"], rho)].set_index("task")["r2"]
            common = base.index.intersection(other.index)
            b = base.loc[common].astype(float)
            o = other.loc[common].astype(float)
            b_rank = b.where(np.isfinite(b), -np.inf)
            o_rank = o.where(np.isfinite(o), -np.inf)
            beats = int((o_rank > b_rank).sum())
            ties = int((o_rank == b_rank).sum())
            both = np.isfinite(b) & np.isfinite(o)
            err_b = np.maximum(1.0 - b[both], ERROR_FLOOR)
            err_o = np.maximum(1.0 - o[both], ERROR_FLOOR)
            ratio = np.log10(err_b / err_o)
            paired[f"{rho:g}"][f"{rule}_vs_cheapest"] = {
                "n_tasks": int(len(common)),
                "share_rule_beats_cheapest": float(beats / len(common)) if len(common) else None,
                "n_rule_beats_cheapest": beats,
                "n_ties": ties,
                "n_both_finite": int(both.sum()),
                "median_log10_error_ratio": float(np.median(ratio)) if len(ratio) else None,
                "share_log10_error_ratio_gt_0": float((ratio > 0).mean()) if len(ratio) else None,
            }

    status_counts = fits["status"].value_counts().to_dict()
    summary = {
        "description": "Non-agent acquisition baselines: each task's archived reference law.py "
                       "fitted on greedily purchased pool subsets and scored with the Active "
                       "verifier metric (pooled R^2 = 1 - NMSE, unclipped).",
        "inputs": {
            "dataset_root": ".",
            "selection": "metadata/evaluation_tasks.json",
            "per_task_files": ["active/<task>/catalogue.parquet", "active/<task>/measurements.parquet",
                               "datasets/<task>/test.parquet", "baselines/reference_laws/<task>.py"],
            "task_to_source": "baselines/tasks.json",
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__,
                        "scipy": __import__("scipy").__version__, "threads_per_fit": 1},
        "design": {
            "rhos": RHOS, "rules": RULES, "random_seeds": RANDOM_SEEDS,
            "budget": "B = rho * sum(cost_flops) over the catalogue; greedy fill over the rule's order",
            "rho_1": "full pool, fitted once (rule 'full_pool'), reported under every rule",
            "random_aggregate": "per task, median over seeds 0,1,2 with failed fits ranked worst",
            "median_r2": "median over tasks with finite R^2",
            "shares": "share of all tasks (failed fits count as not exceeding the threshold)",
            "source_median": "median over sources of the per-source median task R^2 (finite only)",
            "paired_beats": "R2_rule > R2_cheapest, failed fits ranked worst, denominator = all tasks",
            "log10_error_ratio": "log10(max(1-R2_cheapest,1e-12)/max(1-R2_rule,1e-12)), both finite; "
                                 ">0 means the rule has smaller error",
        },
        "n_fits": int(len(fits)),
        "n_tasks": int(n_tasks_total),
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "n_fits_hit_fit_deadline": int(fits["hit_fit_deadline"].fillna(False).astype(bool).sum()),
        "fit_seconds": {"median": float(fits["fit_seconds"].median()),
                        "max": float(fits["fit_seconds"].max()),
                        "sum": float(fits["fit_seconds"].sum())},
        "worker_seconds_sum": float(fits["worker_seconds"].sum()),
        "per_rule_rho": per_rule_rho,
        "paired_vs_cheapest": paired,
        "task_mapping": mapping,
        "run": run_info or {},
    }
    (out_dir / "summary.json").write_text(json.dumps(_clean_json(summary), indent=2,
                                                     allow_nan=False) + "\n")
    print(f"[acq] wrote {out_dir / 'fits.csv'} and {out_dir / 'summary.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600.0, help="hard per-fit timeout (s)")
    parser.add_argument("--tasks", default="", help="comma-separated subset of task directories")
    parser.add_argument("--full-pool-only", action="store_true", help="Run only the 30 full-pool reference fits")
    parser.add_argument("--scratch", default=None,
                        help="scratch directory; defaults to <out>/scratch")
    parser.add_argument("--resume", action="store_true", help="reuse fits from the scratch checkpoint")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--run-one", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run_one is not None:
        record = run_one(json.loads(args.run_one))
        print(RESULT_MARKER + json.dumps(record), flush=True)
        return
    if not 1 <= args.workers <= 100:
        raise SystemExit("--workers must be between 1 and 100")
    if args.summarize_only:
        dataset_root = Path(args.dataset_root).resolve()
        mapping = source_mapping(dataset_root, selected_tasks(dataset_root))
        prev = {}
        summary_path = Path(args.out) / "summary.json"
        if summary_path.exists():
            prev = json.loads(summary_path.read_text(encoding="utf-8")).get("run", {})
        summarize(dataset_root, Path(args.out).resolve(), mapping, prev)
        return
    run_all(args)


if __name__ == "__main__":
    main()
