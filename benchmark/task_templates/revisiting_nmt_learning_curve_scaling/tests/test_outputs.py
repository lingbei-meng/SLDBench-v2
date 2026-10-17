from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


TEST_CANDIDATES = [Path("/tests/data/test.parquet"), Path("tests/data/test.parquet"), Path("data/test.parquet")]
LAW_CANDIDATES = [Path("/app/law.py")]
REWARD_PATH = Path("/logs/verifier/reward.txt")
METRICS_PATH = Path("/logs/verifier/metrics.json")
FEATURE_COLUMNS = ['seen_examples', 'cost_flops']
TARGET_COLUMNS = ['loss']
TARGET = TARGET_COLUMNS[0]


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"None of these paths exists: {paths}")


def main():
    law_path = first_existing(LAW_CANDIDATES)
    spec = importlib.util.spec_from_file_location("candidate_law", law_path)
    if spec is None or spec.loader is None:
        raise ImportError(law_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    test = pd.read_parquet(first_existing(TEST_CANDIDATES))
    predictions = []
    targets = []
    for group, frame in test.groupby("group", sort=True):
        output = module.law(frame[FEATURE_COLUMNS].to_dict(orient="records"), str(group))
        if not isinstance(output, list) or len(output) != len(frame):
            raise AssertionError("law must return one prediction per input")
        predictions.extend(float(row[TARGET]) for row in output)
        targets.extend(frame[TARGET].astype(float).tolist())
    pred = np.asarray(predictions)
    true = np.asarray(targets)
    if not np.isfinite(pred).all():
        raise AssertionError("predictions must be finite")
    mse = float(np.mean((true - pred) ** 2))
    variance = float(np.mean((true - np.mean(true)) ** 2))
    nmse = mse / variance if variance > 1e-12 else float("inf")
    r2 = 1.0 - nmse
    reward = r2 if math.isfinite(r2) else 0.0
    metrics = {"oracle_reward": reward, "r2": r2, "nmse": nmse}
    try:
        REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
        REWARD_PATH.write_text(str(reward), encoding="utf-8")
        METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    except OSError:
        Path("metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
