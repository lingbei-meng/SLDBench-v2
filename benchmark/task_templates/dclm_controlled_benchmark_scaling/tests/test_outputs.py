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
FEATURE_COLUMNS = ['parameter_count', 'training_tokens']


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"None of these paths exists: {paths}")


def load_law():
    path = first_existing(LAW_CANDIDATES)
    spec = importlib.util.spec_from_file_location("candidate_law", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "law"):
        raise AssertionError("law.py must define law(input_data, group)")
    return module.law


def main():
    law = load_law()
    test = pd.read_parquet(first_existing(TEST_CANDIDATES))
    y_true = []
    y_pred = []
    for group, group_df in test.groupby("group", sort=False):
        records = group_df[FEATURE_COLUMNS].to_dict(orient="records")
        predictions = law(records, str(group))
        if not isinstance(predictions, list) or len(predictions) != len(records):
            raise AssertionError("law must return one prediction dictionary per input row")
        for prediction in predictions:
            if not isinstance(prediction, dict) or "target" not in prediction:
                raise AssertionError("Each prediction must be a dict containing 'target'")
            value = float(prediction["target"])
            if not math.isfinite(value):
                raise AssertionError("Predictions must be finite")
            y_pred.append(value)
        y_true.extend(group_df["target"].astype(float).tolist())

    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    squared_error = float(np.mean((true - pred) ** 2))
    variance = float(np.mean((true - np.mean(true)) ** 2))
    normalized_error = (
        squared_error / variance if variance > 1e-12 else float("inf")
    )
    score = 1.0 - normalized_error
    oracle_reward = (
        score if math.isfinite(score) else 0.0
    )
    metrics = {"oracle_reward": oracle_reward}
    try:
        REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
        REWARD_PATH.write_text(str(oracle_reward), encoding="utf-8")
        METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    except OSError:
        Path("metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
