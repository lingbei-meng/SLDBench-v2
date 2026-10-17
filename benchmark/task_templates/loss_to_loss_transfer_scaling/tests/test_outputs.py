from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


TEST_PATH = Path("/tests/data/test.parquet")
LAW_PATH = Path("/app/law.py")
REWARD_PATH = Path("/logs/verifier/reward.txt")
METRICS_PATH = Path("/logs/verifier/metrics.json")
FEATURE_COLS = ['source_loss', 'model_size', 'tokens', 'flops_proxy']
TARGET_COLS = ["target_loss"]


def load_law():
    spec = importlib.util.spec_from_file_location("candidate_law", LAW_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    if not hasattr(module, "law"):
        raise AssertionError("law.py must define law(input_data, group)")
    return module.law


def main() -> None:
    law = load_law()
    test = pd.read_parquet(TEST_PATH)
    y_true = []
    y_pred = []
    for group, group_df in test.groupby("group"):
        records = group_df[FEATURE_COLS].to_dict(orient="records")
        preds = law(records, str(group))
        if len(preds) != len(records):
            raise AssertionError(f"law returned {len(preds)} predictions for {len(records)} rows in group {group}")
        for pred in preds:
            if not isinstance(pred, dict) or "target_loss" not in pred:
                raise AssertionError("Each prediction must be a dict with key 'target_loss'")
            value = float(pred["target_loss"])
            if not math.isfinite(value) or value <= 0:
                raise AssertionError("Predicted target_loss must be positive and finite")
            y_pred.append(value)
        y_true.extend(group_df["target_loss"].astype(float).tolist())

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mse = float(np.mean((y_true_arr - y_pred_arr) ** 2))
    baseline = float(np.mean((y_true_arr - np.mean(y_true_arr)) ** 2))
    nmse = mse / baseline if baseline > 0 else float("inf")
    mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
    nmae = mae / float(np.mean(np.abs(y_true_arr - np.mean(y_true_arr))))
    r2 = 1.0 - nmse
    reward = r2
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(str(reward), encoding="utf-8")
    METRICS_PATH.write_text(
        json.dumps(
            {
                "reward": reward,
                "r2": r2,
                "nmse": nmse,
                "nmae": nmae,
                "mse": mse,
                "mae": mae,
                "n_test": int(len(test)),
                "n_groups": int(test["group"].nunique()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(json.loads(METRICS_PATH.read_text()), indent=2))


if __name__ == "__main__":
    main()
