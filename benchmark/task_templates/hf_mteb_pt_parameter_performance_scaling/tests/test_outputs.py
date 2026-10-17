# -*- coding: utf-8 -*-
import importlib.util
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
LOG_DIR = Path(os.environ.get("LOG_DIR", "/logs/verifier"))
LAW_PATH = APP_DIR / "law.py"
LOCAL_LAW_PATH = LAW_PATH
TEST_PATHS = [
    TESTS_DIR / "data" / "test.parquet",
    APP_DIR / "data" / "test.parquet",
    Path("tests/data/test.parquet"),
    Path("data/test.parquet"),
]
REWARD_OUTPUT_FILE = LOG_DIR / "reward.txt"
FEATURE_COLS = ['parameter_count']
TARGET_COLS = ['benchmark_score']


def get_failure_result(error_msg: str = "Evaluation failed.") -> Dict[str, Any]:
    return {"nmse": 100000.0, "nmae": 100000.0, "r2": -1.0, "error": error_msg}


def calculate_final_metrics(predictions: np.ndarray, true_values: np.ndarray) -> Dict[str, Any]:
    try:
        pred = np.asarray(predictions, dtype=float)
        true = np.asarray(true_values, dtype=float)
    except (ValueError, TypeError):
        return get_failure_result("Could not convert predictions or true values to float arrays.")
    if np.isnan(pred).any() or np.isinf(pred).any():
        return get_failure_result("Predictions contain NaN or Inf values.")
    if true.ndim == 1:
        true = true.reshape(-1, 1)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    if true.shape != pred.shape:
        return get_failure_result(f"Shape mismatch: true values {true.shape} vs. predictions {pred.shape}.")
    variance = np.var(true, axis=0)
    mad = np.mean(np.abs(true - np.mean(true, axis=0)), axis=0)
    mse = np.mean((true - pred) ** 2, axis=0)
    mae = np.mean(np.abs(true - pred), axis=0)
    nmse = np.divide(mse, variance, out=np.full_like(mse, 100000.0), where=variance > 1e-9)
    nmae = np.divide(mae, mad, out=np.full_like(mae, 100000.0), where=mad > 1e-9)
    return {"nmse": float(np.mean(nmse)), "nmae": float(np.mean(nmae)), "r2": float(1.0 - np.mean(nmse))}


def load_law_function():
    path = LAW_PATH if LAW_PATH.exists() else LOCAL_LAW_PATH
    if not path.exists():
        raise FileNotFoundError(f"No law.py found at {LAW_PATH} or {LOCAL_LAW_PATH}")
    spec = importlib.util.spec_from_file_location("law_module", path)
    law_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(law_module)
    if not hasattr(law_module, "law"):
        raise ImportError("law.py exists but does not contain a required law function.")
    return law_module.law


def write_reward(score: float):
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        REWARD_OUTPUT_FILE.write_text(str(score), encoding="utf-8")
    except Exception:
        pass


def load_test_df():
    for path in TEST_PATHS:
        if path.exists():
            return pd.read_parquet(path)
    raise FileNotFoundError("No test.parquet found.")


if __name__ == "__main__":
    final_reward = 0.0
    try:
        law_func = load_law_function()
        df_test = load_test_df()
        all_y_true = []
        all_y_pred = []
        for group_name, group_df in df_test.groupby("group"):
            input_data = group_df[FEATURE_COLS].to_dict(orient="records")
            predictions_list = law_func(input_data, group_name)
            if not isinstance(predictions_list, list):
                raise TypeError(f"Expected list, got {type(predictions_list)}")
            if len(predictions_list) != len(input_data):
                raise ValueError("Prediction length mismatch")
            pred_df = pd.DataFrame(predictions_list)
            if not all(col in pred_df.columns for col in TARGET_COLS):
                raise ValueError(f"Missing prediction columns. Expected {TARGET_COLS}")
            all_y_true.append(group_df[TARGET_COLS].values)
            all_y_pred.append(pred_df[TARGET_COLS].values)
        metrics = calculate_final_metrics(np.vstack(all_y_pred), np.vstack(all_y_true))
        if "error" in metrics:
            print(f"Metric Error: {metrics['error']}")
        else:
            final_reward = metrics["r2"]
            print(metrics)
    except Exception as exc:
        print(f"Evaluation crashed: {exc}")
        final_reward = 0.0
    finally:
        write_reward(final_reward)
        print(f"reward={final_reward}")
