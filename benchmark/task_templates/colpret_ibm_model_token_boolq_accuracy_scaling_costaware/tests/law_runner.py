#!/usr/bin/env python3
"""Run a candidate `law` in a throwaway child process and print its predictions.

The grader never imports agent code. `law.py` is arbitrary code the agent wrote,
and in a shared interpreter it could rewrite the grader's own functions, its
metrics dict, or numpy itself. Here it runs:

* in a separate process, so it cannot reach the grader's state;
* as an unprivileged user, so it cannot write the reward files; and
* after the hidden split has already been read into the parent's memory and
  deleted from disk, so there is nothing left to copy the answers from.

stdin carries {"groups": [{"group": str, "rows": [{feature: value}, ...]}, ...]}
and stdout carries {"ok": bool, "predictions": [...] } -- flat, in the same row
order the groups were given in.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

LAW_PATH = Path("/app/law.py")
TARGET = 'boolq_accuracy'


def load_law():
    if not LAW_PATH.exists():
        raise FileNotFoundError(f"no law.py found at {LAW_PATH}")
    spec = importlib.util.spec_from_file_location("candidate_law", LAW_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {LAW_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "law"):
        raise AssertionError("law.py must define law(input_data, group)")
    return module.law


def main() -> None:
    payload = json.load(sys.stdin)
    predictions: list[float] = []
    try:
        law = load_law()
        for block in payload["groups"]:
            rows = block["rows"]
            output = law(rows, str(block["group"]))
            if not isinstance(output, list) or len(output) != len(rows):
                raise AssertionError("law must return one prediction dict per input row")
            for item in output:
                if not isinstance(item, dict) or TARGET not in item:
                    raise AssertionError(
                        f"each prediction must be a dict containing {TARGET!r}"
                    )
                value = float(item[TARGET])
                if not math.isfinite(value):
                    raise AssertionError("predictions must be finite")
                predictions.append(value)
    except Exception as exc:  # noqa: BLE001 - a bad law scores 0, it does not crash the trial
        json.dump({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, sys.stdout)
        return
    json.dump({"ok": True, "predictions": predictions}, sys.stdout)


if __name__ == "__main__":
    main()
