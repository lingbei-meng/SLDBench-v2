#!/usr/bin/env python3
"""Import the candidate law and answer one JSON request per line.

Runs as an unprivileged uid in its own session, started by ``guard.py``. The
answers are already gone from disk by the time this process exists, and nothing
it does can reach the grader: it has a pipe and nothing else.

Protocol, one JSON object per line each way::

    -> {"group": "...", "rows": [{feature: value}, ...]}
    <- {"ok": true, "predictions": [{target: value}, ...]}
    <- {"ok": false, "error": "..."}

The reply pipe is a dup of the original stdout, and fd 1 is then pointed at
stderr, so anything the law prints becomes console noise instead of corrupting
the protocol.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

LAW_PATH = Path("/app/law.py")


def jsonable(value):
    if hasattr(value, "item") and getattr(value, "shape", None) == ():
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def load_law():
    if not LAW_PATH.is_file():
        raise FileNotFoundError(f"no law.py at {LAW_PATH}")
    spec = importlib.util.spec_from_file_location("candidate_law", LAW_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {LAW_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "law"):
        raise AssertionError("law.py must define law(input_data, group)")
    return module.law


def main() -> None:
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    law = None
    load_error = None
    try:
        law = load_law()
    except BaseException as exc:  # noqa: BLE001 - reported per request, never fatal here
        load_error = f"{type(exc).__name__}: {exc}"

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            if load_error is not None:
                raise RuntimeError(load_error)
            request = json.loads(line)
            predictions = law(request["rows"], request["group"])
            reply = json.dumps({"ok": True, "predictions": predictions}, default=jsonable)
        except BaseException as exc:  # noqa: BLE001 - a bad law scores 0, it does not crash the trial
            reply = json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        protocol.write(reply + "\n")
        protocol.flush()


if __name__ == "__main__":
    main()
