#!/usr/bin/env python3
"""Run this task's verifier with the hidden split already destroyed and the
candidate law held at arm's length.

``test_outputs.py`` is deliberately left byte-identical. The scoring, the metric
names and the reward convention are exactly what they were, and that is
checkable by diffing. Everything hardened here is arranged *around* it:

* The held-out answers are read into memory and the on-disk copy is replaced
  with an empty decoy of the same schema before any candidate code runs. The
  verifier still finds a file where it expects one -- ``pandas.read_parquet``
  is redirected to the in-memory copy for exactly those paths -- while a law
  that opens the file itself finds no rows.
* ``law.py`` is never imported into the grading process. The verifier's
  ``importlib`` calls are answered with a stand-in module whose ``law``
  forwards each call to a child process running as an unprivileged uid, in its
  own session so anything it forks is reaped afterwards. Agent code therefore
  cannot read the answers, patch numpy, rewrite the metrics, or plant a reward.

Neither measure would be worth much if the verifier shared the agent's
container: the agent has root there and could copy the split the moment Harbor
uploads it. That is why these tasks also set ``environment_mode = "separate"``.
"""

from __future__ import annotations

import contextlib
import importlib.machinery
import importlib.util
import json
import os
import runpy
import signal
import subprocess
import sys
import types
from pathlib import Path

import pandas as pd

TESTS_DIR = Path(__file__).resolve().parent
VERIFIER = TESTS_DIR / "test_outputs.py"
LAW_SERVER = TESTS_DIR / "law_server.py"
APP_DIR = Path("/app")
LAW_PATH = APP_DIR / "law.py"
# nobody. Dropping to a bare uid rather than a named account keeps this working
# on any base image, including ones without useradd.
LAW_UID = 65534
LAW_GID = 65534
RUN_TIMEOUT_SEC = 480

SPLIT_CANDIDATES = [
    TESTS_DIR / "data" / "test.parquet",
    Path("/tests/data/test.parquet"),
    Path("/app/data/test.parquet"),
    Path("tests/data/test.parquet"),
    Path("data/test.parquet"),
]

_shim_hits = 0


# --------------------------------------------------------------------------
# The hidden split
# --------------------------------------------------------------------------


def secure_split() -> tuple[pd.DataFrame, set[Path]]:
    """Load every on-disk copy of the answers, then leave empty decoys behind.

    Deleting outright would be simpler, but several verifiers resolve the split
    with an ``exists()`` probe and raise before they ever call ``read_parquet``.
    A zero-row file of the same schema keeps that probe honest and carries no
    information: the law is handed the feature rows anyway, so not even the row
    count is secret.
    """
    frame: pd.DataFrame | None = None
    found: list[Path] = []
    for candidate in SPLIT_CANDIDATES:
        try:
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in {p for p in found}:
            continue
        if frame is None:
            frame = pd.read_parquet(resolved)
        found.append(resolved)

    if frame is None:
        raise FileNotFoundError(f"no held-out split among {SPLIT_CANDIDATES}")

    decoy = frame.iloc[0:0]
    for path in found:
        decoy.to_parquet(path, index=False)
    for path in found:
        if len(pd.read_parquet(path)) != 0:
            raise RuntimeError(f"failed to neutralize the on-disk split at {path}")
    return frame, set(found)


def install_read_shim(frame: pd.DataFrame, paths: set[Path]) -> None:
    """Serve the in-memory answers to the verifier, and only to the verifier."""
    real_read_parquet = pd.read_parquet

    def read_parquet(path, *args, **kwargs):
        global _shim_hits
        try:
            resolved = Path(os.fspath(path)).resolve()
        except (OSError, TypeError, ValueError):
            resolved = None
        if resolved is not None and resolved in paths:
            _shim_hits += 1
            return frame.copy()
        return real_read_parquet(path, *args, **kwargs)

    pd.read_parquet = read_parquet


# --------------------------------------------------------------------------
# The candidate law, out of process
# --------------------------------------------------------------------------


def jsonable(value):
    if hasattr(value, "item") and getattr(value, "shape", None) == ():
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def drop_privileges() -> None:
    if os.geteuid() != 0:
        return
    with contextlib.suppress(OSError):
        os.setgroups([])
    os.setgid(LAW_GID)
    os.setuid(LAW_UID)


class LawProxy:
    """Stand-in for the agent's ``law``, backed by a child process."""

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None

    def _start(self) -> subprocess.Popen:
        # The law may need files it left in /app; make sure the unprivileged
        # child can read them whatever mode the agent created them with.
        subprocess.run(["chmod", "-R", "a+rX", str(APP_DIR)], check=False)
        environment = dict(os.environ, HOME="/tmp", PYTHONDONTWRITEBYTECODE="1")
        return subprocess.Popen(
            [sys.executable, str(LAW_SERVER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            cwd="/tmp",
            env=environment,
            start_new_session=True,
            preexec_fn=drop_privileges,  # noqa: PLW1509 - deliberate, see module docstring
        )

    def __call__(self, records, group):
        if self._process is None:
            self._process = self._start()
        request = json.dumps({"group": str(group), "rows": records}, default=jsonable)
        try:
            self._process.stdin.write(request + "\n")
            self._process.stdin.flush()
            line = self._process.stdout.readline()
        except (BrokenPipeError, ValueError) as exc:
            raise RuntimeError(f"law process is gone: {exc}") from exc
        if not line:
            raise RuntimeError("law process exited without answering")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise RuntimeError(str(reply.get("error", "law failed")))
        return reply["predictions"]

    def reap(self) -> None:
        if self._process is None:
            return
        with contextlib.suppress(Exception):
            self._process.stdin.close()
        with contextlib.suppress(Exception):
            os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
        with contextlib.suppress(Exception):
            self._process.wait(timeout=10)


class _NoopLoader(importlib.machinery.SourceFileLoader):
    """A loader whose ``exec_module`` runs nothing: the module is prebuilt."""

    def __init__(self) -> None:
        super().__init__("candidate_law", str(LAW_PATH))

    def exec_module(self, module):  # noqa: D102
        return None

    def create_module(self, spec):  # noqa: D102
        return None


def install_law_shim(proxy: LawProxy) -> None:
    """Answer the verifier's importlib calls with a module it cannot influence.

    Every verifier in this dataset loads the law the same way: build a spec from
    /app/law.py, make a module from it, exec it, then take ``.law``. Intercepting
    the first two steps yields a module whose ``law`` is the proxy, and whose
    exec step is a no-op, without touching the verifier's source.
    """
    marker_loader = _NoopLoader()
    marker = importlib.machinery.ModuleSpec("candidate_law", marker_loader, origin=str(LAW_PATH))
    real_spec_from_file_location = importlib.util.spec_from_file_location
    real_module_from_spec = importlib.util.module_from_spec

    def spec_from_file_location(name, location=None, *args, **kwargs):
        try:
            same = location is not None and Path(os.fspath(location)).resolve() == LAW_PATH.resolve()
        except (OSError, TypeError, ValueError):
            same = False
        if same:
            return marker
        return real_spec_from_file_location(name, location, *args, **kwargs)

    def module_from_spec(spec):
        if spec is marker:
            module = types.ModuleType("candidate_law")
            module.__file__ = str(LAW_PATH)
            module.__spec__ = marker
            module.__loader__ = marker_loader
            module.law = proxy
            return module
        return real_module_from_spec(spec)

    importlib.util.spec_from_file_location = spec_from_file_location
    importlib.util.module_from_spec = module_from_spec


# --------------------------------------------------------------------------


def fail(reason: str) -> None:
    print(f"guard: {reason}", file=sys.stderr)
    raise SystemExit(1)


def on_alarm(signum, frame):
    raise TimeoutError(f"verification exceeded {RUN_TIMEOUT_SEC}s")


def main() -> None:
    if not VERIFIER.is_file():
        fail(f"no verifier at {VERIFIER}")
    if not LAW_PATH.is_file():
        fail(f"no law at {LAW_PATH}")

    frame, paths = secure_split()
    install_read_shim(frame, paths)
    proxy = LawProxy()
    install_law_shim(proxy)

    with contextlib.suppress(ValueError, AttributeError):
        signal.signal(signal.SIGALRM, on_alarm)
        signal.alarm(RUN_TIMEOUT_SEC)
    try:
        runpy.run_path(str(VERIFIER), run_name="__main__")
    finally:
        with contextlib.suppress(ValueError, AttributeError):
            signal.alarm(0)
        proxy.reap()

    # Fail closed. If the verifier never went through the shim it read the
    # split some other way, which means this guard is not covering it.
    if _shim_hits == 0:
        fail("verifier did not read the split through the guard; hardening not applied")


if __name__ == "__main__":
    main()
