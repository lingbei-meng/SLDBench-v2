#!/usr/bin/env python3
"""Metered observation oracle. Runs as root; the agent may only reach it
through the single pinned sudoers rule.

Security model
--------------
The agent user can neither read the answer table nor write the ledger:

* ``/opt/oracle/private/`` is ``chmod 700 root:root`` -- targets, and the
  ledger signing key, are unreachable.
* ``/var/lib/oracle/`` is ``chmod 700 root:root`` -- the spend log cannot be
  deleted, truncated or forged. Only this script, running as root, appends.
* Every ledger line is HMAC-signed with a per-task key that exists only inside
  ``/opt/oracle/private`` and in the grader. The grader ignores unsigned lines,
  so a ledger substituted anywhere downstream cannot understate spend.
* The sudoers rule pins ``/opt/oracle/observe``, a root-owned wrapper that
  execs this script with ``python3 -I``. Isolated mode drops PYTHON* variables,
  the caller's ``~/.local`` site-packages and the script directory from
  ``sys.path``, so no agent-writable file can be imported into this process.
* ``--ids-file`` is read with the caller's privileges, not root's. Reading it
  as root would turn the oracle into an arbitrary-file-read gadget: unknown
  ids are echoed back in the error message, which leaked the first tokens of
  any root-only file before this was fixed.

Billing is idempotent per observation: re-buying an id costs nothing, so an
agent cannot inflate its own reported spend, and a retry after a crash is free.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import hmac
import json
import os
import sys
import time
from pathlib import Path

ANSWERS_PATH = Path("/opt/oracle/private/answers.parquet")
LEDGER_KEY_PATH = Path("/opt/oracle/private/ledger.key")
LEDGER_PATH = Path("/var/lib/oracle/ledger.jsonl")
MAX_IDS_PER_CALL = 100_000
MAX_IDS_FILE_BYTES = 16 * 1024 * 1024


def fail(message: str, code: int = 2) -> "NoReturn":  # type: ignore[name-defined]
    """Report a refusal as JSON on stdout so callers can always parse us."""
    print(json.dumps({"ok": False, "error": message}, indent=1))
    raise SystemExit(code)


# --------------------------------------------------------------------------
# Ledger signing
# --------------------------------------------------------------------------


def ledger_key() -> bytes:
    if not LEDGER_KEY_PATH.exists():
        fail(f"ledger signing key missing at {LEDGER_KEY_PATH}", code=3)
    return LEDGER_KEY_PATH.read_bytes()


def canonical(record: dict) -> bytes:
    """Byte string a signature covers. Must match the grader exactly."""
    body = {key: record[key] for key in ("obs_id", "cost_flops", "ts")}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign(record: dict, key: bytes) -> str:
    return hmac.new(key, canonical(record), hashlib.sha256).hexdigest()


def read_ledger(key: bytes) -> tuple[dict[str, float], int, int]:
    """Return {obs_id: price} already paid for, plus valid and rejected counts.

    Only correctly signed lines count. A corrupt or partially-written line is
    skipped rather than fatal: the ledger is append-only under a lock, so at
    worst we lose a torn final line, and refusing to serve would strand the
    agent.
    """
    paid: dict[str, float] = {}
    valid = 0
    rejected = 0
    if not LEDGER_PATH.exists():
        return paid, valid, rejected
    with LEDGER_PATH.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                signature = str(record["sig"])
                obs_id = str(record["obs_id"])
                price = float(record["cost_flops"])
            except (ValueError, KeyError, TypeError):
                rejected += 1
                continue
            if not hmac.compare_digest(signature, sign(record, key)):
                rejected += 1
                continue
            valid += 1
            paid.setdefault(obs_id, price)
    return paid, valid, rejected


def summarize(paid: dict[str, float], pool_total: float) -> dict:
    spent = float(sum(paid.values()))
    summary = {
        "observations_bought": len(paid),
        "flops_spent": spent,
        "pool_flops_total": pool_total,
    }
    if pool_total > 0:
        summary["flops_fraction_of_pool"] = spent / pool_total
    return summary


def load_tables():
    import pandas as pd

    if not ANSWERS_PATH.exists():
        fail(f"answer table missing at {ANSWERS_PATH}", code=3)
    answers = pd.read_parquet(ANSWERS_PATH)
    prices = {
        str(row.obs_id): float(row.cost_flops) for row in answers.itertuples(index=False)
    }
    targets = {
        str(row.obs_id): float(row.target) for row in answers.itertuples(index=False)
    }
    pool_total = float(sum(prices.values()))
    return prices, targets, pool_total


# --------------------------------------------------------------------------
# Argument handling
# --------------------------------------------------------------------------


def read_as_caller(path: Path) -> str:
    """Read *path* with the invoking user's privileges, never root's.

    Under sudo the real and effective uid are both 0, so an unguarded read here
    would hand the agent every root-only file on the box by way of the
    "unknown observation id" error message. Dropping the effective uid for the
    duration of the read makes the oracle no more powerful as a reader than the
    agent already is. The saved set-user-id stays 0, so root is restored
    afterwards to append the ledger.
    """
    caller_uid = os.environ.get("SUDO_UID")
    caller_gid = os.environ.get("SUDO_GID")
    dropped = False
    if os.geteuid() == 0 and caller_uid is not None and caller_uid != "0":
        try:
            if caller_gid is not None:
                os.setegid(int(caller_gid))
            os.seteuid(int(caller_uid))
            dropped = True
        except (ValueError, OSError):
            fail("cannot drop privileges to read --ids-file", code=4)
    try:
        if not path.is_file():
            fail(f"--ids-file not found or not a regular file: {path}")
        if path.stat().st_size > MAX_IDS_FILE_BYTES:
            fail(f"--ids-file is too large (> {MAX_IDS_FILE_BYTES} bytes)")
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            fail(f"--ids-file is not readable utf-8 text: {path}")
    finally:
        if dropped:
            os.seteuid(0)
            os.setegid(0)


def collect_requested_ids(args) -> list[str]:
    raw: list[str] = []
    if args.ids:
        raw.extend(part for chunk in args.ids for part in chunk.split(","))
    if args.ids_file:
        raw.extend(read_as_caller(Path(args.ids_file)).replace(",", "\n").split())
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            cleaned.append(item)
    if not cleaned:
        fail("no observation ids given; pass --ids or --ids-file")
    if len(cleaned) > MAX_IDS_PER_CALL:
        fail(f"too many ids in one call ({len(cleaned)} > {MAX_IDS_PER_CALL})")
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="observe",
        description="Buy target values for candidate experiments. Each distinct "
        "observation is charged its FLOPs price once.",
    )
    parser.add_argument(
        "--ids",
        action="append",
        metavar="ID[,ID...]",
        help="observation ids to buy; repeatable and comma-separated",
    )
    parser.add_argument("--ids-file", metavar="PATH", help="file of ids, one per line")
    parser.add_argument(
        "--status",
        action="store_true",
        help="report spend so far without buying anything (free)",
    )
    args = parser.parse_args()

    key = ledger_key()
    prices, targets, pool_total = load_tables()

    if args.status:
        paid, valid, rejected = read_ledger(key)
        print(
            json.dumps(
                {
                    "ok": True,
                    **summarize(paid, pool_total),
                    "ledger_entries": valid,
                    "ledger_entries_rejected": rejected,
                },
                indent=1,
            )
        )
        return

    requested = collect_requested_ids(args)
    unknown = [i for i in requested if i not in prices]
    if unknown:
        preview = ", ".join(unknown[:5]) + (" ..." if len(unknown) > 5 else "")
        fail(
            f"{len(unknown)} unknown observation id(s), nothing was charged: {preview}. "
            "Ids must come from the obs_id column of the catalogue."
        )

    try:
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        handle = LEDGER_PATH.open("a", encoding="utf-8")
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EPERM):
            fail("cannot write the spend ledger; the oracle must run via sudo", code=4)
        raise

    with handle:
        # Hold the lock across read-then-append so two concurrent buys cannot
        # both decide the same id is unpaid.
        fcntl.flock(handle, fcntl.LOCK_EX)
        paid, _, _ = read_ledger(key)
        observations = []
        charged_now = 0.0
        stamp = time.time()
        for obs_id in requested:
            price = prices[obs_id]
            first_time = obs_id not in paid
            if first_time:
                charged_now += price
                paid[obs_id] = price
                record = {"obs_id": obs_id, "cost_flops": price, "ts": stamp}
                record["sig"] = sign(record, key)
                handle.write(json.dumps(record) + "\n")
            observations.append(
                {
                    "obs_id": obs_id,
                    "target": targets[obs_id],
                    "cost_flops": price,
                    "newly_charged": first_time,
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle, fcntl.LOCK_UN)

    print(
        json.dumps(
            {
                "ok": True,
                "observations": observations,
                "flops_charged_now": charged_now,
                **summarize(paid, pool_total),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - never leak a traceback to the agent
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, indent=1))
        sys.exit(1)
