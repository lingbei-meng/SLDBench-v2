"""Build task environments from the repository data and prepare the runtime."""
import argparse
import base64
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
TASKS = ROOT / "tasks"
RUNS = REPO / "runs"
IMAGE = "sldbench-runtime:20261016-three-harnesses"


def verify(path, spec):
    algorithm, expected = spec["integrity"].split("-", 1)
    actual = base64.b64encode(hashlib.new(algorithm, path.read_bytes()).digest()).decode()
    if actual != expected:
        raise RuntimeError(f"Integrity check failed: {path.name}")


def download_dependencies():
    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)
    lock = json.loads((ROOT / "dependencies.json").read_text())
    for name, spec in lock.items():
        dest = cache / name
        if not dest.exists():
            print(f"Downloading {name} ...", flush=True)
            with urllib.request.urlopen(spec["url"], timeout=120) as response:
                raw = response.read()
            temp = dest.with_suffix(".download")
            temp.write_bytes(raw)
            try:
                verify(temp, spec)
                temp.replace(dest)
            finally:
                temp.unlink(missing_ok=True)
        verify(dest, spec)


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f"Path outside expected directory: {relative}")
    return path


def materialize():
    """Copy verified canonical data and templates into isolated build contexts."""
    manifest = json.loads((ROOT / "tasks-manifest.json").read_text())
    selected = json.loads((REPO / "metadata/evaluation_tasks.json").read_text())["tasks"]
    expected_tasks = {t + suffix for t in selected for suffix in ("", "_costaware")}
    if set(manifest["tasks"]) != expected_tasks:
        raise RuntimeError("Runtime tasks do not match the evaluation subset")
    for rel, spec in manifest["files"].items():
        source = inside(REPO, spec["source"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != spec["sha256"]:
            raise RuntimeError(f"Source checksum mismatch: {spec['source']}")
        inside(TASKS, rel)
    if not TASKS.exists():
        # An interrupted copy never leaves an apparently complete task directory.
        with tempfile.TemporaryDirectory(prefix=".tasks-build-", dir=ROOT) as staging:
            target = Path(staging) / "tasks"
            target.mkdir()
            for rel, spec in manifest["files"].items():
                dest = inside(target, rel)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(inside(REPO, spec["source"]), dest)
            target.rename(TASKS)
    for rel, spec in manifest["files"].items():
        if hashlib.sha256(inside(TASKS, rel).read_bytes()).hexdigest() != spec["sha256"]:
            raise RuntimeError(f"Generated task file changed: {rel}. Move benchmark/tasks aside and prepare again.")
    print(f"Verified {len(manifest['tasks'])} tasks ({len(manifest['files'])} files).", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-only", action="store_true", help="Validate and assemble tasks without Docker or API access")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--no-proxy", action="store_true")
    args = parser.parse_args()
    if args.no_proxy:
        for name in list(os.environ):
            if name.lower().endswith("_proxy"):
                del os.environ[name]
    materialize()
    if args.data_only:
        return
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise SystemExit("Run the benchmark on Linux x86_64 (Windows: use WSL2).")
    subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["docker", "compose", "version"], check=True)
    exists = subprocess.run(["docker", "image", "inspect", IMAGE],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if args.rebuild or exists.returncode:
        download_dependencies()
        subprocess.run(["docker", "build", "--network", "host", "-t", IMAGE, str(ROOT)], check=True)
    for cli in ("claude", "codex", "gemini"):
        subprocess.run(["docker", "run", "--rm", "--network", "none", IMAGE, cli, "--version"], check=True)


if __name__ == "__main__":
    main()
