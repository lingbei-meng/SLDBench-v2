"""Check source imports, local documentation links, and accidental credentials."""
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".venv", "__pycache__", ".cache", "runs", ".pytest_cache"}


def release_files():
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if not path.is_file() or any(part in SKIP for part in rel.parts):
            continue
        if rel.parts[:2] == ("benchmark", "tasks"):
            continue
        yield path


def main():
    failures = []
    links = 0
    sources = 0
    for path in release_files():
        if path.suffix in (".parquet", ".png", ".pdf"):
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        if re.search(r"sk-[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----", text):
            failures.append(f"Credential-like content: {path.relative_to(ROOT)}")
        if path.suffix == ".py":
            try:
                compile(path.read_bytes(), str(path), "exec")
                sources += 1
            except SyntaxError as error:
                failures.append(str(error))
        if path.suffix != ".md":
            continue
        targets = re.findall(r"\[[^\]\n]*\]\(([^\s)]+)\)", text)
        targets += re.findall(r'(?:src|href)="([^\"]+)"', text)
        for target in targets:
            url = urlparse(target)
            if url.scheme or target.startswith(("#", "//")):
                continue
            resolved = (path.parent / unquote(url.path)).resolve()
            if not resolved.is_relative_to(ROOT) or not resolved.exists():
                failures.append(f"Broken local link in {path.relative_to(ROOT)}: {target}")
            links += 1
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Release checks passed: {sources} Python sources; {links} local links; no credential patterns.")


if __name__ == "__main__":
    main()
