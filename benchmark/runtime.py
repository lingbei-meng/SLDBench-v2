from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from harbor.environments.docker.docker import DockerEnvironment


class DenseNetworkDockerEnvironment(DockerEnvironment):
    """Give every Harbor trial a collision-safe, isolated /29 network."""

    _BASE = int(ipaddress.IPv4Address("100.64.0.0"))
    _SLOTS = 1 << 19  # 100.64.0.0/10 split into /29 networks.
    _CLAIM_ROOT = Path("/tmp/harbor-dense-network-claims")

    def __init__(
        self,
        *args: Any,
        session_id: str,
        extra_docker_compose: Sequence[Path | str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._network_overlay_dir = tempfile.TemporaryDirectory(
            prefix="harbor-dense-network-"
        )
        self._network_claim: Path | None = None
        subnet = self._claim_subnet(session_id)
        overlay = Path(self._network_overlay_dir.name) / "network.json"
        overlay.write_text(
            json.dumps(
                {
                    "networks": {
                        "default": {
                            "ipam": {"config": [{"subnet": str(subnet)}]}
                        }
                    }
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        overlays = list(extra_docker_compose or ())
        overlays.append(overlay)
        try:
            super().__init__(
                *args,
                session_id=session_id,
                extra_docker_compose=overlays,
                **kwargs,
            )
        except BaseException:
            self._release_subnet(session_id)
            self._network_overlay_dir.cleanup()
            raise

    def _claim_subnet(self, session_id: str) -> ipaddress.IPv4Network:
        self._CLAIM_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
        start = int.from_bytes(
            hashlib.sha256(session_id.encode("utf-8")).digest()[:8], "big"
        ) % self._SLOTS
        for offset in range(self._SLOTS):
            slot = (start + offset) % self._SLOTS
            claim = self._CLAIM_ROOT / f"{slot}.claim"
            try:
                descriptor = os.open(
                    claim,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError:
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(session_id + "\n")
            self._network_claim = claim
            return ipaddress.IPv4Network((self._BASE + slot * 8, 29))
        raise RuntimeError("no dense Docker trial subnet is available")

    def _release_subnet(self, session_id: str) -> None:
        claim = self._network_claim
        self._network_claim = None
        if claim is None:
            return
        try:
            if claim.read_text(encoding="utf-8").strip() == session_id:
                claim.unlink(missing_ok=True)
        except OSError:
            pass

    async def stop(self, delete: bool) -> None:
        try:
            await super().stop(delete)
        finally:
            self._release_subnet(self.session_id)
            self._network_overlay_dir.cleanup()


class BenchmarkDockerEnvironment(DenseNetworkDockerEnvironment):
    """Host networking for trusted image BUILD only; Harbor owns run isolation."""
    def __init__(self, *args, extra_docker_compose=None, **kwargs):
        self._build_overlay = tempfile.TemporaryDirectory(prefix='benchmark-build-')
        path = Path(self._build_overlay.name) / 'build-network.json'
        path.write_text(json.dumps({'services': {'main': {'build': {'network': 'host'}}}}))
        super().__init__(*args, extra_docker_compose=[*(extra_docker_compose or []), path], **kwargs)


import functools
import shlex
import subprocess
import sys
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.verifier.verifier import Verifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'

def quoted_cli_flags(self):
    parts=[]
    for flag in self.CLI_FLAGS:
        value=self._resolved_flags.get(flag.kwarg)
        if value is None: continue
        if flag.format is not None:
            parts.append(flag.format.format(value=shlex.quote(str(value))))
        elif flag.type=='bool':
            if value: parts.append(flag.cli)
        else:
            parts.extend([flag.cli, shlex.quote(str(value))])
    return ' '.join(parts)

BaseInstalledAgent.build_cli_flags=quoted_cli_flags

# New benchmark reward.json includes non-numeric audit metadata. Keep that
# file intact while satisfying Harbor's numeric-only aggregated rewards model.
_parse_reward_json = Verifier._parse_reward_json
def numeric_reward_json(self):
    return {k: v for k, v in _parse_reward_json(self).items() if isinstance(v, (int, float))}
Verifier._parse_reward_json = numeric_reward_json

@functools.cache
def offline_probe():
    p = subprocess.run(['docker', 'run', '--rm', '--network', 'none', DockerEnvironment._EGRESS_CONTROL_KERNEL_PROBE_IMAGE, 'sh', '-c', DockerEnvironment._EGRESS_CONTROL_KERNEL_PROBE_SCRIPT], capture_output=True, timeout=30)
    return p.returncode == 0

DockerEnvironment._egress_control_kernel_support = staticmethod(offline_probe)

if __name__ == '__main__':
    from harbor.cli.main import app
    app()
