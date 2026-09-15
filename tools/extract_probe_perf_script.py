#!/usr/bin/env python3
"""Replay a saved perf.data on the target against the local probe binary.

`perf script` runs remotely with a re-uploaded copy of the local binary so the
recorded IPs resolve to the probe labels we just compiled in.
"""
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST = "spacemit-k1"


def main() -> int:
    artifacts = [Path(arg) for arg in sys.argv[1:]] or [ROOT / "build/profiles/cb-wmainpos"]
    for artifact in artifacts:
        for run_dir in sorted(artifact.glob("run-*")):
            if not (run_dir / "perf.data").exists():
                print(f"skip {run_dir}: no perf.data")
                continue
            match = re.search(r"remote_dir=([^\n]+)", (run_dir / "remote-command.sh").read_text())
            if not match:
                print(f"skip {run_dir}: remote_dir missing")
                continue
            remote_dir = shlex.split(match.group(1))[0]
            binary = run_dir / "ime-llama-bench"
            if not binary.exists():
                print(f"skip {run_dir}: local binary copy missing")
                continue
            subprocess.run(["ssh", HOST, "mkdir", "-p", "--", remote_dir], check=True)
            subprocess.run(["scp", "-q", str(binary), f"{HOST}:{remote_dir}/ime-llama-bench"], check=True)
            subprocess.run(["scp", "-q", str(run_dir / "perf.data"), f"{HOST}:{remote_dir}/replay.perf.data"], check=True)
            result = subprocess.run(
                ["ssh", HOST, f"perf script -i {shlex.quote(remote_dir + '/replay.perf.data')} -F ip,sym,symoff,dso,event 2>/dev/null"],
                check=True,
                text=True,
                capture_output=True,
            )
            (run_dir / "perf-script.txt").write_text(result.stdout)
            subprocess.run(["ssh", HOST, "rm", "-rf", "--", remote_dir], check=True)
            print(f"{run_dir} {len(result.stdout.splitlines())} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
