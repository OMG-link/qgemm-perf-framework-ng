#!/usr/bin/env python3
"""Collect scheduling-qualified K1 memory-path scaling samples over SSH."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess


def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, text=True, capture_output=True, check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", default="spacemit-k1")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--valid-samples", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path("memory-path-scaling.json"))
    args = parser.parse_args()
    if not args.binary.is_file():
        parser.error(f"binary does not exist: {args.binary}")

    remote_dir_result = run(["ssh", args.remote, "mktemp -d /tmp/k1-path.XXXXXX"])
    remote_dir = remote_dir_result.stdout.strip()
    if remote_dir_result.returncode or not remote_dir:
        print(remote_dir_result.stderr, end="")
        return 1
    remote_binary = f"{remote_dir}/k1-memory-path-bandwidth"
    rows: list[dict[str, object]] = []
    try:
        copied = run(["scp", "-q", str(args.binary), f"{args.remote}:{remote_binary}"])
        if copied.returncode:
            print(copied.stderr, end="")
            return copied.returncode
        cases: list[tuple[str, int, str, str]] = []
        for mode, rounds in (("l1", 65536), ("l2", 8192)):
            for workers in range(1, 9):
                cpus = ",".join(str(cpu) for cpu in range(workers))
                extra = "--bytes 98304" if mode == "l2" else ""
                cases.append((mode, rounds, cpus, extra))
        for cpus in ("0", "0,1", "0,1,2,3", "0,4", "0,1,4,5", "0,1,2,4,5,6", "0,1,2,3,4,5,6", "0,1,2,3,4,5,6,7"):
            cases.append(("dram", 16, cpus, ""))

        for mode, rounds, cpus, extra in cases:
            samples: list[float] = []
            attempts = 0
            while len(samples) < args.valid_samples and attempts < args.max_attempts:
                attempts += 1
                command = (f"{shlex.quote(remote_binary)} --mode {mode} --cpus {cpus} "
                           f"--rounds {rounds} {extra} --pmu execution --load-lmul 8")
                result = run(["ssh", args.remote, command])
                if result.returncode:
                    continue
                quality = re.search(r"quality .* sample_valid=(yes|no)", result.stdout)
                aggregate = re.search(r"aggregate .* B/cycle=([0-9.]+)", result.stdout)
                if quality and aggregate and quality.group(1) == "yes":
                    samples.append(float(aggregate.group(1)))
            row = {
                "mode": mode,
                "cpus": cpus,
                "workers": cpus.count(",") + 1,
                "rounds": rounds,
                "attempts": attempts,
                "valid_samples_Bpc": samples,
                "median_Bpc": statistics.median(samples) if samples else None,
                "minimum_Bpc": min(samples) if samples else None,
                "maximum_Bpc": max(samples) if samples else None,
                "complete": len(samples) == args.valid_samples,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    finally:
        run(["ssh", args.remote, f"rm -rf -- {shlex.quote(remote_dir)}"])
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())