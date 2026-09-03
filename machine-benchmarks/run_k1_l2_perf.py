#!/usr/bin/env python3
"""Run K1 L2 bandwidth with non-multiplexed perf event groups."""
from __future__ import annotations

import argparse
import pathlib
import shlex
import subprocess

GROUPS = (
    ("execution", "cycles:u,instructions:u,load_inst:u,vector_load_inst:u"),
    ("cache", "L1-dcache-loads:u,L1-dcache-load-misses:u,l2_load_access:u,l2_load_miss:u"),
    ("l2-request", "l2_ar_channel_request:u,l2_ar_channel_stall_cycle:u,cycles:u"),
)


def run(command: list[str], *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, input=input_text, text=True, capture_output=True, check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", default="spacemit-k1")
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--cpus", default="0,1,2,3")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--duration-ms", type=int, default=250)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if not args.binary.is_file():
        parser.error(f"binary does not exist: {args.binary}")

    remote_dir = run(["ssh", args.remote, "mktemp -d /tmp/k1-l2.XXXXXX"]).stdout.strip()
    if not remote_dir:
        print("unable to create remote directory", flush=True)
        return 1
    try:
        remote_binary = f"{remote_dir}/k1-l2-bandwidth"
        copied = run(["scp", "-q", str(args.binary), f"{args.remote}:{remote_binary}"])
        if copied.returncode:
            print(copied.stderr, end="")
            return copied.returncode
        modes = [("single", "--mode single"), ("concurrent", "--mode concurrent")]
        if args.validate:
            modes = [("l1", "--mode single --single-bytes 16384"),
                     ("l2", "--mode single --single-bytes 262144"),
                     ("dram", "--mode single --single-bytes 16777216")]
        all_valid = True
        for mode_name, mode_args in modes:
            for group_name, events in GROUPS:
                command = (f"perf stat -x ';' --no-scale -e {shlex.quote(events)} -- "
                           f"{shlex.quote(remote_binary)} {mode_args} --cpus {shlex.quote(args.cpus)} "
                           f"--samples {args.samples} --duration-ms {args.duration_ms}")
                result = run(["ssh", args.remote, command])
                print(f"=== mode={mode_name} events={group_name} exit={result.returncode} ===")
                print(result.stdout, end="")
                print(result.stderr, end="")
                bad = result.returncode != 0 or "<not counted>" in result.stderr or "<not supported>" in result.stderr
                if not bad:
                    for line in result.stderr.splitlines():
                        fields = line.split(";")
                        if len(fields) >= 5 and fields[0] not in ("", "<not counted>"):
                            try:
                                running_percent = float(fields[4])
                            except ValueError:
                                continue
                            if running_percent < 99.0:
                                bad = True
                                print(f"INVALID PMU SCHEDULING: {line}")
                print("validity=INVALID" if bad else "validity=VALID")
                all_valid = all_valid and not bad
    finally:
        run(["ssh", args.remote, f"rm -rf -- {shlex.quote(remote_dir)}"])
    return 0 if all_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())