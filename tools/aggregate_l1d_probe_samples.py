#!/usr/bin/env python3
"""Aggregate `perf script -F ip` sample PCs into IME L1D probe ranges."""

import argparse
import collections
import re
import subprocess
import sys
from pathlib import Path

START_RE = re.compile(r"^(ime_l1d_probe_(m8b(?:_cb)?)_(.+)_([0-9]+))_start$")


def read_symbols(elf: Path, nm: str):
    try:
        output = subprocess.check_output([nm, "-n", "--defined-only", str(elf)], text=True)
    except FileNotFoundError:
        fallback = "nm" if nm != "nm" else None
        if not fallback:
            raise
        output = subprocess.check_output([fallback, "-n", "--defined-only", str(elf)], text=True)

    symbols = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 3:
            try:
                symbols[fields[-1]] = int(fields[0], 16)
            except ValueError:
                pass

    probes = []
    for name, start in symbols.items():
        match = START_RE.match(name)
        if not match:
            continue
        probe, kernel, category, index = match.groups()
        end_name = probe + "_end"
        if end_name not in symbols:
            raise SystemExit(f"missing end symbol for {probe}")
        end = symbols[end_name]
        if end <= start:
            raise SystemExit(f"invalid symbol range for {probe}: {start:#x}..{end:#x}")
        probes.append((start, end, probe, kernel, category, int(index)))
    if not probes:
        raise SystemExit("no ime_l1d_probe_*_{start,end} symbols found")
    return sorted(probes)


def sample_pcs(stream, base: int):
    for line_number, line in enumerate(stream, 1):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        token = fields[0].rstrip(":")
        try:
            yield int(token, 16) - base
        except ValueError:
            print(f"warning: line {line_number}: expected PC first (use `perf script -F ip`)", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("elf", type=Path, help="target ELF containing probe symbols")
    parser.add_argument("samples", nargs="?", type=Path, help="`perf script -F ip` output (default: stdin)")
    parser.add_argument("--base", type=lambda value: int(value, 0), default=0, help="runtime ELF load bias to subtract from sample PCs")
    parser.add_argument("--nm", default="llvm-nm", help="nm-compatible symbol reader (default: llvm-nm, then nm)")
    args = parser.parse_args()

    probes = read_symbols(args.elf, args.nm)
    per_probe = collections.Counter()
    unmatched = 0
    stream = args.samples.open() if args.samples else sys.stdin
    try:
        for pc in sample_pcs(stream, args.base):
            for start, end, probe, kernel, category, index in probes:
                if start <= pc < end:
                    per_probe[probe] += 1
                    break
            else:
                unmatched += 1
    finally:
        if args.samples:
            stream.close()

    per_category = collections.Counter()
    print("probe samples kernel category range")
    for start, end, probe, kernel, category, index in probes:
        count = per_probe[probe]
        per_category[(kernel, category)] += count
        print(f"{probe} {count} {kernel} {category} [{start:#x},{end:#x})")
    print("\ncategory samples kernel")
    for (kernel, category), count in sorted(per_category.items()):
        print(f"{category} {count} {kernel}")
    print(f"\nmatched {sum(per_probe.values())}\nunmatched {unmatched}")


if __name__ == "__main__":
    main()
