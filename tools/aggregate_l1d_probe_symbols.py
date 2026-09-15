#!/usr/bin/env python3
"""Aggregate `perf script -F ip,sym,symoff,dso` samples into IME L1D probes.

Symbol-based variant of `aggregate_l1d_probe_samples.py`: the probe macros emit
globally visible `<probe>_start`/`<probe>_end` symbols, and the 64-NOP sled
immediately follows `_start`, so a sample whose symbol is `<probe>_start` and
whose symoff is smaller than the sled length belongs to that probe.  This avoids
having to reconstruct the PIE load bias.
"""
import argparse
import collections
import re
import subprocess
from pathlib import Path

START_RE = re.compile(r"^(ime_l1d_probe_(m8b(?:_cb|_dpu)?)_(.+)_([0-9]+))_start$")


def read_symbols(elf: Path, nm: str):
    output = subprocess.check_output([nm, "-n", "--defined-only", str(elf)], text=True)
    symbols = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 3:
            try:
                symbols[fields[-1]] = int(fields[0], 16)
            except ValueError:
                pass
    probes = {}
    for name, start in symbols.items():
        match = START_RE.match(name)
        if not match:
            continue
        end_name = name[:-len("_start")] + "_end"
        if end_name not in symbols:
            raise SystemExit(f"missing end symbol for {name}")
        probes[name] = (start, symbols[end_name], match.groups())
    return probes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("elf", type=Path)
    parser.add_argument("samples", type=Path, nargs="?")
    parser.add_argument("--nm", default="llvm-nm")
    parser.add_argument("--kernel-symbol", action="append", default=[], help="restrict the denominator to these symbols")
    args = parser.parse_args()

    probes = read_symbols(args.elf, args.nm)
    import sys
    stream = args.samples.open() if args.samples else sys.stdin
    line_re = re.compile(r"^(\S+:)?\s*([0-9a-f]+)\s+(\S+?)(?:\+0x([0-9a-f]+))?\s+\((.*?)\)\s*$")
    per_probe = collections.Counter()
    denominator = collections.Counter()
    unmatched = 0
    total = 0
    try:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            match = line_re.match(line)
            if not match:
                continue
            symbol = match.group(3)
            symoff = int(match.group(4), 16) if match.group(4) else 0
            total += 1
            if args.kernel_symbol:
                for wanted in args.kernel_symbol:
                    if symbol.startswith(wanted):
                        denominator[symbol] += 1
                        break
                else:
                    denominator["<other>"] += 1
            else:
                denominator[symbol] += 1
            if symbol in probes:
                start, end, _ = probes[symbol]
                if symoff < end - start:
                    per_probe[symbol] += 1
                    continue
            unmatched += 1
    finally:
        if args.samples:
            stream.close()

    per_category = collections.Counter()
    print("probe samples kernel category")
    for name, (start, end, (probe, kernel, category, index)) in sorted(probes.items(), key=lambda item: item[1][0]):
        count = per_probe[name]
        per_category[(kernel, category)] += count
        print(f"{name} {count} {kernel} {category} sled={end - start}")
    print("\ncategory samples kernel")
    for (kernel, category), count in sorted(per_category.items()):
        print(f"{category} {count} {kernel}")
    print("\ndenominator symbol samples")
    for symbol, count in denominator.most_common(20):
        print(f"{count} {symbol}")
    print(f"\ntotal {total}\nmatched {sum(per_probe.values())}\nunmatched {unmatched}")


if __name__ == "__main__":
    main()
