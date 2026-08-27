#!/usr/bin/env python3
"""Automate no-instrumentation perf PC sampling for ime-llama benchmarks.

The tool deliberately keeps the benchmark binary unmodified. It runs perf on
an SSH target, preserves the remote evidence, and optionally aggregates
instruction samples into source-region ranges supplied in a JSON file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REMOTE = os.environ.get("DEPLOYMENT_SERVER", "spacemit-k1")
DEFAULT_BINARY = ROOT / "build" / "ime-llama-bench"
DEFAULT_REGIONS = ROOT / "tools" / "ime_perf" / "perf-regions.json"


class ToolError(RuntimeError):
    pass


def die(message: str) -> None:
    raise ToolError(message)


def run_local(argv: list[str], *, check: bool = True, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True)
    if check and result.returncode:
        raise ToolError(
            f"command failed ({result.returncode}): {shlex.join(argv)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def run_ssh(host: str, script: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, "bash -s"],
        input=script,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if check and result.returncode:
        raise ToolError(
            f"remote command failed on {host} ({result.returncode})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def remote_output(host: str, command: str) -> str:
    result = run_local(["ssh", "-o", "BatchMode=yes", host, command])
    return result.stdout.strip()


def scp_to(local_path: Path, host: str, remote_path: str) -> None:
    run_local(["scp", "-q", str(local_path), f"{host}:{remote_path}"])


def scp_from(host: str, remote_path: str, local_path: Path) -> bool:
    result = run_local(["scp", "-q", f"{host}:{remote_path}", str(local_path)], check=False)
    return result.returncode == 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    result = run_local(["git", *args], check=False)
    return result.stdout.strip() or "unknown"


def iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def parse_int(value: str) -> int:
    try:
        return int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value}") from exc


def positive(value: str) -> int:
    parsed = parse_int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_regions(path: Path, kernel: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    entry = data.get(kernel, [])
    selectors: dict[str, str | None] = {}
    if isinstance(entry, dict):
        selectors = {
            "dso": str(entry["dso"]) if entry.get("dso") else None,
            "symbol": str(entry["symbol"]) if entry.get("symbol") else None,
        }
        entry = entry.get("regions", [])
    regions: list[dict[str, Any]] = []
    names: set[str] = set()
    for item in entry:
        if not isinstance(item, dict):
            die(f"region entry must be an object: {item!r}")
        name = str(item.get("name", ""))
        if not name:
            die("region is missing name")
        if name in names:
            die(f"duplicate region name: {name}")
        names.add(name)
        dso = str(item.get("dso") or selectors.get("dso") or "")
        symbol = str(item.get("symbol") or selectors.get("symbol") or "")
        if not dso or not symbol:
            die(f"region {name} must define exact dso and symbol selectors")
        start = int(str(item["start"]), 0)
        end = int(str(item["end"]), 0)
        if end < start:
            die(f"region {name} has end before start")
        region = {
            **item,
            "name": name,
            "dso": dso,
            "symbol": symbol,
            "start_int": start,
            "end_int": end,
        }
        for previous in regions:
            same_selector = previous["dso"] == dso and previous["symbol"] == symbol
            overlaps = start <= previous["end_int"] and previous["start_int"] <= end
            if same_selector and overlaps:
                die(f"regions {previous['name']} and {name} overlap")
        regions.append(region)
    return regions


def benchmark_args(args: argparse.Namespace) -> list[str]:
    result = [
        "--kernel", args.kernel,
        "--m", str(args.m),
        "--n", str(args.n),
        "--k", str(args.k),
        "--warmup", str(args.warmup),
        "--samples", str(args.samples),
        "--iterations", str(args.iterations),
        "--no-verify" if not args.verify else "--verify",
    ]
    # The benchmark currently supports --no-verify but not an explicit --verify.
    if args.verify:
        result.remove("--verify")
    return result


def make_remote_script(
    remote_dir: str,
    event: str,
    period: int,
    cpu: int,
    binary_args: list[str],
    call_graph: str,
) -> str:
    q = shlex.quote
    record_options = ["perf", "record", "-q", "-e", event, "-c", str(period)]
    if call_graph != "none":
        record_options += ["--call-graph", call_graph]
    # Keep the profiled process layout deterministic without changing the
    # remote system-wide ASLR setting.  This also matches standalone runs that
    # are launched with `setarch -R` when address-layout reproducibility matters.
    record_options += [
        "-o", f"{remote_dir}/perf.data", "--", "setarch", "-R",
        "taskset", "-c", str(cpu), f"{remote_dir}/ime-llama-bench",
    ]
    record_options += binary_args
    record_command = " ".join(q(item) for item in record_options)
    report_command = (
        f"perf report --stdio -i {q(remote_dir + '/perf.data')} "
        "--sort symbol,dso --percent-limit 0 --stdio-color never "
        f">{q(remote_dir + '/perf-report.txt')} 2>{q(remote_dir + '/perf-report.stderr')}"
    )
    annotate_command = (
        f"perf annotate --stdio -i {q(remote_dir + '/perf.data')} "
        "--percent-limit 0 --stdio-color never --demangle "
        f">{q(remote_dir + '/perf-annotate.txt')} 2>{q(remote_dir + '/perf-annotate.stderr')}"
    )
    return f"""#!/bin/sh
set -u
remote_dir={q(remote_dir)}
{{
    echo 'captured_at='$(date -Is)
    echo 'hostname='$(hostname)
    uname -a
    echo 'perf_version:'
    perf --version 2>&1
    echo 'perf_event_paranoid:'
    cat /proc/sys/kernel/perf_event_paranoid 2>&1 || true
    echo 'randomize_va_space:'
    cat /proc/sys/kernel/randomize_va_space 2>&1 || true
    echo 'profile_execution:'
    echo 'setarch -R taskset -c {cpu} ime-llama-bench'
    echo 'cpu_governor:'
    cat /sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor 2>&1 || true
    echo 'cpu_min_freq:'
    cat /sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_min_freq 2>&1 || true
    echo 'cpu_max_freq:'
    cat /sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_max_freq 2>&1 || true
    echo 'cpu_cur_freq:'
    cat /sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq 2>&1 || true
    echo 'cpu_model:'
    grep -m1 -E '^(model name|uarch|cpudesc)' /proc/cpuinfo 2>&1 || true
}} >"$remote_dir/environment.txt" 2>&1
sha256sum "$remote_dir/ime-llama-bench" >"$remote_dir/remote-binary.sha256" 2>&1 || true
set +e
{record_command} >"$remote_dir/benchmark.stdout" 2>"$remote_dir/benchmark.stderr"
record_rc=$?
printf '%s\\n' "$record_rc" >"$remote_dir/record.exitcode"
if test -s "$remote_dir/perf.data"; then
    {report_command}
    {annotate_command}
fi
exit "$record_rc"
"""


def copy_remote_artifacts(host: str, remote_dir: str, destination: Path) -> None:
    names = [
        "perf.data",
        "environment.txt",
        "remote-binary.sha256",
        "benchmark.stdout",
        "benchmark.stderr",
        "record.exitcode",
        "perf-report.txt",
        "perf-report.stderr",
        "perf-annotate.txt",
        "perf-annotate.stderr",
    ]
    for name in names:
        scp_from(host, f"{remote_dir}/{name}", destination / name)


def remote_cleanup(host: str, remote_dir: str) -> None:
    run_local(
        ["ssh", "-o", "BatchMode=yes", host, "rm", "-rf", "--", remote_dir],
        check=False,
    )


def make_manifest(args: argparse.Namespace, binary: Path, root: Path) -> dict[str, Any]:
    return {
        "tool": "tools/ime_perf/ime_perf.py",
        "tool_version": 1,
        "created_at": iso_now(),
        "git_revision": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "kernel": args.kernel,
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "warmup": args.warmup,
        "samples": args.samples,
        "iterations": args.iterations,
        "verify": args.verify,
        "remote": args.remote,
        "cpu": args.cpu,
        "event": args.event,
        "period": args.period,
        "repetitions": args.repetitions,
        "call_graph": args.call_graph,
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "regions_file": str(args.regions_file),
        "artifact_root": str(root),
    }


def profile_once(args: argparse.Namespace, binary: Path, run_dir: Path, repetition: int) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    remote_dir = remote_output(args.remote, "mktemp -d /tmp/ime-perf.XXXXXX")
    if not remote_dir:
        die("remote mktemp returned an empty directory")
    try:
        scp_to(binary, args.remote, f"{remote_dir}/ime-llama-bench")
        command = make_remote_script(
            remote_dir,
            args.event,
            args.period,
            args.cpu,
            benchmark_args(args),
            args.call_graph,
        )
        (run_dir / "remote-command.sh").write_text(command)
        result = run_ssh(args.remote, command, check=False)
        copy_remote_artifacts(args.remote, remote_dir, run_dir)
        (run_dir / "ssh.stdout").write_text(result.stdout)
        (run_dir / "ssh.stderr").write_text(result.stderr)
        shutil.copy2(binary, run_dir / "ime-llama-bench")
        (run_dir / "binary.sha256").write_text(sha256(binary) + "  ime-llama-bench\n")
        (run_dir / "repetition.txt").write_text(str(repetition) + "\n")
        return result.returncode
    finally:
        remote_cleanup(args.remote, remote_dir)


def parse_symbol_report(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)%\s+\[.\]\s+(.*?)\s{2,}(\S+)\s+-\s+-\s*$")
    for line in path.read_text(errors="replace").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        symbol = match.group(2).strip()
        if symbol:
            rows.append({
                "percent": float(match.group(1)),
                "dso": match.group(3),
                "symbol": symbol,
            })
    return rows


ANNOTATE_SECTION_RE = re.compile(
    r"^\s*Percent\s*\|\s*Source code & Disassembly of (?P<dso>.+?) "
    r"for (?P<event>.+?) \((?P<samples>\d+) samples, percent: (?P<metric>[^)]+)\)\s*$"
)
ANNOTATE_SYMBOL_RE = re.compile(
    r"^\s*:\s*\d+\s+[0-9A-Fa-f]+\s+<(?P<symbol>.*)>:\s*$"
)
ANNOTATE_INSTRUCTION_RE = re.compile(
    r"^\s*(?P<percent>(?:\d+(?:\.\d*)?|\.\d+))\s*:\s+"
    r"(?P<address>[0-9A-Fa-f]+):(?:\s|$)"
)
PERCENT_TOLERANCE = 0.5


def parse_annotate_regions(
    path: Path, regions: list[dict[str, Any]]
) -> tuple[dict[str, float], list[str]]:
    values = {str(region["name"]): 0.0 for region in regions}
    diagnostics: list[str] = []
    if not path.exists():
        return values, [f"annotate file missing: {path}"]

    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        section_match = ANNOTATE_SECTION_RE.match(line)
        if section_match:
            current = {
                "dso": section_match.group("dso").strip(),
                "event": section_match.group("event").strip(),
                "samples": int(section_match.group("samples")),
                "symbol": None,
                "line": line_number,
            }
            sections.append(current)
            continue
        if current is None:
            continue

        symbol_match = ANNOTATE_SYMBOL_RE.match(line)
        if symbol_match:
            if current["symbol"] is None:
                current["symbol"] = symbol_match.group("symbol").strip()
            continue

        instruction_match = ANNOTATE_INSTRUCTION_RE.match(line)
        if not instruction_match or current["symbol"] is None:
            continue
        try:
            percent = float(instruction_match.group("percent"))
            address = int(instruction_match.group("address"), 16)
        except ValueError:
            diagnostics.append(f"line {line_number}: malformed instruction row")
            continue
        if not math.isfinite(percent) or percent < 0:
            diagnostics.append(f"line {line_number}: invalid instruction percentage {percent!r}")
            continue

        for region in regions:
            if current["dso"] != region["dso"] or current["symbol"] != region["symbol"]:
                continue
            if region["start_int"] <= address <= region["end_int"]:
                values[str(region["name"])] += percent
                break

    for region in regions:
        name = str(region["name"])
        matching_sections = [
            section for section in sections
            if section["dso"] == region["dso"] and section["symbol"] == region["symbol"]
        ]
        if not matching_sections:
            diagnostics.append(
                f"region {name!r} did not match annotate section "
                f"dso={region['dso']!r}, symbol={region['symbol']!r}"
            )
        elif len(matching_sections) > 1:
            diagnostics.append(
                f"region {name!r} matched {len(matching_sections)} annotate sections "
                f"for dso={region['dso']!r}, symbol={region['symbol']!r}"
            )
        if values[name] > 100.0 + PERCENT_TOLERANCE:
            diagnostics.append(f"region {name!r} exceeds 100%: {values[name]:.2f}%")

    selector_totals: dict[tuple[str, str], float] = {}
    selector_names: dict[tuple[str, str], list[str]] = {}
    for region in regions:
        selector = (str(region["dso"]), str(region["symbol"]))
        name = str(region["name"])
        selector_totals[selector] = selector_totals.get(selector, 0.0) + values[name]
        selector_names.setdefault(selector, []).append(name)
    for selector, total in selector_totals.items():
        if len(selector_names[selector]) > 1 and total > 100.0 + PERCENT_TOLERANCE:
            diagnostics.append(
                f"regions {selector_names[selector]} for dso={selector[0]!r}, "
                f"symbol={selector[1]!r} sum to {total:.2f}%"
            )
    return values, diagnostics


def summarize_runs(root: Path, manifest: dict[str, Any], regions_file: Path) -> dict[str, Any]:
    regions = load_regions(regions_file, manifest["kernel"])
    run_dirs = sorted(path for path in root.glob("run-*") if path.is_dir())
    runs: list[dict[str, Any]] = []
    warnings: list[str] = []
    for run_dir in run_dirs:
        region_values, diagnostics = parse_annotate_regions(run_dir / "perf-annotate.txt", regions)
        symbols = parse_symbol_report(run_dir / "perf-report.txt")
        exitcode = (run_dir / "record.exitcode").read_text().strip() if (run_dir / "record.exitcode").exists() else "unknown"
        runs.append({
            "run": run_dir.name,
            "record_exitcode": exitcode,
            "regions_percent": region_values,
            "parser_diagnostics": diagnostics,
            "top_symbols": symbols[:20],
        })
        warnings.extend(f"{run_dir.name}: {diagnostic}" for diagnostic in diagnostics)
    region_summary: dict[str, Any] = {}
    for region in regions:
        name = str(region["name"])
        samples = [run["regions_percent"][name] for run in runs if name in run["regions_percent"]]
        region_summary[name] = {
            "dso": region["dso"],
            "symbol": region["symbol"],
            "start": region["start"],
            "end": region["end"],
            "description": region.get("description", ""),
            "runs_percent": samples,
            "mean_percent": statistics.mean(samples) if samples else None,
            "stdev_percent": statistics.stdev(samples) if len(samples) > 1 else 0.0 if samples else None,
        }
    if not regions:
        warnings.append(f"no configured address regions for {manifest['kernel']}")
    return {
        "kernel": manifest["kernel"],
        "regions_file": str(regions_file),
        "runs": runs,
        "regions": region_summary,
        "warnings": warnings,
    }


def write_summary(root: Path, manifest: dict[str, Any], summary: dict[str, Any]) -> None:
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        f"# perf PC profile: `{manifest['kernel']}`",
        "",
        f"- Remote: `{manifest['remote']}`, CPU: `{manifest['cpu']}`",
        f"- Shape: `M={manifest['shape']['m']}, N={manifest['shape']['n']}, K={manifest['shape']['k']}`",
        f"- Event: `{manifest['event']}`, period: `{manifest['period']}`, repetitions: `{manifest['repetitions']}`",
        f"- Binary SHA-256: `{manifest['binary_sha256']}`",
        "",
        "The percentages below are sums of PC samples in the exact configured DSO/symbol section and address ranges. They are local percentages from `perf annotate`, not hardware cycle counts.",
        "",
        "## Configured regions",
        "",
        "| Region | DSO | Symbol | Address range | Mean sample share | Stddev |",
        "|---|---|---|---:|---:|---:|",
    ]
    for name, value in summary["regions"].items():
        mean = "n/a" if value["mean_percent"] is None else f"{value['mean_percent']:.2f}%"
        stdev = "n/a" if value["stdev_percent"] is None else f"{value['stdev_percent']:.2f} pp"
        lines.append(
            f"| `{name}` | `{value['dso']}` | `{value['symbol']}` | "
            f"`{value['start']}..{value['end']}` | {mean} | {stdev} |"
        )
    lines += ["", "## Run status", "", "| Run | perf exit code |", "|---|---:|"]
    for run in summary["runs"]:
        lines.append(f"| `{run['run']}` | `{run['record_exitcode']}` |")
    if summary["warnings"]:
        lines += ["", "## Warnings", ""] + [f"- {warning}" for warning in summary["warnings"]]
    (root / "summary.md").write_text("\n".join(lines) + "\n")


def default_output(args: argparse.Namespace) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}-{args.kernel}-m{args.m}-n{args.n}-k{args.k}"
    return ROOT / "tools" / "ime_perf" / ".artifacts" / "perf-cpu-clock" / name


def validate_profile(args: argparse.Namespace) -> None:
    if args.kernel == "all":
        die("profile requires one kernel; run each kernel separately for independent artifacts")
    if args.samples <= 0 or args.iterations <= 0 or args.repetitions <= 0:
        die("samples, iterations, and repetitions must be positive")
    if args.period <= 0:
        die("period must be positive")
    if args.cpu < 0:
        die("cpu must be non-negative")


def command_preflight(args: argparse.Namespace) -> int:
    script = f"""set +e
echo 'host:'; hostname
uname -a
echo 'perf:'; perf --version 2>&1
echo 'perf_event_paranoid:'; cat /proc/sys/kernel/perf_event_paranoid 2>&1
echo 'cpu:'; taskset -pc $$ 2>&1
echo 'cpufreq:'
for f in /sys/devices/system/cpu/cpu{args.cpu}/cpufreq/scaling_driver /sys/devices/system/cpu/cpu{args.cpu}/cpufreq/scaling_governor /sys/devices/system/cpu/cpu{args.cpu}/cpufreq/scaling_min_freq /sys/devices/system/cpu/cpu{args.cpu}/cpufreq/scaling_max_freq /sys/devices/system/cpu/cpu{args.cpu}/cpufreq/scaling_cur_freq; do
  printf '%s=' "$f"; cat "$f" 2>&1 || true
done
echo 'events:'; perf list 2>/dev/null | grep -E 'cpu-clock|task-clock|cycles|instructions' | head -30
"""
    result = run_ssh(args.remote, script, check=False)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


def command_profile(args: argparse.Namespace) -> int:
    validate_profile(args)
    binary = resolve_path(args.binary)
    if args.build:
        run_local(["./compile-and-test.sh", "build"])
    if not binary.is_file() or not os.access(binary, os.X_OK):
        die(f"benchmark binary is not executable: {binary}; pass --build or build it first")
    output = resolve_path(args.out) if args.out else default_output(args)
    output.mkdir(parents=True, exist_ok=False)
    manifest = make_manifest(args, binary, output)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "command.txt").write_text(" ".join(shlex.quote(item) for item in sys.argv) + "\n")
    exitcodes = []
    for repetition in range(1, args.repetitions + 1):
        run_dir = output / f"run-{repetition:02d}"
        exitcodes.append(profile_once(args, binary, run_dir, repetition))
    summary = summarize_runs(output, manifest, args.regions_file)
    write_summary(output, manifest, summary)
    print(f"artifact: {output}")
    print(f"summary:  {output / 'summary.md'}")
    if any(exitcodes):
        print(f"warning: remote perf exit codes: {exitcodes}", file=sys.stderr)
        return 1
    return 0


def command_report(args: argparse.Namespace) -> int:
    root = resolve_path(args.artifact)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        die(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    regions_file = resolve_path(args.regions_file) if args.regions_file else Path(manifest["regions_file"])
    summary = summarize_runs(root, manifest, regions_file)
    write_summary(root, manifest, summary)
    print(root / "summary.md")
    return 0


def add_common_remote(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--remote", default=DEFAULT_REMOTE, help=f"SSH target (default: {DEFAULT_REMOTE})")
    parser.add_argument("--cpu", type=nonnegative, default=0, help="CPU to pin the benchmark to")


def nonnegative(value: str) -> int:
    parsed = parse_int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="check remote perf and CPU prerequisites")
    add_common_remote(preflight)
    preflight.set_defaults(function=command_preflight)

    profile = subparsers.add_parser("profile", help="record repeated remote PC profiles and aggregate them")
    add_common_remote(profile)
    profile.add_argument("--kernel", required=True)
    profile.add_argument("--m", type=positive, required=True)
    profile.add_argument("--n", type=positive, required=True)
    profile.add_argument("--k", type=positive, required=True)
    profile.add_argument("--warmup", type=positive, default=2)
    profile.add_argument("--samples", type=positive, default=1)
    profile.add_argument("--iterations", type=positive, default=20)
    profile.add_argument("--repetitions", type=positive, default=3)
    profile.add_argument("--event", default="cpu-clock:u")
    profile.add_argument("--period", type=positive, default=10000)
    profile.add_argument("--call-graph", choices=("none", "fp", "dwarf"), default="none")
    profile.add_argument("--verify", action="store_true", help="include benchmark correctness verification")
    profile.add_argument("--build", action="store_true", help="build with compile-and-test.sh before profiling")
    profile.add_argument("--binary", default=str(DEFAULT_BINARY))
    profile.add_argument("--out", help="local artifact directory; must not already exist")
    profile.add_argument("--regions-file", type=resolve_path, default=DEFAULT_REGIONS)
    profile.set_defaults(function=command_profile)

    report = subparsers.add_parser("report", help="rebuild summary files from a saved artifact")
    report.add_argument("artifact")
    report.add_argument("--regions-file", type=resolve_path)
    report.set_defaults(function=command_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.function(args))
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
