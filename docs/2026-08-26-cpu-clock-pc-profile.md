# No-instrumentation `cpu-clock` PC profile — 2026-08-26

## Scope

This experiment profiles the unmodified optimized binary for
`q4_0-q8_0-IME-m8b` at the representative shape:

```text
M = 480, N = 1536, K = 1536
```

The goal is to identify the distribution of execution time inside the three
source regions of `kernels/q4_0-q8_0-IME-m8b/kernel.cpp` without inserting
`rdcycle`, logging, or other measurement code into the kernel.

## Platform and build

```text
Host:             bpi-f3 / SpacemiT X60 / riscv64
Kernel:           Linux 6.6.63
Remote perf:      6.6.63.gd5b2ef2c6af4
perf_event_paranoid: 2
CPU:              0
Governor:         performance
min/max/current:  614.4 / 1600 / 1600 MHz
Build:            CMake Release, fixed 256-bit RVV length
Binary SHA-256:   f5b782b8c88d908b6637280d053e8c6c4ad619d2de00805f73ee157b61c2aef5
```

The source tree was clean apart from the profiling-tool changes. The profiled
binary was copied to the target and retained in each local artifact directory
so the binary and `perf.data` remain matched.

## Method

The reusable driver is `tools/ime_perf/ime_perf.py`. It performs the following steps:

1. Runs a remote preflight and records kernel, perf, permission, CPU, and
   cpufreq information.
2. Uses the existing Release binary without recompiling or modifying kernel
   source.
3. Copies the exact binary to a remote temporary directory.
4. Runs `perf record -e cpu-clock:u -c 10000` under `taskset -c 0`.
5. Preserves `perf.data`, stdout/stderr, environment, the exact binary,
   `perf report`, and `perf annotate` output.
6. Repeats the profile three times and aggregates configured instruction
   ranges into mean and standard deviation.

The benchmark command used by the driver was equivalent to:

```bash
build/ime-llama-bench \
  --kernel q4_0-q8_0-IME-m8b \
  --m 480 --n 1536 --k 1536 \
  --warmup 2 --samples 1 --iterations 20 --no-verify
```

`cpu-clock` is a software time-sampling event, not the benchmark's hardware
cycle counter. Because CPU0 remained at 1.6 GHz, its sample distribution is a
stable proxy for relative time/cycle distribution. It must not be reported as
an exact hardware cycle count or as FMA/cycle.

## Results

The kernel function accounted for 90.36% of all samples on average across the
three runs. The configured ranges were mapped from the current uninstrumented
Release disassembly:

| Source region | Address range | Mean local sample share | Stddev |
|---|---:|---:|---:|
| L51–104: int8 unpack and `vmadot` inner loop | `0xe804..0xe8fc` | **40.12%** | 0.17 pp |
| L105–170: int32-to-float conversion and buffer unpack stores | `0xe908..0xea0e` | **28.05%** | 0.12 pp |
| L172–213: batch scale/FMA reduction | `0xea16..0xeb4c` | **21.31%** | 0.31 pp |
| Other instructions in the kernel | — | approximately 10.52% | — |

The three configured ranges together account for 89.48% of local samples in
the kernel. The remainder includes accumulator setup, outer-loop/control-flow
work, and final writeback that is outside the requested source ranges.

The three runs produced the following local shares:

| Run | L51–104 | L105–170 | L172–213 | Kernel symbol share of process |
|---|---:|---:|---:|---:|
| 1 | 40.00% | 27.94% | 21.63% | 90.25% |
| 2 | 40.04% | 28.03% | 21.28% | 90.47% |
| 3 | 40.31% | 28.18% | 21.02% | 90.36% |

The most prominent sampled instructions correspond to the following
operations:

- L51–104: Q4 unpacking, repeated `vsetvli/vsetivli`, vector loads, and the
  `smt.vmadot` sequence.
- L105–170: `vfcvt.f.x.v`, masked/non-contiguous `vse32.v`, and address
  generation for the reduction buffer.
- L172–213: fp16/fp32 scale loads, widening conversion, `vfmul.vf`, and
  `vfmacc.vv` inside the two reduction loops.

## Reproduction

Check the target without modifying the kernel:

```bash
python3 tools/ime_perf/ime_perf.py preflight --remote spacemit-k1 --cpu 0
```

Run three profiles and create a local artifact directory:

```bash
python3 tools/ime_perf/ime_perf.py profile \
  --remote spacemit-k1 --cpu 0 \
  --kernel q4_0-q8_0-IME-m8b \
  --m 480 --n 1536 --k 1536 \
  --warmup 2 --samples 1 --iterations 20 \
  --repetitions 3 \
  --event cpu-clock:u --period 10000
```

The default artifact path is:

```text
tools/ime_perf/.artifacts/perf-cpu-clock/<timestamp>-<kernel>-m<m>-n<n>-k<k>/
```

Rebuild reports from an existing artifact without contacting the target:

```bash
python3 tools/ime_perf/ime_perf.py report tools/ime_perf/.artifacts/perf-cpu-clock/<run>
```

For another kernel, add its instruction ranges to
`tools/ime_perf/perf-regions.json`. Each kernel entry must bind regions to the
exact annotated DSO and demangled symbol; an address alone is unsafe because
different ELF images can reuse the same relative address. Address ranges are
tied to the exact binary and must be refreshed whenever code generation
changes. The parser records diagnostics when the selected section is missing
or configured percentages exceed a valid local-period total. If no ranges are
configured, the tool still preserves and reports symbol-level hotspots.

## Why this method is preferred over temporary instrumentation

A temporary `rdcycle` bracket measures only an instrumented binary. Even after
subtracting an empty-bracket cost, it can change register allocation, stack
layout, scheduling, and code-cache behavior. In the earlier m8 experiment,
the corrected temporary-instrumentation estimate was 44.60%, 27.92%, and
21.84%, while the unmodified PC profile reported 40.12%, 28.05%, and 21.31%.
The first-region difference demonstrates that bracket subtraction does not
remove all compiler perturbation.

On this target, `perf stat` can count hardware cycles, but
`perf record -e cycles` produces no overflow samples. Therefore the tool uses
`cpu-clock` for no-instrumentation PC sampling and keeps hardware cycle totals
as a separate benchmark metric. Repeated profiles, fixed CPU/frequency, long
explicit iterations, and preserved raw artifacts are required for stable
comparisons.

## Artifacts and extension points

Each profile stores:

```text
manifest.json             # command, shape, host, event, hashes, tool version
command.txt
run-*/perf.data           # raw perf evidence for every repetition
run-*/perf-report.txt
run-*/perf-annotate.txt
run-*/benchmark.stdout
run-*/environment.txt
run-*/ime-llama-bench      # exact profiled binary
summary.json
summary.md
```

Region percentages are computed only after selecting the exact DSO and symbol
section from `perf-annotate.txt`; they are not process-wide percentages. Each
run records `parser_diagnostics` in `summary.json`, and missing sections,
invalid data, or percentages over the valid local-period total are surfaced
in the summary warnings instead of being silently treated as valid samples.

`tools/ime_perf/ime_perf.py` intentionally uses only Python's standard library. New
sampling events, call-graph modes, kernels, and region maps can be added
without changing benchmark or kernel source. The artifact manifest is versioned
so future report formats can be introduced without losing the raw evidence.
