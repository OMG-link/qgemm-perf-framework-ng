# Q4_0 x Q8_0 IME-RR inner-loop per-region cycles — 2026-09-29

## Scope

Measure the average cycles consumed per inner-loop iteration by the three source
regions requested from
`kernels/q4_0-q8_0-IME-RR/implementations/smt-ime2/kernel.cpp`:

```text
L76-L96    barrier + activation loads + int32 clear + eight int8 smt.vmadot
L98-L106   tile_c address + fp32 clear + four fp16 smt.vfwmadot
L108-L119  four REDUCE_GROUP(int32->fp32 convert, C load, vfmacc, C store)
```

The loop body is the `#pragma clang loop unroll(disable)` loop at `L74-L121`, so
one "iteration" is one execution of that loop body, i.e. one M8 activation tile
consumed against the register-resident N32 x K32 weight tile.

## Platform and build

```text
Platform ID:            spacemit-k3-a100
Host:                   spacemit-k3 (k3-pico-itx), Linux 6.18.3-generic, riscv64
CPU:                    SpacemiT A100, pinned CPU 8 (first A100 core)
Remote perf:            6.18.3.gaa43092a60c9, perf_event_paranoid=2
CPU frequency governor: userspace
Min/max/current freq:   614400 / 1800000 / 1800000 kHz
Architecture:           rv64gcv_zvfh_zicbop_zba_zvl1024b_xsmtvdotii
Tune:                   spacemit-a100
Build type:             CMake Release, fixed 1024-bit VLEN
Base commit:            18fad78a6047a1097eaaea3c055d05487c39293b
Binary SHA-256:         069b9c9f475d4cf8da62a40cb22b86414c0dbaa937ec176e9be70c4de1f68dfe
```

`echo 0 > /proc/set_ai_thread` runs before `taskset` so the profiled process and
its child can enter the A100 CPU set 8-15.

## Method

No instrumentation is added to the kernel. Cycles are reconstructed from
no-instrumentation PC sampling exactly as in
[2026-08-26-cpu-clock-pc-profile.md](2026-08-26-cpu-clock-pc-profile.md), with one
addition: the same invocation is also measured with an external hardware counter
so the sample share can be converted into absolute cycles.

1. `perf stat -e cycles:u` over the exact benchmark invocation gives the total
   process cycles `C_proc` (event running ratio 100%).
2. `perf record -e cpu-clock:u -c 10000` over the same invocation, `setarch -R`,
   `taskset -c 8`, produces the PC sample distribution.
3. `perf annotate` reports per-instruction shares local to the profiled symbol.
   Summing the rows whose absolute PC falls inside a region gives that region's
   share of the kernel symbol, and `perf report` gives the kernel symbol's share
   of the whole process.
4. `cycles(region) = C_proc x kernel_share x region_share`.
5. Divide by the total number of inner-loop iterations executed by the profiled
   process.

The driver is the extended `tools/pfng_perf/pfng_perf.py`
(`--platform spacemit-k3-a100`, `--cycles-event cycles:u`), and the region map is
the `q4_0-q8_0-IME-RR` entry of `tools/pfng_perf/perf-regions.json`.

## Address map

Derived from the final linked binary
(`llvm-objdump -d --demangle build/spacemit-k3-a100/pfng-bench`); the three spans
are contiguous and cover `0x17464..0x17530`, function offset `+0x106..+0x1d2`.
Addresses are inclusive.

| Source region | Absolute PC range | Function offset | Generated instructions |
|---|---|---|---|
| `L76-96` | `0x17464..0x174a8` | `+0x106..+0x14a` | `vl1r.v` x2, `addi`, `vsetvli e32,m2`, `vmv.v.i` x4, `vsetvli e8`, `smt.vmadot` x4, `vsetvli e8`, `smt.vmadot` x4 |
| `L98-106` | `0x174ac..0x174d4` | `+0x14e..+0x176` | `vsetvli e32,m2`, `vmv.v.i` x4, `vsetvli e16`, `smt.vfwmadot` x4, `add` (the `L98` `tile_c` address, scheduled here) |
| `L108-119` | `0x174d8..0x17530` | `+0x17a..+0x1d2` | per group: `vfcvt.f.x.v`, `vl2re32.v`, `vfmacc.vv`, `vs2r.v`; plus the store-offset `addi`s |

The scheduler placed three statements from outside `L108-L119` inside that
address span: the `L120` `vsetvli e16` + `vslidedown.vi` pair at `0x17524` /
`0x17528` and the loop-advance `addi a3,a3,1024` at `0x1752c` (zero samples).
The `L80` activation pointer increment and the `L74` loop back edge landed after
the span at `0x17534` / `0x17538`; they are listed as the supplementary
`loop-tail` region.

## Iteration count

For `M=480, N=1536, K=1536`, one thread, one `run` pass assembles:

```text
a_tiles   = M / 8   = 60    (kMTilesPerChunk = 128, so one chunk of 60 tiles)
b_tiles   = N / 32  = 48
blocks_k  = K / 32  = 48
kernel calls per pass = b_tiles x blocks_k = 2304
iterations per call   = chunk_tiles = 60
iterations per pass   = 2304 x 60 = 138240
```

The benchmark runs `warmup + samples x iterations = 2 + 1 x 20 = 22` passes inside
the profiled process, so:

```text
total inner-loop iterations = 22 x 138240 = 3041280
```

## Results

Mean over three `perf record` repetitions of the same invocation. `Mean share` is
the share of the kernel symbol; `Mean cycles` is the total cycles attributed to
the region over the whole profiled process; `cycles/iter` is that total divided
by `3041280`.

| Region | Mean share of kernel | Mean total cycles | Stddev | **Cycles per iteration** |
|---|---:|---:|---:|---:|
| `L76-96` | 27.95% | 137,608,417 | 3,470,976 | **45.25** |
| `L98-106` | 16.29% | 80,190,699 | 1,331,211 | **26.37** |
| `L108-119` | 49.24% | 242,424,908 | 3,371,425 | **79.71** |
| `loop-tail` (supplementary) | 4.31% | 21,203,706 | 865,808 | 6.97 |
| whole kernel symbol | 58.20% | 492,336,000 | — | 161.87 |

Total process cycles averaged 845,820,506 per repetition. The three requested
regions plus the loop tail and the kernel prologue/while-loop control account for
the whole kernel symbol:

```text
45.25 + 26.37 + 79.71 + 6.97 = 158.30   (regions measured by PC range)
+ 3.6                                    (kernel prologue, while-loop control)
= 161.9 cycles/iteration
```

## Interpretation

- `L108-119` is the dominant cost. It is dominated by the four `vfmacc.vv`
  instructions that read-modify-write the packed C tile: together they hold
  32.7% of the kernel's samples on average (run 1: 1.50%, 9.54%, 10.83%, 11.12%
  individually). The C load/`vfcvt`/store around them add the rest. At 1024-bit
  VLEN each group moves 2 KiB of C in and out of L1D/TCM, so this region is
  C-traffic bound, not matrix bound.
- `L76-96` is the second cost. The single largest PC is the scalar
  `addi t2,t0,128` immediately after the first `vl1r.v` (9.10% of the kernel in
  run 1). This is the documented skid target of a dependent activation load
  miss, so most of the region's 45.25 cycles is activation-load latency hiding,
  not `smt.vmadot` issue.
- `L98-106` is the cheapest of the three. The scale operand `v30` is
  loop-invariant, so the four `smt.vfwmadot` plus the accumulator clear cost
  26.37 cycles/iteration.
- Subtracting the `L120` pair the scheduler moved into the `L108-119` span
  (`vsetvli` + `vslidedown`, 1.07% of the kernel = 1.73 cycles/iteration) gives a
  pure `L108-119` cost of **77.99 cycles/iteration**.

## Cross-checks

- The `L76-96` figure of 45.25 cycles/iteration is within 1.2% of the earlier
  45.8 cycles/tile PC-sampling estimate recorded in the header comment of
  `machine-benchmarks/spacemit-k3-a100/k3_rr_l76_overhead.cpp` for the same
  region, from an independent session on the same board.
- The benchmark's own hardware `cycles` counter reports 25.67-26.04 M cycles per
  `run` pass. Dividing by `138240` gives 185.7-188.4 cycles per pass-iteration for
  `run()` (kernel + `unpack_c_chunk` + `fill_n`). The 161.87 kernel
  cycles/iteration above leaves ~24 cycles/iteration for the adapter, consistent
  with `unpack_c_chunk` sampling at 5.30% of the process.
- `perf annotate` reports `percent: local period`, so the region shares are
  relative to the kernel symbol only; the `cycles(region)` formula above applies
  the symbol's process share explicitly. Run-01/02/03 symbol shares were
  58.02% / 58.42% / 58.17%.

## Caveats

- `cpu-clock` is a software time-sampling event, not the hardware cycle counter.
  It is used only for the *distribution*; the absolute cycle scale comes from
  `perf stat -e cycles:u`. The A100 governor is `userspace` at a fixed 1.8 GHz,
  so the two are proportional during the measurement.
- PMU overflow has skid. As calibrated in
  [2026-09-04-perf-cpu-clock-pc-semantics-calibration.md](2026-09-04-perf-cpu-clock-pc-semantics-calibration.md),
  a stalled load is charged to the *next* control PC, so the per-region totals
  attribute stall time to the instruction the sample lands on, not necessarily to
  the source statement that caused it. Region boundaries are contiguous address
  spans, so this only redistributes samples within a region or into `loop-tail`.
- Address ranges are tied to the exact binary hash above. Any code-generation
  change to `kernel.cpp` invalidates them and requires refreshing
  `tools/pfng_perf/perf-regions.json`.
- The external `perf stat -e cycles:u` counter coexists with the benchmark's own
  internal `cycles` event without multiplexing on this platform: the benchmark
  reports `min_running_pct=100.000000` in all three repetitions.

## Reproduction

```bash
cd perf-framework-ng
./compile-and-test.sh build --platform spacemit-k3-a100

python3 tools/pfng_perf/pfng_perf.py profile \
  --remote spacemit-k3 --platform spacemit-k3-a100 --cpu 8 \
  --kernel q4_0-q8_0-IME-RR \
  --m 480 --n 1536 --k 1536 \
  --warmup 2 --samples 1 --iterations 20 --repetitions 3 \
  --event cpu-clock:u --period 10000 --cycles-event cycles:u
```

The artifact, including every `perf.data`, the exact binary, the external
counter dump and the annotated disassembly, is preserved under
`tools/pfng_perf/.artifacts/perf-cpu-clock/20260929-125212-q4_0-q8_0-IME-RR-m480-n1536-k1536/`.

`tools/pfng_perf/perf-regions.json` is a local, gitignored file
(`tools/pfng_perf/.gitignore`), because address ranges are valid only for one
exact binary. Recreate its `q4_0-q8_0-IME-RR` entry from the address map above:

```json
"q4_0-q8_0-IME-RR": {
  "dso": "pfng-bench",
  "symbol": "q4_0_q8_0_ime_rr_w_stationary(signed char const*, unsigned short const*, unsigned char const*, unsigned short const*, float*, unsigned long)",
  "regions": [
    {"name": "L76-96",    "start": "0x17464", "end": "0x174a8"},
    {"name": "L98-106",   "start": "0x174ac", "end": "0x174d4"},
    {"name": "L108-119",  "start": "0x174d8", "end": "0x17530"},
    {"name": "loop-tail", "start": "0x17534", "end": "0x17538"}
  ]
}
```
