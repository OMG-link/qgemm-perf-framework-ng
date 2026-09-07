# DPU-CB steady-state L1-B / L2-A prefetch interference

## Question

After excluding all B-panel warmup misses, does hardware prefetch remain effective for the DPU-CB memory pattern: repeatedly scan an 8 KiB B panel in L1D while repeatedly streaming a 129,024-byte A chunk from L2?

## Exact memory trace

The independent `dpu-cb-prefetch` assembly reproduces the memory instruction order and cache-block loop shape of `q4_0-q8_0-IME-m8b-DynPreUnpack-CacheBlocking`, without reproducing arithmetic:

- one A chunk contains 28 M tiles × 16 QK32 blocks × 288 B = 129,024 B;
- one B panel contains 16 blocks × 512 B = 8 KiB;
- every block executes two K16 inners;
- inner 0 loads B `[block*512 + 0, +256)` with one `vl8r.v`, then A at `+32,+64,+96,+128` with four 32-byte `vle8.v` operations;
- inner 1 loads B `[block*512 + 256, +512)`, then A at `+160,+192,+224,+256`;
- 16 uncompressed scalar NOPs follow each inner's loads, giving prefetch requests time to issue without adding memory traffic;
- after 16 blocks, the next A tile continues sequentially while B restarts from the same panel beginning;
- after 28 tiles, A restarts from the chunk beginning for the next pass.

Each pass therefore executes exactly `28 * 16 * 2 = 896` inners. The final test runs 4,096 passes, or 3,670,016 inners per sample.

Final-ELF disassembly confirms the two vector-load groups, 16-block loop, 28-tile loop, parameterized pass loop, 288-byte A advancement, B restart for every tile, and 16 NOPs after every inner. No `vmadot` or other arithmetic-memory traffic is present.

## Cache preparation and PMU boundary

Every sample prepares cache state while PMU counting is disabled:

1. touch the complete 129,024-byte A chunk, placing it in the 512 KiB L2;
2. touch a separate 64 KiB scrub buffer, evicting A's L1D copies while the combined footprint remains well below L2 capacity;
3. touch the complete 8 KiB B panel last, placing B in L1D;
4. execute `fence rw,rw`;
5. only then reset and enable the pinned cycles/L1D-read-miss perf group.

Thus initial A/B population is excluded from every count. CPU 0 is used throughout, and all events reported 100% `time_running/time_enabled`.

Four paired cases retain the same loop structure:

- `a_only`: L2 A stream, B load replaced by a NOP;
- `b_only`: cyclic L1D B panel, A loads replaced by NOPs;
- `mixed`: real L2 A stream plus real cyclic 8 KiB B panel;
- `fixed_b`: real L2 A stream plus the same B load positions, repeatedly accessing only one hot 512-byte B area.

## Results

Medians over seven samples:

| case | cycles/inner | L1D misses/inner |
|---|---:|---:|
| A only | 30.44 | **0.026339** |
| B only, prewarmed cyclic 8 KiB | 22.60 | **0.000043** |
| A + cyclic 8 KiB B | 73.46 | **0.489879** |
| A + fixed hot 512 B | 54.54 | **0.054488** |
| packed A only, no 32 B gaps | 32.10 | **0.081414** |
| packed A + cyclic 8 KiB B | 65.75 | **0.369240** |

The B-only value is effectively zero, confirming that B warmup/fill misses are outside the measured window. Subtracting it from the original mixed result changes it only from 0.489879 to 0.489836 miss/inner.

The real mixed pattern causes about 18.6 times as many misses as A-only. The fixed-B control reaches only 0.054488 miss/inner, so extra B load instructions alone do not explain the effect. Repeatedly scanning the full 8 KiB B address range is the dominant source of interference.

## Stability of the 32-byte-gap effect

The original Q8 block contains 256 bytes of `qs` payload plus a 32-byte scale prefix, giving a 288-byte block stride. Packed controls remove that prefix: eight 32-byte loads cover one contiguous 256-byte block and the next block starts immediately at `+256`. B accesses, loops, NOP spacing, PMU boundaries, and inner normalization are unchanged.

A first seven-sample run suggested that removing the gaps reduced mixed misses from 0.489879 to 0.369240 miss/inner. Broader testing shows that this exact reduction is **not stable** and must not be treated as a fixed attribution.

The benchmark was parameterized with A/B base offsets in 64-byte units, a configurable L1 scrub size, case selection, and different pass counts. Results include:

| parameter change | packed A-only | packed mixed |
|---|---:|---:|
| 512 passes, 64 KiB scrub | 0.0263 | 0.2616 |
| 2,048 passes, 64 KiB scrub | 0.026--0.092 | 0.4382 |
| 8,192 passes, 64 KiB scrub | 0.0264 | 0.2974 |
| A offset 64/128/256 B | about 0.0264 | 0.427--0.452 |
| scrub 32 KiB | 0.0264 | 0.3133 |
| scrub 128 KiB | 0.0919 | 0.4337 |
| B offset 64/128/256/512 B | not measured | 0.273--0.456 |

Ten additional independent processes used identical parameters (`2,048` passes, 64 KiB scrub, zero offsets, one sample per process):

- original A-only was stable near 0.0263 miss/inner;
- packed A-only was near 0.0263 in eight runs but jumped to approximately 0.092 in two runs;
- original mixed ranged from **0.391 to 0.553**, with cross-process median approximately **0.485**;
- packed mixed ranged from **0.250 to 0.508**, with cross-process median approximately **0.378**;
- packed mixed was lower in seven of ten paired runs, but higher in three.

Therefore removing the 32-byte holes shows a tendency to reduce the mixed miss rate, but neither the magnitude nor even the per-run ordering is invariant. The earlier 24.6% total reduction and 37.9% incremental reduction describe one run only and are withdrawn as general conclusions.

The stable conclusion is instead that the prefetch behavior is strongly state- and phase-sensitive. Pass count, cache preparation footprint, B cache-set phase, and apparently prior prefetcher state can move the result between distinct regimes. The 32-byte periodic holes may contribute, but the current experiment cannot assign them a stable fraction of the approximately 0.4--0.55 original mixed miss rate.

## Isolation of run-to-run variation

The benchmark now exposes controls for the remaining software-visible sources of variation:

- `--warmup-passes N` executes the selected target trace before enabling PMU counting;
- `--lock-memory` locks A, B, scrub, and output pages with `mlock`;
- `--continuous` performs cache preparation and target-trace warmup only before sample 0, then measures adjacent windows without resetting cache or prefetch state;
- `--cpu N` pins the process internally and reports both the requested and actual CPU;
- output records A/B/scrub virtual addresses, context switches, CPU migrations, and perf running percentage.

The internal CPU option is important: the original program pinned itself to CPU 0, so an external `taskset -c N` could be overwritten by the process. Any earlier cross-CPU scan made with only external `taskset` is invalid and is not used below.

### Stepwise exclusions

1. **Target-trace warmup:** scanning 0, 1, 4, 16, 64, and 256 warmup passes changed the selected regime but did not force one stable value. Even 256 passes did not eliminate variation.
2. **Page residency and scheduling:** `mlock` succeeded. Every retained sample had `context_switches=0`, `migrations=0`, and 100% perf running time. The CPU-frequency governor was `performance`.
3. **Same allocation and uninterrupted trace:** with `--continuous`, the same process used the same virtual and physical pages, prepared cache only once, and then executed adjacent measurement windows. Original mixed still moved between approximately **0.27 and 0.58 miss/inner**. Therefore ASLR, physical-page reallocation, page faults, and repeated cache preparation are not necessary for the variation.
4. **Explicit CPU pinning:** after adding `--cpu`, all eight CPUs could exhibit different regimes over time. A fixed CPU 5 test across ten independent processes still covered approximately **0.32--0.55 miss/inner**. CPU identity is not sufficient to select a regime.
5. **Control traces:** on CPU 5 with 256 warmup passes, locked pages, and 12 continuous windows, the non-interfering controls were highly stable:

| case | observed L1D misses/inner |
|---|---:|
| A only | 0.02618--0.02641 |
| B only | 0.00009--0.00017 |
| A + fixed 512-byte B area | 0.05334--0.05537 |
| packed A only | 0.02634--0.02644 |
| A + cyclic 8 KiB B | 0.49334--0.54854 in that run; about 0.27--0.58 overall |

The stable PMU controls rule out generic counter instability, ordinary scheduler noise, and either stream in isolation. The fixed-B result also rules out the B load instruction count alone. The large dynamic variation appears only when the sequential A stream is interleaved with a changing, cyclic B address stream.

### Root-cause conclusion

The strongest supported conclusion is that the variation is intrinsic to the **dynamic interaction between the sequential A stream and cyclic B stream in the hardware prefetch/cache machinery**. That interaction can enter and leave distinct effectiveness regimes while addresses, physical pages, CPU, instruction trace, and cache preparation remain fixed. This is consistent with a history- or phase-sensitive prefetch state machine; the available generic PMU events cannot identify a specific undocumented prefetcher table or transition.

Physical placement and B cache-set phase may bias which regime is reached, but they are not the root necessary condition because transitions occur within one fixed allocation. Transparent huge pages were configured as `madvise`, but the machine reported `AnonHugePages: 0 kB` and no reserved hugetlb pages. A huge-page experiment was therefore not used to claim causality; more importantly, it could not explain transitions observed without any remapping.

## Comparison with the real DPU-CB observation

The real DPU-CB measurement discussed for the large shape was approximately:

```text
13,417,390 L1D misses / 10,321,920 inners = 1.2999 misses/inner
```

Both measurements now use the same denominator, `misses/inner`:

| measurement | L1D misses/inner |
|---|---:|
| steady-state memory-only simulation | **0.391--0.553** across independent runs |
| real DPU-CB | **1.2999** |

Using the cross-process original-mixed range, the isolated L1-B/L2-A pattern reproduces approximately **30.1%--42.5%** of the real miss rate; the cross-process median near 0.485 corresponds to about 37.3%. It confirms a substantial interference mechanism but does **not** reproduce the full DPU-CB count, and a single-run percentage is not robust.

A contributes 288 B per two inners, or 2.25 new 64-byte lines per inner. Conservatively charging every mixed miss to A gives an avoided-demand-miss range of roughly **75.4%--82.6%**. Prefetch remains active but is far less effective than in A-only.

The remaining difference to 1.2999 miss/inner must come from behavior absent from this memory-only steady-state model, potentially including real `vmadot` timing/resource interaction, scale/reduction/output accesses, different cache-set alignment, N/K panel transitions, chunk boundaries, or additional real-kernel state. Generic L1D PMU counting cannot assign misses to individual load PCs.

## Build and run

```bash
./compile-and-test.sh configure
cmake --build build --target dpu-cb-prefetch
scp build/machine-benchmarks/dpu-cb-prefetch spacemit-k1:/tmp/
ssh spacemit-k1 '/tmp/dpu-cb-prefetch --cpu 5 --case mixed --passes 2048 --samples 12 --warmup-passes 256 --lock-memory --continuous'
```
