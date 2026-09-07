# DPU-CB B-outer full-A sweep

## Questions

This experiment answers two related questions for the requested B-outer loop:

1. How many direct L1D read-miss events are caused by advancing to another 8 KiB B panel?
2. With the complete B allocation and total work held constant, how does the B-panel address sequence affect mixed B/A misses?

The four compared sequences are:

```text
fixed:      B0, B0, B0, ...
alternate:  B0, B1, B0, B1, ...
sequential: B0, B1, B2, ...
shuffled:   B37, B4, B81, ...
```

## Exact modeled loop order

```text
for pass:
    for visit in 0 .. b_panels:
        B_panel = B_base + order[visit]
        A = A_begin
        for A_tile in A[0 .. a_tiles) step 4,608 B:
            for block in 0 .. 16:
                inner 0: load B_panel[block*512 + 0],   then four A vectors
                inner 1: load B_panel[block*512 + 256], then four A vectors
```

Important properties:

- one B panel is 8 KiB;
- A is reset to its beginning for every B visit;
- one A tile contains `16 × 288 B = 4,608 B`;
- the representative complete A range uses 60 tiles = 276,480 B;
- one B visit with all 60 A tiles executes `60 × 16 × 2 = 1,920` inners;
- each pass performs 96 visits and `96 × 1,920 = 184,320` inners;
- every case allocates, initializes, and locks the same complete `96 × 8 KiB = 768 KiB` B array, including panels not referenced by fixed or alternate;
- every order table has 96 entries, so all cases perform identical loop and instruction counts.

This differs from the discarded preliminary experiment, which switched B after only one 28-tile A chunk and used a configurable non-8-KiB panel stride. That implementation and its document were removed.

## Unified order-table implementation

All four sequence cases call the same assembly function, `dpu_cb_b_outer_mixed`. The only case-specific input is an equal-length table of byte offsets:

- `b_pattern_fixed`: all entries are 0;
- `b_pattern_alternate`: `(visit % 2) × panel_bytes`;
- `b_pattern_sequential`: `visit × panel_bytes`;
- `b_pattern_shuffled`: deterministic xorshift/Fisher-Yates permutation, with panels 37, 4, and 81 moved to the first three positions when available.

For 96 panels, the shuffled table starts:

```text
37, 4, 81, 48, 88, 33, 27, 72, ...
```

Final-ELF disassembly confirms that the kernel:

1. loads `order[visit]` only at the panel boundary;
2. computes `B_panel = B_base + byte_offset`;
3. resets A for every panel visit;
4. preserves the inner B load, four A loads, 16 NOPs, a runtime block count, and two inners across all cases.

`--b-panel-kib` accepts 4 or 8 for B-pattern cases. Each B block remains 512 B, so the corresponding block counts are 8 and 16. The legacy non-B-pattern cases remain fixed at 8 KiB/16 blocks.

The normalization is:

```text
inners              = passes × b_panels × a_tiles × blocks_per_panel × 2
panel visits         = passes × b_panels
misses/inner         = L1D miss events / inners
misses/panel visit   = L1D miss events / panel visits
misses/pass          = L1D miss events / passes
```

## Four-sequence paired result

Command and controls:

```bash
ssh spacemit-k1 '/tmp/dpu-cb-prefetch --cpu 5 --case b_patterns --b-panels 96 --a-tiles 60 --passes 16 --samples 7 --warmup-passes 2 --lock-memory'
```

Each sample executes 2,949,120 inners. All four cases run in one process and share the same A and B allocations. Every retained sample had zero context switches, zero CPU migrations, and 100% perf running time. Values below are medians of seven samples.

| B sequence | L1D misses/inner | L1D misses/panel visit | L1D misses/pass | Change vs fixed |
|---|---:|---:|---:|---:|
| fixed `B0,B0,...` | 0.455034 | 873.666016 | 83,871.937500 | baseline |
| alternate `B0,B1,...` | 0.104522 | 200.682292 | 19,265.500000 | -77.0% |
| sequential `B0,B1,B2,...` | 0.121392 | 233.073568 | 22,375.062500 | -73.3% |
| shuffled `B37,B4,B81,...` | 0.095788 | 183.913411 | 17,655.687500 | -79.0% |

The per-inner sample ranges were:

| B sequence | Minimum | Maximum | Median |
|---|---:|---:|---:|
| fixed | 0.403956 | 0.466881 | 0.455034 |
| alternate | 0.104257 | 0.105934 | 0.104522 |
| sequential | 0.119599 | 0.122350 | 0.121392 |
| shuffled | 0.094687 | 0.097289 | 0.095788 |

Fixed B remains both much higher and more variable. Alternate, sequential, and shuffled are tightly clustered within each case. Their differences are small compared with the gap from fixed B.

## 4 KiB panels with equal total B size and work

The follow-up halves each panel to 4 KiB and doubles the panel count to 192. This preserves:

- complete B payload: `192 × 4 KiB = 768 KiB`;
- inners per pass: `192 × 60 × 8 × 2 = 184,320`;
- inners per measured sample: 2,949,120;
- warmup work: one 4 KiB pass equals two 8 KiB passes, both 184,320 inners and 192 panel visits.

The exact command was:

```bash
ssh spacemit-k1 '/tmp/dpu-cb-prefetch --cpu 5 --case b_patterns --b-panel-kib 4 --b-panels 192 --a-tiles 60 --passes 16 --samples 7 --warmup-passes 1 --lock-memory'
```

Every retained sample again had zero context switches, zero CPU migrations, and 100% perf running time. The deterministic shuffled prefix for 192 panels was `37,4,81,112,35,145,104,57`. Medians of seven samples:

| B sequence | L1D misses/inner | L1D misses/panel visit | L1D misses/pass | Change vs 4 KiB fixed | Change vs same 8 KiB sequence |
|---|---:|---:|---:|---:|---:|
| fixed `B0,B0,...` | 0.297582 | 285.678385 | 54,850.250000 | baseline | -34.6% |
| alternate `B0,B1,...` | 0.068833 | 66.079753 | 12,687.312500 | -76.9% | -34.1% |
| sequential `B0,B1,B2,...` | 0.089037 | 85.475911 | 16,411.375000 | -70.1% | -26.7% |
| shuffled `B37,B4,B81,...` | 0.093970 | 90.211263 | 17,320.562500 | -68.4% | -1.9% |

Per-inner sample ranges:

| B sequence | Minimum | Maximum | Median |
|---|---:|---:|---:|
| fixed | 0.293720 | 0.301665 | 0.297582 |
| alternate | 0.068456 | 0.069381 | 0.068833 |
| sequential | 0.088607 | 0.090216 | 0.089037 |
| shuffled | 0.092998 | 0.094620 | 0.093970 |

The core result survives: repeatedly using one fixed B address is still pathological, while changing between two or more addresses lowers misses by 68%--77%. With 4 KiB panels, alternate is lowest, followed by sequential and shuffled; unlike the 8 KiB run, shuffled is not best. This reinforces that the exact ranking among non-fixed patterns is phase/state dependent, while their large separation from fixed is the robust effect.

### Comparison boundary

Equal total inner work does not mean an identical per-visit A footprint. An 8 KiB visit executes 16 blocks and advances A by `16 × 288 = 4,608 B` per configured A tile. A 4 KiB visit executes 8 blocks and advances A by 2,304 B per configured tile. Therefore each 4 KiB visit covers half as many A bytes, and the doubled visit count restores the same aggregate inner/load count per pass.

Consequently, the 4-vs-8 KiB percentage differences combine panel-size/address-period effects with the changed A restart frequency and per-visit A footprint. They should not be attributed solely to B capacity. The within-4-KiB comparison is fully controlled: all four sequences use the same complete B allocation, visit count, A behavior, instruction count, and unified assembly kernel.

## Interpretation

A sequential B stream is not required to obtain the low-miss state: the deterministic shuffled order has the lowest median, and alternating only two panels is also low. Therefore the large reduction cannot primarily be explained by ordinary sequential prefetching across the 768 KiB B array.

The decisive distinction is whether the same B address identity is repeated indefinitely. Changing between even two B panels breaks the pathological high-miss state produced by cycling one fixed 8 KiB B panel alongside the A stream. This is consistent with an interaction in the hardware cache/prefetch state machines between the repeating B address pattern and the sequential A stream. The PMU data establishes the behavior but does not by itself identify a specific undocumented state machine.

Randomizing B does **not** increase mixed misses in this experiment. Shuffled is 0.095788 miss/inner, 79.0% below fixed and also below sequential. Thus B footprint or lack of spatial continuity is not the dominant mixed-miss cost here.

## Earlier panel-count and direct-B controls

An earlier equal-work scan used contiguous panel order and exactly 2,949,120 inners per row:

| B panels | passes | B-only misses/inner | mixed misses/inner |
|---:|---:|---:|---:|
| 1 | 1536 | 0.000044 | 0.524534 |
| 2 | 768 | 0.000175 | 0.124399 |
| 4 | 384 | 0.000499 | 0.102962 |
| 8 | 192 | 0.000999 | 0.118269 |
| 16 | 96 | 0.000173 | 0.113916 |
| 32 | 48 | 0.000659 | 0.115538 |
| 64 | 24 | 0.000158 | 0.116340 |
| 96 | 16 | 0.000142 | 0.119453 |

The direct B-only difference was:

```text
96-panel B-only - 1-panel B-only
= 0.000142 - 0.000044
= 0.000098 additional L1D miss events/inner
```

That is approximately 0.188 additional PMU miss events per 1,920-inner B visit, or 18.1 events per complete 96-panel sweep. These are missed vector-load events, not cache-line counts. The direct cost is tiny compared with the mixed reduction caused by leaving the fixed-B state.

## Conclusion

For 8 KiB panels, changing B address identity reduces mixed L1D misses by 73%--79%. For 4 KiB panels with the same 768 KiB allocation and aggregate inner work, it reduces misses by 68%--77%. The fixed sequence remains the clear outlier in both experiments.

Halving the panel lowers fixed from 0.455034 to 0.297582 miss/inner, but also changes the non-fixed cases unevenly: alternate falls to 0.068833, sequential to 0.089037, while shuffled changes only slightly to 0.093970. Thus contiguous progression is not required, and the ordering among non-fixed patterns is not robust across panel sizes. Repeated fixed-B access is a pathological A/B interaction rather than evidence that switching or randomizing B is intrinsically expensive.

## Reproduction

```bash
cmake --build build --target dpu-cb-prefetch
scp build/machine-benchmarks/dpu-cb-prefetch spacemit-k1:/tmp/

# 96 × 8 KiB
ssh spacemit-k1 '/tmp/dpu-cb-prefetch --cpu 5 --case b_patterns --b-panel-kib 8 --b-panels 96 --a-tiles 60 --passes 16 --samples 7 --warmup-passes 2 --lock-memory'

# 192 × 4 KiB: same 768 KiB B payload and same inners/pass
ssh spacemit-k1 '/tmp/dpu-cb-prefetch --cpu 5 --case b_patterns --b-panel-kib 4 --b-panels 192 --a-tiles 60 --passes 16 --samples 7 --warmup-passes 1 --lock-memory'
```
