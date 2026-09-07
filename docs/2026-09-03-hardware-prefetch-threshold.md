# SpacemiT K1 sequential hardware-prefetch threshold

## Question

How many consecutive cache-line accesses are needed before the current machine starts hardware prefetching? The test intentionally inserts fixed scalar NOP corridors after every load, rather than issuing a pure back-to-back load stream, so the software does not outrun prefetch request generation/completion.

## Method

The independent `prefetch-threshold` machine benchmark uses 64-byte cache lines and 4 KiB-aligned regions. For each trial it:

1. reads `train` consecutive lines at offsets `0, 64, ...`;
2. executes a fixed corridor of 0, 16, 64, or 256 uncompressed scalar `nop` instructions after every load;
3. probes the immediately following line and executes the same NOP corridor;
4. consumes every loaded value in an XOR sink.

Each measured sample contains 2,048 independent regions. Before each sample, a 96 MiB buffer is scanned for capacity eviction. Regions are visited through a randomized page-pointer array, which is prewarmed before PMU enable. This is important: the first implementation advanced linearly by 4 KiB between trials, and the hardware learned that second stream, producing a false near-zero-miss `train=4` result. Random page order removes that cross-trial stream.

The process and remote command are both pinned to CPU 0. A pinned perf group counts hardware cycles and generic `L1D read miss`; every reported event had `time_running == time_enabled` (100%). Five samples are collected for each case. The final ELF was disassembled: `.option norvc` is effective, the loop and scalar `ld` operations remain, and the 16/64/256 cases contain exactly that many 4-byte NOPs after each training load and probe.

Build and run:

```bash
./compile-and-test.sh configure
cmake --build build --target prefetch-threshold
scp build/machine-benchmarks/prefetch-threshold spacemit-k1:/tmp/
ssh spacemit-k1 'taskset -c 0 /tmp/prefetch-threshold \
  --trials 2048 --samples 5 --max-train 8 --gap 64'
```

## Results

Median L1D read misses per trial:

| training lines | total explicit loads | 0 NOP | 16 NOP | 64 NOP | 256 NOP |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1.013 | 1.022 | 1.014 | 1.017 |
| 1 | 2 | 2.034 | 2.044 | 2.044 | 2.052 |
| 2 | 3 | 3.090 | 3.107 | 3.090 | 3.101 |
| 3 | 4 | 3.078 | 3.089 | 3.100 | 3.094 |
| 4 | 5 | 4.081* | 3.100 | 3.098 | 3.100 |
| 5 | 6 | 3.113 | 3.101 | 3.107 | 3.105 |
| 6 | 7 | 3.120 | 3.110 | 3.106 | 3.110 |
| 7 | 8 | 3.121 | 3.110 | 3.112 | 3.117 |
| 8 | 9 | 3.128 | 3.110 | 3.116 | 3.117 |

`*` The zero-gap `train=4` case was unstable across its five raw samples (3.086 to 4.313 misses/trial). It is not used for the threshold conclusion because the experiment specifically aims to avoid software outrunning prefetch, and all NOP-spaced cases are stable.

## Conclusion

For a forward unit-stride stream on this machine, the first three distinct cache lines miss in L1D. After the third consecutive access, the fourth line is already available through hardware prefetch in the NOP-spaced tests. Additional sequential loads do not increase misses: with 16, 64, or 256 NOPs, cases containing 4 through 9 explicit loads remain at approximately 3.09--3.12 L1D misses per trial.

Therefore the observed trigger behavior is:

- one or two prior consecutive addresses are insufficient;
- **three consecutive 64-byte cache-line accesses train/trigger the prefetcher; the fourth and subsequent lines are prefetched**;
- 16 scalar NOPs after each load are already sufficient to expose this behavior; increasing to 64 or 256 NOPs does not change the threshold.

This establishes the threshold for forward `+64 B` streams within one 4 KiB page. It does not yet characterize backward streams, non-unit strides, page-boundary crossing, prefetch distance, or the number of simultaneous streams. Generic L1D miss semantics are used; no claim is made about a dedicated native prefetch-request event.
