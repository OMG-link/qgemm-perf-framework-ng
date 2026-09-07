# SpacemiT K1 concurrent hardware-prefetch streams

## Question

Can the hardware prefetcher correctly track multiple concurrent forward address streams?

## Method

`prefetch-multistream` creates `S` independent forward streams. Stream bases are 4,160 bytes apart: one 4 KiB page plus one 64-byte cache line. The extra line rotates streams across L1 cache sets and avoids the same-set conflict present in an initial 4 KiB-spacing smoke test.

Within each trial, accesses are line-major and round-robin across streams:

1. load line 0 of every stream;
2. load line 1 of every stream;
3. load line 2 of every stream;
4. probe line 3 of every stream.

Every load is followed by exactly 16 or 64 uncompressed scalar NOPs. A tracked stream should therefore produce three L1D misses and a prefetched line-3 hit. An untracked stream should produce four misses.

Each sample contains 1,024 trials, with trial regions visited through a randomized and prewarmed pointer array. A 96 MiB capacity-eviction scan runs before each sample. The process is pinned to CPU 0. Hardware cycles and generic L1D read misses are measured as a pinned perf group; every sample had 100% `time_running/time_enabled`. Five samples are collected per stream count.

The final ELF was disassembled. It retains the three nested training passes, one probe pass, scalar `ld` operations, a 4,160-byte stream stride, and exact 16/64-instruction NOP corridors.

## Results

Median L1D misses per stream, with four explicit loads per stream:

| concurrent streams | 16 NOP | 64 NOP | ideal if all tracked | no prefetch |
|---:|---:|---:|---:|---:|
| 1 | 3.107 | 3.098 | 3.000 | 4.000 |
| 2 | 3.578 | 3.571 | 3.000 | 4.000 |
| 3 | 3.713 | 3.709 | 3.000 | 4.000 |
| 4 | 3.786 | 3.785 | 3.000 | 4.000 |
| 5 | 3.831 | 3.829 | 3.000 | 4.000 |
| 6 | 3.865 | 3.863 | 3.000 | 4.000 |
| 7 | 3.891 | 3.890 | 3.000 | 4.000 |
| 8 | 3.960 | 3.959 | 3.000 | 4.000 |
| 9 | 3.965 | 3.964 | 3.000 | 4.000 |
| 10 | 3.969 | 3.968 | 3.000 | 4.000 |
| 11 | 3.972 | 3.972 | 3.000 | 4.000 |
| 12 | 3.977 | 3.978 | 3.000 | 4.000 |

A useful way to interpret the result is the total number of successful prefetched probes per trial:

`prefetched probes = streams * (4 - misses_per_stream)`

| streams | prefetched probes, 16 NOP | prefetched probes, 64 NOP |
|---:|---:|---:|
| 1 | 0.893 | 0.902 |
| 2 | 0.844 | 0.857 |
| 3 | 0.862 | 0.872 |
| 4 | 0.856 | 0.859 |
| 5 | 0.845 | 0.853 |
| 6 | 0.808 | 0.824 |
| 7 | 0.761 | 0.767 |
| 8 | 0.318 | 0.329 |
| 12 | 0.272 | 0.269 |

If all streams were tracked, the second table would be approximately equal to the stream count. Instead it remains below one.

## Conclusion

For this round-robin access pattern, the hardware prefetcher does **not** correctly maintain independent useful prefetch state for multiple concurrent streams:

- one stream works as expected: its fourth line is prefetched;
- with 2--7 concurrent streams, only about one probe in total is prefetched per trial, equivalent to roughly one useful active stream rather than all `S` streams;
- at 8 or more streams, even that single useful prefetch drops to roughly 0.3 probes per trial;
- 16 and 64 NOP corridors produce nearly identical results, so this is not explained by software issuing the probe too soon.

The measurements support an **effective useful concurrency of one stream** for this exact interleaved pattern. They do not by themselves prove that the hardware contains exactly one stream-tracker entry: replacement policy, page-based tracking, prefetch issue throttling, or another internal policy could produce the same externally visible behavior.

This test uses streams separated by 4,160 bytes, forward `+64 B` progression, three training accesses, and a synchronized probe phase. Other scheduling patterns—such as completing one stream's four accesses before touching the next, using longer training sequences, or spacing streams within one page—may exercise different policies and should be treated as separate experiments.
