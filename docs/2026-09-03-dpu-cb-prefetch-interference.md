# DPU-CB hot-panel interference with A-stream prefetch

## Question

Does hardware prefetch remain effective for the DPU-CB memory pattern: repeatedly scan a small L1D-resident array while sequentially streaming another much larger array?

## Reproduced access trace

The benchmark follows the compute-body memory topology of `q4_0-q8_0-IME-m8b-DynPreUnpack-CacheBlocking`:

- B is an 8 KiB unpacked panel. Each K16 inner loads 256 B with one `vl8r.v`; after 32 inners the address wraps to the panel beginning.
- A consists of 288-byte Q8 blocks. Each block has two inners. Inner 0 loads four 32-byte vectors from offsets `+32,+64,+96,+128`; inner 1 loads offsets `+160,+192,+224,+256`. The next block begins 288 B later.
- Load order is B first, then the four A loads.
- Sixteen uncompressed scalar NOPs follow each inner's loads, standing in for the 16 `smt.vmadot` instructions and preventing the load stream from outrunning prefetch generation.

The test uses 65,536 A blocks: 18 MiB of sequential A data and 131,072 K16 inners. Before every sample a 96 MiB buffer is scanned for capacity eviction, then the complete 8 KiB B panel is touched while PMU counting is disabled. Seven samples are collected on CPU 0 with a pinned cycles/L1D-read-miss perf group; every event had 100% `time_running/time_enabled`.

Four paired assembly cases use identical loop/address-control structure:

- `a_only`: real sequential A trace, B load replaced by a NOP;
- `b_only`: real cyclic 8 KiB B trace, four A loads replaced by NOPs;
- `mixed`: real sequential A plus real cyclic 8 KiB B;
- `fixed_b`: real sequential A plus the same number and placement of B loads, but B repeatedly reads one fixed hot 512-byte area.

Final-ELF disassembly confirms two inners per block, one 256-byte B load and four 32-byte A loads per mixed inner, 16 NOPs per inner, B wrap by `(block & 15) * 512`, and 288-byte A block advancement.

## Results

Medians over seven samples:

| case | cycles/inner | L1D misses/inner |
|---|---:|---:|
| A only | 34.32 | **0.004837** |
| B only, cyclic 8 KiB | 23.90 | **0.000206** |
| A + cyclic 8 KiB B | 74.49 | **0.550011** |
| A + fixed hot 512 B | 53.91 | **0.036346** |

The real mixed case has approximately 114 times as many misses as A-only. Subtracting the independently measured B-only residual does not materially change it: `0.550011 - 0.000206 = 0.549805 miss/inner`. This subtraction is diagnostic rather than strict attribution because cache replacement and prefetch state are nonlinear.

The fixed-B control is important. Merely inserting B loads at the same instruction positions raises misses from 0.0048 to 0.0363, but scanning and wrapping the real 8 KiB panel raises them to 0.5500. Therefore most degradation is associated with the active cyclic B address stream/cache footprint, not simply extra load instructions or insufficient NOP delay.

## How much prefetch remains?

A occupies 288 B per block, or 4.5 unique 64-byte cache lines per two inners: **2.25 cold A lines per inner**. With no useful prefetch, the sequential A side alone would therefore approach 2.25 demand misses/inner. The measured mixed total is only 0.55 misses/inner, and that total also includes any B misses.

Consequently hardware prefetch is still substantially active in the mixed case: at least roughly `1 - 0.55/2.25 = 75.6%` of the potential cold-line demand misses are avoided. This is a conservative lower bound on A coverage because all measured mixed misses were charged to A for the calculation.

However, it is not operating at the A-only level. A-only reduces 2.25 potential cold lines to 0.0048 misses/inner, while adding the real cyclic B panel leaves about 0.55 misses/inner. Thus the answer is:

- **prefetch still works partially and avoids most cold A demand misses;**
- **it does not remain normally/fully effective under the DPU-CB interleaving;**
- the cyclic 8 KiB B stream causes a large, repeatable degradation compared with both A-only and fixed-hot-B controls.

This agrees with the separate multi-stream experiment, where the same machine showed only about one useful concurrent prefetch stream for round-robin streams. It does not prove whether the mixed misses are demand misses from A, B lines evicted by A, or both; generic PMU counting cannot attribute misses to load PCs. The result characterizes the combined externally visible behavior relevant to DPU-CB.

## Build and run

```bash
./compile-and-test.sh configure
cmake --build build --target dpu-cb-prefetch
scp build/machine-benchmarks/dpu-cb-prefetch spacemit-k1:/tmp/
ssh spacemit-k1 'taskset -c 0 /tmp/dpu-cb-prefetch --blocks 65536 --samples 7'
```
