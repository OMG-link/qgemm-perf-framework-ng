# Kernel performance history

This document is updated as kernel implementations change. The only performance
metric recorded is compute-unit utilization relative to the project peak of 128
FMA/cycle.

Unless noted otherwise, measurements use `M=480`, `N=1536`, `K=1536`, a CMake
Release build with a fixed 256-bit vector length, and are pinned to CPU 0 on
`spacemit-k1`.

## q4_0-q8_0-IME-m8b

| Date | Kernel | Change | Compute-unit utilization |
|---|---|---|---:|
| 2026-08-25 | `q4_0-q8_0-IME-m8b` | Fully unrolled inner loop | 5.38% |
| 2026-08-25 | `q4_0-q8_0-IME-m8b` | Disable inner-loop unrolling | 5.96% |
| 2026-08-26 | `q4_0-q8_0-IME-m8b` | Use four loose M2 reduction accumulators | 6.14% |
| 2026-08-27 | `q4_0-q8_0-IME-m8b` | Convert eight M2 accumulators before rebuilding two M8 results | 6.24% |
| 2026-08-27 | `q4_0-q8_0-IME-m8b` | Use SoA weight planes for M8 batch reduction | 6.45% |
| 2026-08-27 | `q4_0-q8_0-IME-m8b` | 调整指令排布（通过添加 barrier 以及延后 fcvt） | 6.71% |
| 2026-08-28 | `q4_0-q8_0-IME-m8b` | Expand component writeback and batch reduction to 8x16; reduce four rows per spill-free group | 6.99% |
| 2026-09-16 | `q4_0-q8_0-IME-m8b` | Move the N tiling and the acc -> C copy into the adapter; the microkernel accumulates into the caller's `acc[kMr][kNr]` tile | 7.08% |

## q4_0-q8_0-IME-m8b-CB

| Date | Kernel | Change | Compute-unit utilization |
|---|---|---|---:|
| 2026-09-16 | `q4_0-q8_0-IME-m8b-CB` | K-panel blocked baseline with a staging C tile and a per-panel C update pass | 6.66% |
| 2026-09-16 | `q4_0-q8_0-IME-m8b-CB` | Let the microkernel accumulate straight into the packed C tile: no staging tile, no second C pass | 7.03% |

The 2026-09-16 microkernel ABI refactor costs the plain variant nothing (N tiling
only moved into the adapter) and lifts the blocked variant to the plain
variant's level. Both binaries measured in one session with the same runner and
pinning (`setarch -R`), one run per cell: `M480 N8960 K1536` gives 1 thread
828.3 M -> 786.0 M cycles (-5.1%) and 4 threads 262.3 M -> 254.5 M (-3.0%),
`M480 N1536 K8960` gives 1 thread 750.4 M -> 714.9 M (-4.7%) and 4 threads
217.5 M -> 204.1 M (-6.2%).

## q4_0-q8_0-IME-m4i

| Date | Kernel | Change | Compute-unit utilization |
|---|---|---|---:|
| 2026-08-28 | `q4_0-q8_0-IME-m4i` | Rolled inner loop baseline before the M2 accumulator change | 6.28% |
| 2026-08-28 | `q4_0-q8_0-IME-m4i` | Keep four M2 accumulators; convert and accumulate per M2 instead of rebuilding an M8 per block | 6.63% |
| 2026-09-16 | `q4_0-q8_0-IME-m4i` | Move the N tiling and the acc -> C copy into the adapter; the microkernel accumulates into the caller's `acc[kMr][kNr]` tile | 7.01% |

## q4_0-q8_0-IME-m4i-CB

| Date | Kernel | Change | Compute-unit utilization |
|---|---|---|---:|
| 2026-09-16 | `q4_0-q8_0-IME-m4i-CB` | K-panel blocked baseline with a staging C tile and a per-panel C update pass | 6.29% |
| 2026-09-16 | `q4_0-q8_0-IME-m4i-CB` | Let the microkernel accumulate straight into the packed C tile: no staging tile, no second C pass | 6.27% |

The 2026-09-16 microkernel ABI refactor pays off for the plain variant: an
interleaved A/B session at `M480 N1536 K1536` gives 131.7 M -> 126.2 M cycles
(-4.2%) at one thread and 40.2 M -> 37.1 M (-7.8%) at four threads, with parity
at `M480 N8960 K1536`. The blocked variant is at parity at one thread, but the
C accumulation now runs as sixteen narrow read-modify-write accesses inside the
microkernel instead of a wide add loop in the adapter, which costs 45.2 M ->
47.1 M cycles (+4%) at four threads on `M480 N1536 K1536` while the K-heavy
shape stays at parity.
