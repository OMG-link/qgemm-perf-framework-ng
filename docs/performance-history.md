# Kernel performance history

This document is updated as kernel implementations change. The only performance
metric recorded is compute-unit utilization relative to the project peak of 128
FMA/cycle.

Unless noted otherwise, measurements use `M=480`, `N=1536`, `K=1536`, a CMake
Release build with a fixed 256-bit vector length, and are pinned to CPU 0 on
`spacemit-k1`.

## m8-batch-reduction

| Date | Kernel | Change | Compute-unit utilization |
|---|---|---|---:|
| 2026-08-25 | `m8-batch-reduction` | Fully unrolled inner loop | 5.38% |
| 2026-08-25 | `m8-batch-reduction` | Disable inner-loop unrolling | 5.96% |
| 2026-08-26 | `m8-batch-reduction` | Use four loose M2 reduction accumulators | 6.14% |
| 2026-08-27 | `m8-batch-reduction` | Convert eight M2 accumulators before rebuilding two M8 results | 6.24% |
| 2026-08-27 | `m8-batch-reduction` | Use SoA weight planes for M8 batch reduction | 6.45% |
| 2026-08-27 | `m8-batch-reduction` | 调整指令排布（通过添加 barrier 以及延后 fcvt） | 6.71% |
| 2026-08-28 | `m8-batch-reduction` | Expand component writeback and batch reduction to 8x16; reduce four rows per spill-free group | 6.99% |

## m4-immediate-reduction

| Date | Kernel | Change | Compute-unit utilization |
|---|---|---|---:|
| 2026-08-28 | `m4-immediate-reduction` | Rolled inner loop baseline before the M2 accumulator change | 6.28% |
| 2026-08-28 | `m4-immediate-reduction` | Keep four M2 accumulators; convert and accumulate per M2 instead of rebuilding an M8 per block | 6.63% |
