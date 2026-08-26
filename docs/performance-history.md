# Kernel performance history

This document is updated as kernel implementations change. The only performance
metric recorded is compute-unit utilization relative to the project peak of 128
FMA/cycle.

Unless noted otherwise, measurements use `M=480`, `N=1536`, `K=1536`, a CMake
Release build with a fixed 256-bit vector length, and are pinned to CPU 0 on
`spacemit-k1`.

| Date | Kernel | Change | Compute-unit utilization |
|---|---|---|---:|
| 2026-08-25 | `m8-batch-reduction` | Fully unrolled inner loop | 5.38% |
| 2026-08-25 | `m8-batch-reduction` | Disable inner-loop unrolling | 5.96% |
| 2026-08-26 | `m8-batch-reduction` | Use four loose M2 reduction accumulators | 6.14% |