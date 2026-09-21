# Real-model-shape hardware cycles

## Kernel

Kernel: `q4_0-q8_0-IME-m4i`

## Test platform and environment

```text
Host:                   spacemit-k1 (bpi-f3)
CPU:                    Spacemit X60
Pinned CPUs:            0-(threads-1)
OS:                     Bianbu 2.2.1
Kernel:                 Linux 6.6.63, riscv64
CPU frequency governor: performance
Current/max frequency:  1.6 GHz / 1.6 GHz
ASLR:                   disabled for benchmark invocation with setarch -R
Compiler:               Clang 23.0.0git, commit bfe57bf4eeda21ce7301a0d226bbfd2335df5f6b
Architecture:           rv64gcv_zvfh_zicbop_zba_xsmtvdot
Tune:                   spacemit-x60
Vector length:          fixed at 256 bits
Build type:             CMake Release
Base commit:            ad4c38572f699edc42be4ddb75f64d2405bfac63
Worktree:               ad4c38572f699edc42be4ddb75f64d2405bfac63
Binary SHA-256:         3fa3efa464f82e1a980a825643ee56b1fad98ed4b7e09d200fd28b21afb7b7b2
```

## Packed-W layout

The m4i and m4b IME N16 packed-W block stores the quantized values before the scales:

```text
offset 0:   qs[256]
offset 256: d[16] (32 bytes of FP16 scales)
block size: 288 bytes
```

The qs-first layout matches the kernels' access order: each K block consumes its
256 B of quants before its 32 B of scales.
The upstream IME adapter converts the shared packed data to a separate
scale-first compatibility buffer during preparation, so its hand-written
assembly remains unchanged.

At `M=480, N=1536, K=1536`, the single-thread result is `133,035,410` cycles and
6.65% utilization.

## Measurement method

Each result is the median of seven hardware-cycle samples with three warmup
iterations and three measured kernel iterations per sample. The reported cycle
count is normalized per kernel iteration. The process is pinned to CPU
`0-(threads-1)` with `OMP_DYNAMIC=FALSE`. All cycle samples had 100% PMU running
time. Input preparation, packing, reset, checksum, and output formatting are
outside the measured region.

The average per-core utilization is calculated as:

```text
(M * N * K / median_cycles) / (threads * 128 FMA/cycle) * 100%
```

The real-model-shape runs used `--no-verify`. A separate reference check at
`M=24, N=32, K=256` passed for 1, 2, 3, and 4 threads with zero differing
elements, maximum absolute error `1.90735e-06`, and checksum `3.84864689`.

## Hardware cycle results

| Model configuration | Matrix dimensions | M | N | K | Threads | Median cycles | Average per-core utilization | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 1 | 133,035,410 | 6.65% | 1.00x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 2 | 68,346,561 | 6.47% | 1.95x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 3 | 46,622,873 | 6.33% | 2.85x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 4 | 42,062,366 | 5.26% | 3.16x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 1 | 772,240,223 | 6.68% | 1.00x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 2 | 396,325,026 | 6.51% | 1.95x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 3 | 279,481,961 | 6.16% | 2.76x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 4 | 257,528,454 | 5.01% | 3.00x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 1 | 714,901,590 | 7.22% | 1.00x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 2 | 361,655,370 | 7.14% | 1.98x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 3 | 265,712,725 | 6.47% | 2.69x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 4 | 270,200,145 | 4.78% | 2.65x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 1 | 909,895,736 | 6.91% | 1.00x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 2 | 464,430,789 | 6.77% | 1.96x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 3 | 318,730,305 | 6.58% | 2.85x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 4 | 299,688,043 | 5.25% | 3.04x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 1 | 2,377,098,055 | 7.11% | 1.00x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 2 | 1,214,799,351 | 6.96% | 1.96x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 3 | 845,708,274 | 6.66% | 2.81x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 4 | 827,293,872 | 5.11% | 2.87x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 1 | 2,346,809,911 | 7.20% | 1.00x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 2 | 1,185,772,011 | 7.13% | 1.98x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 3 | 885,973,018 | 6.36% | 2.65x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 4 | 945,648,941 | 4.47% | 2.48x |
