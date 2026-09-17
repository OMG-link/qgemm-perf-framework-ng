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
Base commit:            f647fc137c24c3dc74290216807c8bf8973c220d
Worktree:               microkernel ABI refactor (uncommitted)
Binary SHA-256:         11bcdd2f96e5ddc8358807256661a5f31e877aef6291cb98d656be74592757de
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

At `M=480, N=1536, K=1536`, the single-thread result is `126,186,078` cycles and
7.01% utilization.

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
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 1 | 126,186,078 | 7.01% | 1.00x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 2 | 65,114,694 | 6.79% | 1.94x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 3 | 44,591,807 | 6.61% | 2.83x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 4 | 37,720,444 | 5.86% | 3.35x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 1 | 754,702,489 | 6.84% | 1.00x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 2 | 387,086,463 | 6.67% | 1.95x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 3 | 270,839,217 | 6.35% | 2.79x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 4 | 248,311,042 | 5.20% | 3.04x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 1 | 725,198,346 | 7.12% | 1.00x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 2 | 367,921,206 | 7.01% | 1.97x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 3 | 265,873,217 | 6.47% | 2.73x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 4 | 272,402,130 | 4.74% | 2.66x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 1 | 919,998,205 | 6.84% | 1.00x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 2 | 471,079,683 | 6.68% | 1.95x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 3 | 332,106,879 | 6.31% | 2.77x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 4 | 305,466,424 | 5.15% | 3.01x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 1 | 2,464,121,435 | 6.86% | 1.00x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 2 | 1,265,317,233 | 6.68% | 1.95x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 3 | 888,046,134 | 6.35% | 2.77x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 4 | 827,405,651 | 5.11% | 2.98x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 1 | 2,353,122,220 | 7.19% | 1.00x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 2 | 1,194,051,505 | 7.08% | 1.97x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 3 | 907,587,981 | 6.21% | 2.59x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 4 | 987,329,455 | 4.28% | 2.38x |