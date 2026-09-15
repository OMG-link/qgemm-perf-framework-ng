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
Base commit:            cb35a30ae504308e612e2267fd102a24548830b0
Worktree:               m4i OpenMP and qs-first packed-W changes
Binary SHA-256:         f0f264e9811377a558186ef4a3e2090fe8dec318fde486457d86ca145b7c32d9
```

## Packed-W layout

The m4i and m4b IME N16 packed-W block stores the quantized values before the scales:

```text
offset 0:   qs[256]
offset 256: d[16] (32 bytes of FP16 scales)
block size: 288 bytes
```

The qs-first layout matches the kernels' access order: each K block consumes the
The upstream IME adapter converts the shared packed data to a separate
scale-first compatibility buffer during preparation, so its hand-written
assembly remains unchanged.

At the historical `M=480, N=1536, K=1536` shape, the qs-first single-thread
result is `134,316,293` cycles and 6.59% utilization, consistent with the 6.63%
entry in `docs/performance-history.md`.

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
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 1 | 134,316,293 | 6.59% | 1.00x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 2 | 68,678,964 | 6.44% | 1.96x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 3 | 48,281,182 | 6.11% | 2.78x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 4 | 40,474,699 | 5.46% | 3.32x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 1 | 746,336,481 | 6.92% | 1.00x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 2 | 377,928,392 | 6.83% | 1.97x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 3 | 283,372,096 | 6.07% | 2.63x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 4 | 246,666,742 | 5.23% | 3.03x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 1 | 726,495,325 | 7.10% | 1.00x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 2 | 367,566,848 | 7.02% | 1.98x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 3 | 262,386,044 | 6.56% | 2.77x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 4 | 266,157,799 | 4.85% | 2.73x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 1 | 937,961,237 | 6.71% | 1.00x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 2 | 475,724,800 | 6.61% | 1.97x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 3 | 339,854,408 | 6.17% | 2.76x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 4 | 305,893,655 | 5.14% | 3.07x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 1 | 2,457,058,962 | 6.88% | 1.00x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 2 | 1,241,423,187 | 6.81% | 1.98x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 3 | 860,637,278 | 6.55% | 2.85x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 4 | 811,642,036 | 5.21% | 3.03x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 1 | 2,361,785,923 | 7.16% | 1.00x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 2 | 1,192,610,694 | 7.09% | 1.98x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 3 | 881,644,650 | 6.39% | 2.68x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 4 | 940,709,208 | 4.49% | 2.51x |