# Real-model-shape hardware cycles

## Kernel

Kernel: `q4_0-q8_0-IME-m8b`

## Test platform and environment

```text
Platform ID:            spacemit-k3-a100
Host:                   spacemit-k3 (k3-pico-itx)
CPU:                    SpacemiT A100
Pinned CPUs:            8-(8+threads-1)
OS:                     Bianbu 4.0rc2
Kernel:                 Linux 6.18.3-generic, riscv64
CPU frequency governor: userspace
Current/max frequency:  1.8 GHz / 2.0 GHz
Compiler:               Clang 24.0.0git, commit f55e9a9c98b14b6ccb835731d8bd62c0e61c3ed4
Architecture:           rv64gcv_zvfh_zicbop_zba_zvl1024b_xsmtvdotii
Tune:                   spacemit-a100
Vector length:          fixed at 1024 bits
Component buffer:       16 KiB in TCM, /dev/tcm block 0 (owned by cpus 8 and 9)
Build type:             CMake Release
Base commit:            59a5fe2a9b28ab4efa0d09ee9b40731f7343d1b1
Worktree:               smt-ime2 implementation (uncommitted)
Binary SHA-256:         7e376a93f85e11eb00975cc4374945bbcefb346a1d78e1f2c8ab8360a6088731
```

## Hardware cycle results

| Model configuration | Matrix dimensions | M | N | K | Threads | Median cycles | Average per-core utilization | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 1 | 48,499,036 | 2.28% | 1.00x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 2 | 24,286,643 | 2.28% | 2.00x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 3 | 16,215,178 | 2.27% | 2.99x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 4 | 12,995,748 | 2.13% | 3.73x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 8 | 6,568,521 | 2.10% | 7.38x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 1 | 279,867,511 | 2.31% | 1.00x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 2 | 140,968,597 | 2.29% | 1.99x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 3 | 94,610,561 | 2.27% | 2.96x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 4 | 75,680,968 | 2.13% | 3.70x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 8 | 37,631,462 | 2.14% | 7.44x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 1 | 274,844,158 | 2.35% | 1.00x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 2 | 138,681,302 | 2.33% | 1.98x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 3 | 93,518,010 | 2.30% | 2.94x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 4 | 76,795,236 | 2.10% | 3.58x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 8 | 38,347,139 | 2.10% | 7.17x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 1 | 336,970,949 | 2.33% | 1.00x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 2 | 169,243,859 | 2.32% | 1.99x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 3 | 113,600,762 | 2.31% | 2.97x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 4 | 91,495,834 | 2.15% | 3.68x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 8 | 45,298,218 | 2.17% | 7.44x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 1 | 907,167,545 | 2.33% | 1.00x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 2 | 454,943,898 | 2.32% | 1.99x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 3 | 304,446,919 | 2.31% | 2.98x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 4 | 245,899,159 | 2.15% | 3.69x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 8 | 122,052,662 | 2.16% | 7.43x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 1 | 903,393,638 | 2.34% | 1.00x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 2 | 456,143,146 | 2.32% | 1.98x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 3 | 313,681,737 | 2.25% | 2.88x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 4 | 247,519,018 | 2.13% | 3.65x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 8 | 126,701,903 | 2.09% | 7.13x |
