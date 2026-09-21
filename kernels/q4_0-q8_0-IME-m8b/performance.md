# Real-model-shape hardware cycles

## Kernel

Kernel: `q4_0-q8_0-IME-m8b`

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
Binary SHA-256:         1e191abf8af27226904cd54f137c469bb6fb250e57c8738533d1089d82c8cfed
```

## Hardware cycle results

| Model configuration | Matrix dimensions | M | N | K | Threads | Median cycles | Average per-core utilization | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 1 | 124,987,856 | 7.08% | 1.00x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 2 | 64,077,359 | 6.90% | 1.95x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 3 | 43,064,537 | 6.85% | 2.90x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 4 | 32,753,597 | 6.75% | 3.82x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 1 | 720,355,482 | 7.16% | 1.00x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 2 | 368,975,358 | 6.99% | 1.95x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 3 | 247,476,747 | 6.95% | 2.91x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 4 | 186,871,661 | 6.90% | 3.85x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 1 | 711,028,589 | 7.26% | 1.00x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 2 | 353,711,830 | 7.30% | 2.01x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 3 | 244,516,449 | 7.04% | 2.91x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 4 | 219,544,885 | 5.88% | 3.24x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 1 | 880,422,485 | 7.15% | 1.00x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 2 | 446,101,446 | 7.05% | 1.97x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 3 | 304,529,967 | 6.89% | 2.89x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 4 | 243,148,193 | 6.47% | 3.62x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 1 | 2,298,534,633 | 7.36% | 1.00x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 2 | 1,178,096,908 | 7.18% | 1.95x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 3 | 802,721,977 | 7.02% | 2.86x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 4 | 682,023,005 | 6.20% | 3.37x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 1 | 2,288,717,542 | 7.39% | 1.00x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 2 | 1,164,362,925 | 7.26% | 1.97x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 3 | 829,082,475 | 6.80% | 2.76x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 4 | 775,198,114 | 5.45% | 2.95x |
