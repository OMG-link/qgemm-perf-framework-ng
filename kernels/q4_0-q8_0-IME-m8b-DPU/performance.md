# Real-model-shape hardware cycles

## Kernel

Kernel: `q4_0-q8_0-IME-m8b-DPU`

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
Base commit:            f3bb53dbf4ede367104682d9252c999637fdd4f4
Worktree:               128 B dynamic-unpack repack in both m8b unpack helpers
                        (uncommitted); git status --short lists those two kernel.cpp
                        files as modified plus this retest task document as untracked
Binary SHA-256:         d53bdfd50c631c56daf6fdfb7e5f7e39c26712861d4dd69d05ffd39b32861525
```

## Hardware cycle results

| Model configuration | Matrix dimensions | M | N | K | Threads | Median cycles | Average per-core utilization | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 1 | 141,110,971 | 6.27% | 1.00x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 2 | 76,038,042 | 5.82% | 1.86x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 3 | 54,075,565 | 5.45% | 2.61x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 4 | 46,282,742 | 4.78% | 3.05x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 1 | 816,648,889 | 6.32% | 1.00x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 2 | 432,819,901 | 5.96% | 1.89x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 3 | 314,403,401 | 5.47% | 2.60x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 4 | 274,783,011 | 4.70% | 2.97x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 1 | 811,321,747 | 6.36% | 1.00x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 2 | 428,077,167 | 6.03% | 1.90x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 3 | 370,738,712 | 4.64% | 2.19x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 4 | 358,288,449 | 3.60% | 2.26x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 1 | 1,042,979,626 | 6.03% | 1.00x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 2 | 529,404,791 | 5.94% | 1.97x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 3 | 392,103,037 | 5.35% | 2.66x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 4 | 363,776,462 | 4.32% | 2.87x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 1 | 2,729,281,496 | 6.20% | 1.00x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 2 | 1,497,505,476 | 5.65% | 1.82x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 3 | 1,131,701,786 | 4.98% | 2.41x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 4 | 1,054,638,272 | 4.01% | 2.59x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 1 | 2,684,744,216 | 6.30% | 1.00x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 2 | 1,510,716,942 | 5.60% | 1.78x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 3 | 1,281,255,317 | 4.40% | 2.10x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 4 | 1,277,637,042 | 3.31% | 2.10x |
