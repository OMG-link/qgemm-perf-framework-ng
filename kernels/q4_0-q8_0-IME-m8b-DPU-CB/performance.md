# Real-model-shape hardware cycles

## Kernel

Kernel: `q4_0-q8_0-IME-m8b-DPU-CB`

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
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 1 | 122,979,736 | 7.19% | 1.00x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 2 | 66,929,538 | 6.61% | 1.84x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 3 | 48,222,800 | 6.12% | 2.55x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 4 | 39,817,420 | 5.55% | 3.09x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 1 | 723,102,569 | 7.14% | 1.00x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 2 | 386,211,108 | 6.68% | 1.87x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 3 | 272,666,028 | 6.31% | 2.65x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 4 | 219,046,240 | 5.89% | 3.30x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 1 | 723,035,868 | 7.14% | 1.00x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 2 | 411,284,256 | 6.27% | 1.76x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 3 | 281,651,306 | 6.11% | 2.57x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 4 | 230,918,180 | 5.59% | 3.13x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 1 | 891,273,167 | 7.06% | 1.00x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 2 | 476,344,124 | 6.60% | 1.87x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 3 | 336,022,042 | 6.24% | 2.65x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 4 | 274,747,275 | 5.72% | 3.24x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 1 | 2,460,344,322 | 6.87% | 1.00x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 2 | 1,329,614,219 | 6.36% | 1.85x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 3 | 948,334,948 | 5.94% | 2.59x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 4 | 772,346,838 | 5.47% | 3.19x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 1 | 2,511,318,302 | 6.73% | 1.00x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 2 | 1,417,308,895 | 5.96% | 1.77x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 3 | 960,930,703 | 5.87% | 2.61x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 4 | 865,410,992 | 4.88% | 2.90x |
