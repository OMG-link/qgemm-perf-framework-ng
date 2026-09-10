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
Base commit:            809e7b0faf059800aef3f088f5ec80b142a7b5d2
Worktree:               OpenMP changes
Binary SHA-256:         b2e65ae166384290ff21d0b58b9c2dd0b44cae650fa21faf3e00a950145b7cca
```

## Hardware cycle results

| Model configuration | Matrix dimensions | M | N | K | Threads | Median cycles | Average per-core utilization |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 1 | 124,295,835 | 7.12% |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 2 | 63,573,801 | 6.96% |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 3 | 43,129,095 | 6.84% |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 4 | 32,825,023 | 6.74% |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 1 | 719,564,616 | 7.17% |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 2 | 365,792,245 | 7.05% |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 3 | 247,731,297 | 6.94% |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 4 | 188,689,583 | 6.84% |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 1 | 688,580,011 | 7.50% |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 2 | 350,552,646 | 7.36% |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 3 | 245,508,857 | 7.01% |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 4 | 220,063,158 | 5.86% |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 1 | 880,455,868 | 7.15% |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 2 | 468,629,601 | 6.71% |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 3 | 304,272,089 | 6.89% |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 4 | 235,610,745 | 6.68% |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 1 | 2,291,583,991 | 7.38% |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 2 | 1,162,145,455 | 7.27% |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 3 | 795,170,975 | 7.09% |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 4 | 612,056,505 | 6.91% |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 1 | 2,262,407,030 | 7.47% |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 2 | 1,168,982,210 | 7.23% |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 3 | 842,199,110 | 6.69% |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 4 | 799,712,563 | 5.29% |
