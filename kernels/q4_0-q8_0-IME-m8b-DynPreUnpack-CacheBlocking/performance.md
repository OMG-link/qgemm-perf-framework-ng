# Real-model-shape hardware cycles

## Kernel

Kernel: `q4_0-q8_0-IME-m8b-DynPreUnpack-CacheBlocking`

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
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 1 | 130,160,271 | 6.80% |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 2 | 71,563,311 | 6.18% |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 3 | 51,439,175 | 5.73% |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 4 | 42,473,043 | 5.21% |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 1 | 757,753,597 | 6.81% |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 2 | 415,507,200 | 6.21% |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 3 | 294,319,841 | 5.85% |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 4 | 236,689,449 | 5.45% |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 1 | 825,666,283 | 6.25% |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 2 | 436,659,638 | 5.91% |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 3 | 327,237,552 | 5.26% |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 4 | 250,238,987 | 5.16% |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 1 | 1,003,021,504 | 6.27% |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 2 | 548,691,084 | 5.73% |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 3 | 373,454,877 | 5.62% |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 4 | 327,615,123 | 4.80% |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 1 | 2,486,280,765 | 6.80% |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 2 | 1,348,914,151 | 6.27% |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 3 | 993,236,313 | 5.67% |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 4 | 793,856,524 | 5.32% |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 1 | 2,834,399,225 | 5.97% |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 2 | 1,550,506,124 | 5.45% |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 3 | 1,086,638,213 | 5.19% |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 4 | 905,853,422 | 4.67% |
