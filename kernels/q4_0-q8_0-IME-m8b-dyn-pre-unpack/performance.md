# Real-model-shape hardware cycles

## Kernel

Kernel: `q4_0-q8_0-IME-m8b-dyn-pre-unpack`

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
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 1 | 147,923,348 | 5.98% |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 2 | 86,099,700 | 5.14% |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 3 | 60,369,889 | 4.89% |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 4 | 52,342,724 | 4.23% |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 1 | 856,746,308 | 6.02% |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 2 | 572,047,306 | 4.51% |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 3 | 371,557,310 | 4.63% |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 4 | 304,846,504 | 4.23% |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 1 | 855,060,842 | 6.04% |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 2 | 502,039,978 | 5.14% |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 3 | 419,271,422 | 4.10% |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 4 | 400,417,211 | 3.22% |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 1 | 1,074,519,029 | 5.86% |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 2 | 693,563,517 | 4.54% |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 3 | 442,915,891 | 4.73% |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 4 | 404,126,552 | 3.89% |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 1 | 2,793,278,797 | 6.05% |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 2 | 1,542,699,040 | 5.48% |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 3 | 1,210,070,911 | 4.66% |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 4 | 1,123,663,170 | 3.76% |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 1 | 2,779,621,046 | 6.08% |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 2 | 1,619,671,643 | 5.22% |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 3 | 1,357,852,765 | 4.15% |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 4 | 1,360,327,287 | 3.11% |
