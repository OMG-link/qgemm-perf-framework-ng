# Real-model-shape hardware cycles

## Kernel

Kernel: `q4_0-q8_0-IME-m8b-DynPreUnpack-CacheBlocking`

## Test platform and environment

```text
Host:                   spacemit-k1 (bpi-f3)
CPU:                    Spacemit X60
Pinned CPU:             0
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
Source commit:          e385b105816707c589c2b731225a1758f0591c9d
Binary SHA-256:         a827b4d7e7a8cae46290b7a355993c3183a7f82b5a3ae356dfab638e9252442e
```

## Hardware cycle results

| Model configuration | Matrix dimensions | M | N | K | Median cycles | Compute-unit utilization |
|---|---|---:|---:|---:|---:|---:|
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 129,854,426 | 6.81% |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 759,200,979 | 6.80% |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 769,379,153 | 6.71% |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 938,892,114 | 6.70% |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 2,505,833,105 | 6.75% |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 2,706,353,579 | 6.25% |
