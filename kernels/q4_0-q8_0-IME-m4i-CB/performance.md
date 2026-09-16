# Real-model-shape hardware cycles

## Kernel

Kernel: `q4_0-q8_0-IME-m4i-CB`

## Cache-blocking parameters

This variant reuses the `q4_0-q8_0-IME-m4i` microkernel (`kernel.h` declares it;
the implementation stays in the `q4_0-q8_0-IME-m4i` directory) and only changes
the loop structure, so the packed B stream stops being re-read once per M tile:

```text
K panel:            kBlocksPerPanel = 32 packed blocks (1024 K values)
L1D panel pair:     packed B (32 * 288 B = 9 KiB) + activation (32 * 144 B =
                    4.5 KiB) = 13.5 KiB resident, i.e. 42% of the 32 KiB L1D
activation chunk:   kAChunkBudgetBytes = 128 KiB -> 28 M tiles per chunk
parallel dimension: N tiles (one N16 tile per parallel iteration)
C accumulation:     one kMr * kNr tile per (M tile, N tile) in a packed C
                    buffer, accumulated across the K panels
```

`kBlocksPerPanel = 32` is a deliberate L1D budget choice: one K block needs a
packed B block (288 B) plus an activation block (4 floats + 4 * QK8_0 = 144 B),
so 32 blocks hold 13.5 KiB resident and leave room for the C tile and the
microkernel scratch in the 32 KiB L1D. 64 blocks would need 27 KiB and crowd out
the streaming activation panel; 16 blocks halve the footprint again but cost
12-19% at one thread through per-call and packed-C traffic. Halving the panel from
64 to 32 costs 3-8% at four threads but doubles the chunk tile count (14 -> 28),
i.e. it halves the packed B reloads.

## Test platform and environment

```text
Host:                   spacemit-k1 (bpi-f3)
CPU:                    Spacemit X60
Pinned CPUs:            0-(threads-1)
OS:                     Bianbu 2.2.1
Kernel:                 Linux 6.6.63, riscv64
CPU frequency governor: performance
Current/max frequency:  1.6 GHz / 1.6 GHz
Compiler:               Clang 23.0.0git, commit bfe57bf4eeda21ce7301a0d226bbfd2335df5f6b
Architecture:           rv64gcv_zvfh_zicbop_zba_xsmtvdot
Tune:                   spacemit-x60
Vector length:          fixed at 256 bits
Build type:             CMake Release
Base commit:            5b43e42ac
Worktree:               q4_0 IME cache-blocked variants
Binary SHA-256:         692a18247e18d6dfc577b6d3c4f0d3c0d2034bc33592377a68093a62332ad727
```

## Hardware cycle results

| Model configuration | Matrix dimensions | M | N | K | Threads | Median cycles | Average per-core utilization | Speedup | 3t/4t |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 1 | 141,445,326 | 6.25% | 1.00x | |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 2 | 87,334,430 | 5.07% | 1.62x | |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 3 | 58,359,577 | 5.05% | 2.42x | |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 4 | 44,764,775 | 4.94% | 3.16x | 1.304 |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 1 | 816,154,187 | 6.32% | 1.00x | |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 2 | 496,848,153 | 5.19% | 1.64x | |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 3 | 325,777,470 | 5.28% | 2.51x | |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 4 | 249,164,519 | 5.18% | 3.28x | 1.307 |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 1 | 768,017,448 | 6.72% | 1.00x | |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 2 | 396,674,191 | 6.51% | 1.94x | |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 3 | 313,036,587 | 5.50% | 2.45x | |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 4 | 229,346,767 | 5.63% | 3.35x | 1.365 |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 1 | 930,432,035 | 6.76% | 1.00x | |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 2 | 495,137,118 | 6.35% | 1.88x | |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 3 | 325,446,816 | 6.44% | 2.86x | |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 4 | 268,793,211 | 5.85% | 3.46x | 1.211 |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 1 | 2,471,424,034 | 6.84% | 1.00x | |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 2 | 1,273,706,521 | 6.64% | 1.94x | |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 3 | 861,138,126 | 6.54% | 2.87x | |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 4 | 665,803,357 | 6.35% | 3.71x | 1.293 |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 1 | 2,506,018,011 | 6.75% | 1.00x | |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 2 | 1,334,271,851 | 6.34% | 1.88x | |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 3 | 875,562,403 | 6.44% | 2.86x | |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 4 | 733,686,653 | 5.76% | 3.42x | 1.193 |
