# Real-model-shape hardware cycles

## Kernel

Kernel: `q4_0-q8_0-IME-m8b-CB`

## Cache-blocking parameters

This variant reuses the `q4_0-q8_0-IME-m8b` microkernel (`kernel.h` declares it;
the implementation stays in the `q4_0-q8_0-IME-m8b` directory) and only changes
the loop structure, so the packed B stream stops being re-read once per M tile:

```text
K panel:            kBlocksPerPanel = 32 packed blocks (1024 K values)
L1D panel pair:     packed B (32 * 288 B = 9 KiB) + activation (32 * 288 B =
                    9 KiB) = 18 KiB resident, i.e. 56% of the 32 KiB L1D
activation chunk:   kAChunkBudgetBytes = 128 KiB -> 14 M tiles per chunk
parallel dimension: N tiles (one N16 tile per parallel iteration)
C accumulation:     one kMr * kNr tile per (M tile, N tile) in a packed C
                    buffer, accumulated across the K panels
```

`kBlocksPerPanel = 32` is a deliberate L1D budget choice: one K block needs a
packed B block (256 B) plus scales (32 B) and an activation block (256 B) plus
scales (32 B), so 32 blocks hold 18 KiB resident and leave about a third of the
L1D for the microkernel scratch (`acc[128]` and the 4 KiB reduction staging).
64 blocks would need 36 KiB and evict the very panel the chunk is meant to
reuse; 16 blocks halve the footprint again but cost 12-19% at one thread through
per-call and packed-C traffic. Halving the panel from 64 to 32 costs 3-8% at four
threads but doubles the chunk tile count (7 -> 14), i.e. it halves the packed B
reloads.

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
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 1 | 131,635,300 | 6.72% | 1.00x | |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 2 | 91,172,901 | 4.85% | 1.44x | |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 3 | 59,678,419 | 4.94% | 2.21x | |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 4 | 40,247,657 | 5.50% | 3.27x | 1.483 |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 1 | 824,364,184 | 6.26% | 1.00x | |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 2 | 519,761,955 | 4.96% | 1.59x | |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 3 | 341,972,905 | 5.03% | 2.41x | |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 4 | 259,142,496 | 4.98% | 3.18x | 1.320 |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 1 | 750,076,906 | 6.88% | 1.00x | |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 2 | 398,050,922 | 6.48% | 1.88x | |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 3 | 275,636,494 | 6.24% | 2.72x | |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 4 | 212,656,577 | 6.07% | 3.53x | 1.296 |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 1 | 912,902,951 | 6.89% | 1.00x | |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 2 | 474,864,922 | 6.62% | 1.92x | |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 3 | 321,907,581 | 6.51% | 2.84x | |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 4 | 253,345,301 | 6.21% | 3.60x | 1.271 |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 1 | 2,426,261,852 | 6.97% | 1.00x | |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 2 | 1,260,218,024 | 6.71% | 1.93x | |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 3 | 857,702,699 | 6.57% | 2.83x | |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 4 | 661,465,813 | 6.39% | 3.67x | 1.297 |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 1 | 2,444,860,379 | 6.92% | 1.00x | |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 2 | 1,291,251,602 | 6.55% | 1.89x | |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 3 | 878,470,132 | 6.42% | 2.78x | |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 4 | 696,932,172 | 6.07% | 3.51x | 1.260 |
