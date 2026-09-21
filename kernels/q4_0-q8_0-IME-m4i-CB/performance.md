# Real-model-shape hardware cycles

## Kernel

Kernel: `q4_0-q8_0-IME-m4i-CB`

## Cache-blocking parameters

This variant reuses the `q4_0-q8_0-IME-m4i` microkernel (`kernel.h` declares it;
the implementation stays in the `q4_0-q8_0-IME-m4i` directory) and only changes
the loop structure: K is split into panels and an activation chunk is walked
inside each panel, so a packed B panel is reused by every M tile of the chunk
while it stays L1D-resident:

```text
K panel:            kBlocksPerPanel = 32 packed blocks (1024 K values)
L1D panel pair:     packed B (32 * 288 B = 9 KiB) + activation (32 * 144 B =
                    4.5 KiB) = 13.5 KiB resident, i.e. 42% of the 32 KiB L1D
activation chunk:   kAChunkBudgetBytes = 128 KiB -> 28 M tiles per chunk
parallel dimension: N tiles (one N16 tile per parallel iteration)
C accumulation:     one kMr * kNr tile per (M tile, N tile) in a packed C
                    buffer; the microkernel accumulates that tile directly
                    across the K panels
```

`kBlocksPerPanel = 32` is a deliberate L1D budget choice: one K block needs a
packed B block (288 B) plus an activation block (4 floats + 4 * QK8_0 = 144 B),
so 32 blocks hold 13.5 KiB resident and leave room for the C tile and the
microkernel scratch in the 32 KiB L1D; the fp32 accumulator is the packed C tile
itself, so the microkernel keeps no additional accumulator scratch. 64 blocks
would need 27 KiB and crowd out the streaming activation panel, so 32 is the
largest panel that stays L1D-resident.

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
Base commit:            ad4c38572f699edc42be4ddb75f64d2405bfac63
Worktree:               ad4c38572f699edc42be4ddb75f64d2405bfac63
Binary SHA-256:         3fa3efa464f82e1a980a825643ee56b1fad98ed4b7e09d200fd28b21afb7b7b2
```

## Hardware cycle results

| Model configuration | Matrix dimensions | M | N | K | Threads | Median cycles | Average per-core utilization | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 1 | 144,387,376 | 6.13% | 1.00x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 2 | 91,003,618 | 4.86% | 1.59x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 3 | 66,547,243 | 4.43% | 2.17x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 4 | 47,919,042 | 4.62% | 3.01x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 1 | 825,312,958 | 6.25% | 1.00x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 2 | 512,347,490 | 5.04% | 1.61x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 3 | 335,655,874 | 5.13% | 2.46x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 4 | 260,521,305 | 4.95% | 3.17x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 1 | 747,918,432 | 6.90% | 1.00x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 2 | 387,467,540 | 6.66% | 1.93x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 3 | 290,091,438 | 5.93% | 2.58x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 4 | 215,101,575 | 6.00% | 3.48x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 1 | 959,377,103 | 6.56% | 1.00x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 2 | 514,068,203 | 6.12% | 1.87x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 3 | 344,776,798 | 6.08% | 2.78x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 4 | 280,850,548 | 5.60% | 3.42x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 1 | 2,438,723,904 | 6.93% | 1.00x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 2 | 1,257,924,851 | 6.72% | 1.94x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 3 | 854,313,420 | 6.60% | 2.85x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 4 | 666,302,979 | 6.34% | 3.66x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 1 | 2,701,874,686 | 6.26% | 1.00x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 2 | 1,401,826,551 | 6.03% | 1.93x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 3 | 951,643,744 | 5.92% | 2.84x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 4 | 746,258,334 | 5.66% | 3.62x |
