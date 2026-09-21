# Real-model-shape hardware cycles

## Kernel

Kernel: `q4_0-q8_0-IME-m8b-CB`

## Cache-blocking parameters

This variant reuses the `q4_0-q8_0-IME-m8b` microkernel (`kernel.h` declares it;
the implementation stays in the `q4_0-q8_0-IME-m8b` directory) and only changes
the loop structure: K is split into panels and an activation chunk is walked
inside each panel, so a packed B panel is reused by every M tile of the chunk
while it stays L1D-resident:

```text
K panel:            kBlocksPerPanel = 32 packed blocks (1024 K values)
L1D panel pair:     packed B (32 * 288 B = 9 KiB) + activation (32 * 288 B =
                    9 KiB) = 18 KiB resident, i.e. 56% of the 32 KiB L1D
activation chunk:   kAChunkBudgetBytes = 128 KiB -> 14 M tiles per chunk
parallel dimension: N tiles (one N16 tile per parallel iteration)
C accumulation:     one kMr * kNr tile per (M tile, N tile) in a packed C
                    buffer; the microkernel accumulates that tile directly
                    across the K panels
```

`kBlocksPerPanel = 32` is a deliberate L1D budget choice: one K block needs a
packed B block (256 B) plus scales (32 B) and an activation block (256 B) plus
scales (32 B), so 32 blocks hold 18 KiB resident and leave about a third of the
L1D for the microkernel's 4 KiB reduction staging buffer; the fp32 accumulator is
the packed C tile itself, so the microkernel keeps no additional accumulator
scratch. 64 blocks would need 36 KiB and evict the very panel the chunk is meant
to reuse, so 32 is the largest panel that stays L1D-resident.

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
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 1 | 125,910,374 | 7.03% | 1.00x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 2 | 70,886,501 | 6.24% | 1.78x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 3 | 49,047,291 | 6.01% | 2.57x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 4 | 38,699,602 | 5.72% | 3.25x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 1 | 785,959,197 | 6.57% | 1.00x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 2 | 497,950,788 | 5.18% | 1.58x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 3 | 330,723,521 | 5.20% | 2.38x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 4 | 254,505,025 | 5.07% | 3.09x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 1 | 714,883,636 | 7.22% | 1.00x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 2 | 380,999,106 | 6.77% | 1.88x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 3 | 265,311,013 | 6.48% | 2.69x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 4 | 204,082,955 | 6.32% | 3.50x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 1 | 874,072,694 | 7.20% | 1.00x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 2 | 456,271,400 | 6.89% | 1.92x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 3 | 309,794,890 | 6.77% | 2.82x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 4 | 245,569,944 | 6.40% | 3.56x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 1 | 2,341,019,593 | 7.22% | 1.00x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 2 | 1,211,118,403 | 6.98% | 1.93x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 3 | 827,395,390 | 6.81% | 2.83x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 4 | 640,165,725 | 6.60% | 3.66x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 1 | 2,344,809,039 | 7.21% | 1.00x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 2 | 1,242,602,477 | 6.80% | 1.89x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 3 | 842,527,655 | 6.69% | 2.78x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 4 | 674,639,305 | 6.27% | 3.48x |
