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
Base commit:            f647fc137c24c3dc74290216807c8bf8973c220d
Worktree:               microkernel ABI refactor (uncommitted)
Binary SHA-256:         11bcdd2f96e5ddc8358807256661a5f31e877aef6291cb98d656be74592757de
```

## Hardware cycle results

| Model configuration | Matrix dimensions | M | N | K | Threads | Median cycles | Average per-core utilization | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 1 | 139,393,498 | 6.35% | 1.00x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 2 | 88,576,476 | 4.99% | 1.57x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 3 | 65,756,356 | 4.48% | 2.12x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 4 | 47,072,524 | 4.70% | 2.96x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 1 | 805,980,918 | 6.40% | 1.00x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 2 | 501,719,750 | 5.14% | 1.61x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 3 | 330,160,257 | 5.21% | 2.44x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 4 | 266,992,567 | 4.83% | 3.02x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 1 | 748,317,958 | 6.90% | 1.00x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 2 | 388,713,872 | 6.64% | 1.93x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 3 | 298,811,828 | 5.76% | 2.50x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 4 | 218,439,268 | 5.91% | 3.43x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 1 | 990,154,391 | 6.35% | 1.00x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 2 | 530,649,960 | 5.93% | 1.87x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 3 | 364,510,785 | 5.75% | 2.72x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 4 | 286,720,739 | 5.49% | 3.45x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 1 | 2,422,982,149 | 6.98% | 1.00x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 2 | 1,251,662,413 | 6.75% | 1.94x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 3 | 849,243,155 | 6.64% | 2.85x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 4 | 667,723,355 | 6.33% | 3.63x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 1 | 2,792,084,901 | 6.06% | 1.00x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 2 | 1,453,504,688 | 5.82% | 1.92x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 3 | 984,989,525 | 5.72% | 2.83x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 4 | 766,755,464 | 5.51% | 3.64x |
