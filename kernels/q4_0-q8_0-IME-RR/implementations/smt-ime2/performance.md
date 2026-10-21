# Real-model-shape hardware cycles

## Kernel

Kernel: `q4_0-q8_0-IME-RR`

## Test platform and environment

```text
Platform ID:            spacemit-k3-a100
Host:                   spacemit-k3 (k3-pico-itx)
CPU:                    SpacemiT A100
Pinned CPUs:            8-(8+threads-1)
OS:                     Bianbu 4.0rc2
Kernel:                 Linux 6.18.3-generic, riscv64
CPU frequency governor: userspace
Current/max frequency:  1.8 GHz / 2.0 GHz
Compiler:               Clang 24.0.0git, commit f55e9a9c98b14b6ccb835731d8bd62c0e61c3ed4
Architecture:           rv64gcv_zvfh_zicbop_zba_zvl1024b_xsmtvdotii
Tune:                   spacemit-a100
Vector length:          fixed at 1024 bits
Component buffer:       128 KiB packed C chunk in TCM, per-CPU 384 KiB arena
Build type:             CMake Release
Base commit:            5f5222a933bebff188c385156dd3210d5e3163fc
Worktree:               smt-ime2 implementation (uncommitted)
Binary SHA-256:         2495de06e0182ef3b2c99b8bf16b0a93c4283ac6c3af469596f0fea3671f5e83
```

## Hardware cycle results

| Model configuration | Matrix dimensions | M | N | K | Threads | Median cycles | Average per-core utilization | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 1 | 25,294,660 | 4.37% | 1.00x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 2 | 12,753,695 | 4.34% | 1.98x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 3 | 8,581,354 | 4.30% | 2.95x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 4 | 6,606,943 | 4.18% | 3.83x |
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 | 8 | 3,420,626 | 4.04% | 7.39x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 1 | 148,369,338 | 4.35% | 1.00x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 2 | 74,803,256 | 4.31% | 1.98x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 3 | 50,726,490 | 4.24% | 2.92x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 4 | 38,171,428 | 4.23% | 3.89x |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 | 8 | 18,961,774 | 4.25% | 7.82x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 1 | 136,845,096 | 4.71% | 1.00x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 2 | 68,645,023 | 4.70% | 1.99x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 3 | 45,918,430 | 4.68% | 2.98x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 4 | 34,521,506 | 4.67% | 3.96x |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 | 8 | 17,258,863 | 4.67% | 7.93x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 1 | 175,845,244 | 4.47% | 1.00x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 2 | 87,917,804 | 4.47% | 2.00x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 3 | 58,941,689 | 4.45% | 2.98x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 4 | 45,197,306 | 4.35% | 3.89x |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 | 8 | 22,660,262 | 4.34% | 7.76x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 1 | 468,188,919 | 4.51% | 1.00x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 2 | 233,687,788 | 4.52% | 2.00x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 3 | 155,839,483 | 4.52% | 3.00x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 4 | 116,676,752 | 4.53% | 4.01x |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 | 8 | 58,554,817 | 4.51% | 8.00x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 1 | 452,214,158 | 4.67% | 1.00x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 2 | 226,796,837 | 4.66% | 1.99x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 3 | 152,373,676 | 4.62% | 2.97x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 4 | 113,691,402 | 4.65% | 3.98x |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 | 8 | 57,104,192 | 4.63% | 7.92x |

## Inner-loop region cost

No-instrumentation `perf` PC sampling on the same board (`cpu-clock:u` distribution
scaled by an external `perf stat -e cycles:u` total) attributes the register-tile
inner loop of `q4_0_q8_0_ime_rr_w_stationary` at `M=480, N=1536, K=1536`, one
thread, as follows. One iteration is one execution of the `L74-L121` loop body.

| Source region | Cycles per iteration |
|---|---:|
| `L76-96` activation loads + int8 `smt.vmadot` | 45.25 |
| `L98-106` fp32 clear + fp16 `smt.vfwmadot` | 26.37 |
| `L108-119` four `REDUCE_GROUP` C reduce groups | 79.71 |
| `L74/L80` loop tail (supplementary) | 6.97 |
| whole kernel function | 161.87 |

`L108-119` is C-traffic bound: its four `vfmacc.vv` read-modify-write instructions
alone hold ~33% of the kernel's samples. `L76-96` is activation-load-latency bound;
the scalar instruction after the first `vl1r.v` is the skid target of that load
miss. Full method, address map, evidence and caveats are in
[2026-09-29-rr-per-region-cycles.md](../../../../docs/2026-09-29-rr-per-region-cycles.md).

