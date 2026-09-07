# DynPreUnpack inner-loop microbench (SpacemiT K1)

## Scope and method

Target: `q4_0-q8_0-IME-m8b-DynPreUnpack-CacheBlocking` inner iteration. The benchmark is independent assembly and does not modify kernel behavior. Each full iteration has one `vl8r.v` (256 B B), four `vle8.v` (4x32 B A), the same eight B physical registers (`v8..v15`), and 16 `smt.vmadot` updating eight m2 accumulators. A/B offsets wrap every 64 iterations inside allocated 8 KiB/16 KiB buffers, so no iteration is out of bounds. Eight accumulator groups are stored after the loop and consumed by a scalar checksum.

The runner pins itself to CPU0 and verifies `sched_getcpu()==0`, warms each case three times (4096 iterations), then records seven samples of 200,000 iterations. Compiler barriers and `fence rw,rw` surround the timed call. `rdcycle` is executed before and after every sample. On this K1/Linux image it always returned zero (also the reason the project timer uses monotonic time for `SPACEMIT_X60`), so **direct rdcycle results are invalid**. Reported cycle estimates use measured monotonic nanoseconds times the board's nominal 1.6 GHz, exactly matching `bench/timer.hpp`. Raw output retains both `rdcycle_delta=0` and `duration_ns`; these are frequency-based estimated cycles, not hardware cycle-counter observations.

Experiment A:

* `A_load_use`: one A m1 load, one B m8 whole-register load, then one consuming vmadot; `A_load_base` preserves address generation, vset transitions, loop and three issue slots using nops. The subtraction is a combined hot-L1 load-to-use path, not a pure single-load latency.
* `A_dot_chain`: 16 consecutive vmadot operations on one accumulator (true RAW chain), with A/B loaded once per outer iteration. `A_dot_base` replaces the 16 dots one-for-one with nops. `(median(chain)-median(base))/16` estimates dependent vmadot latency.
* `B_full` is the available full-body critical-path/steady-state measurement. There is no defensible single “critical path baseline” that removes all groups without changing issue-resource competition, so no isolated full-body latency is claimed.

Experiment B keeps loop/address generation, accumulator initialization, final stores and checksum. A removed B load is replaced by one nop; four A loads plus their three address `addi`s by seven nops; 16 dots by 16 nops. These preserve static instruction count but nops consume scalar decode/issue resources, so deltas are sensitivity measurements, not additive intrinsic instruction costs. The source-level eight-way B extraction is zero machine instructions (physical aliases v8..v15), hence a direct extraction ablation is impossible. `B_b_fanout1` is the valid alternative: all loads and 16 dots remain, but every dot reads v8; its delta only tests B-register fanout sensitivity.

## Build, disassembly and run

```bash
./compile-and-test.sh configure
cmake --build build --target dynpreunpack-inner-loop
$TOOLCHAIN_DIR/llvm-objdump -d --no-show-raw-insn \
  build/machine-benchmarks/dynpreunpack-inner-loop \
  > build/machine-benchmarks/dynpreunpack-inner-loop.objdump
scp build/machine-benchmarks/dynpreunpack-inner-loop spacemit-k1:/tmp/
ssh spacemit-k1 'taskset -c 0 /tmp/dynpreunpack-inner-loop --iterations 200000 --samples 7'
```

Final-ELF loop-body audit (each function has one backward `bltu`; eight `vs2r.v` checksum stores are after the loop):

| case | vl8r | vle8 | vmadot | replacement nops |
|---|---:|---:|---:|---:|
| B_full | 1 | 4 | 16 | 0 |
| B_no_bload | 0 | 4 | 16 | 1 |
| B_no_aloads | 1 | 0 | 16 | 7 |
| B_b_fanout1 | 1 | 4 | 16 | 0 |
| B_no_dots | 1 | 4 | 0 | 16 |
| A_load_use / base | 1 / 0 | 1 / 0 | 1 / 0 | 0 / 3 |
| A_dot_chain / base | 0 | 2 | 16 / 0 | 0 / 16 |

No loop was unrolled or deleted. The initial `vsetvli` uses a nonzero destination (`t5`) because K1 treated the x0/x0 vtype-change form as VILL and faulted the following vector load; this was discovered and corrected before collecting data.

## Raw results

Values below are estimated cycles/iteration for samples 0..6. Complete machine output, including duration_ns, rdcycle_delta and checksums, is in `build/machine-benchmarks/dynpreunpack-inner-loop-results.txt`.

| case | seven raw samples |
|---|---|
| A_load_use | 21.186768, 20.990760, 20.608752, 20.781752, 21.026432, 20.880088, 21.408768 |
| A_load_base | 12.007576, 12.814592, 12.002576, 12.001576, 12.457920, 12.006912, 12.372920 |
| A_dot_chain | 99.688360, 98.736008, 98.531328, 98.535664, 98.323664, 98.554336, 98.479008 |
| A_dot_base | 26.235200, 26.479536, 28.528576, 27.599560, 26.136192, 26.002192, 26.109536 |
| B_full | 46.573952, 47.141960, 46.522944, 46.400616, 46.446280, 46.341936, 46.469944 |
| B_no_bload | 32.207656, 32.096320, 32.099984, 32.256992, 32.129984, 32.098984, 32.092648 |
| B_no_aloads | 38.576448, 38.483448, 38.483112, 38.614784, 38.523448, 38.473448, 38.610784 |
| B_b_fanout1 | 46.468608, 46.355608, 46.417272, 46.378936, 46.548616, 46.942952, 46.763952 |
| B_no_dots | 43.346216, 43.359544, 43.566880, 43.340544, 43.360880, 43.457552, 43.331208 |

## Summary and interpretation

| case | median | min..max | full-minus-ablation delta |
|---|---:|---:|---:|
| A load/use path | 20.990760 | 20.608752..21.408768 | minus base = **8.983184** |
| A load/use base | 12.007576 | 12.001576..12.814592 | - |
| A 16-dot dependency chain | 98.535664 | 98.323664..99.688360 | minus base / 16 = **4.518779 cycles/dot** |
| A 16-nop base | 26.235200 | 26.002192..28.528576 | - |
| B full | **46.469944** | 46.341936..47.141960 | - |
| B no B load | 32.099984 | 32.092648..32.256992 | **14.369960** |
| B no A loads | 38.523448 | 38.473448..38.614784 | **7.946496** |
| B fanout=1 alternative | 46.468608 | 46.355608..46.942952 | **0.001336** (no measurable fanout penalty) |
| B no vmadot | 43.359544 | 43.331208..43.566880 | **3.110400** |

The three direct ablation deltas sum to 25.426856 cycles, but they are **not additive**: loads and dots overlap, removing one group changes dependency/issue pressure, and scalar nops are not resource-neutral. In particular, interpreting the small no-vmadot delta as “16 dots cost only 3.11 cycles” would be misleading; the 16 replacement nops themselves occupy front-end slots. The dependency-chain result is the more appropriate vmadot latency estimate.

## Validity risks

1. Highest-impact limitation: userspace `rdcycle` returns zero on this board image. Absolute “cycles” depend on the nominal 1.6 GHz conversion and will scale if DVFS changes. Nanoseconds are the primary observation; CPU0 pinning does not itself lock frequency.
2. The 8/16 KiB wrapped and prewarmed sets measure hot-cache execution, intentionally avoiding misses and out-of-bounds accesses; they do not represent streaming DRAM behavior.
3. Nop substitution preserves instruction count but not execution-port demand or data dependencies. B deltas are controlled sensitivity tests, not isolated costs and not additive.
4. Removed-load variants consume stale live vector registers left by the preceding warmup/case. This avoids introducing a replacement memory hierarchy and keeps the 16 dots/checksum live, but means their numerical checksum differs by construction and their operand values are not equivalent to full.
5. B extraction has no instruction to remove. The fanout alternative is explicitly not an extraction-cost measurement.
6. The benchmark uses a hand-fixed assembly schedule to ensure reproducibility. It matches the target group counts and dependency topology, but its address wrap arithmetic differs from the kernel's two-iteration inner-loop arithmetic.

## Hardware-cycle rerun

The runner now opens a pinned `PERF_COUNT_HW_CPU_CYCLES` event for the CPU0-affined current thread (`exclude_kernel=1`, `exclude_hv=1`). Each case is warmed three times before sampling; `PERF_EVENT_IOC_RESET/ENABLE` occurs immediately before the 200,000-iteration kernel call and `DISABLE/read` immediately after it. Operand setup, warmup, checksum, summaries, and output are therefore outside the counted interval. Wall time is retained only to compute an implied frequency. `--case NAME` is also supported for one-case-per-process runs and was used for the final collection.

Every raw sample had `time_running == time_enabled` (100.000%, no multiplexing). Seven raw hardware cycle counts:

| case | raw hardware cycles (200k iterations) | median cycles/iteration | change from first-round estimate |
|---|---|---:|---:|
| A_load_use | 4127705, 4270478, 4192136, 4113667, 4146910, 4161059, 4118431 | 20.734550 | -1.221% |
| A_load_base | 2416043, 2557675, 2406303, 2474419, 2424428, 2405863, 2429977 | 12.122140 | +0.954% |
| A_dot_chain | 19807067, 19949911, 19767453, 19751709, 19709017, 19752290, 19998411 | 98.837265 | +0.306% |
| A_dot_base | 5288073, 5374774, 5240931, 5252799, 5243074, 5268290, 5342166 | 26.341450 | +0.405% |
| B_full | 9400339, 9639220, 9429555, 9407272, 9367987, 9365273, 9315986 | 47.001695 | +1.144% |
| B_no_bload | 6727138, 6466070, 6581277, 6445614, 6433665, 6445585, 6507953 | 32.330350 | +0.718% |
| B_no_aloads | 7775137, 7771199, 7818193, 7751415, 7842604, 7722792, 7741474 | 38.855995 | +0.863% |
| B_b_fanout1 | 9405063, 9358637, 9349393, 9427748, 9374371, 9604485, 9591358 | 47.025315 | +1.198% |
| B_no_dots | 8809163, 8824803, 8905193, 8777433, 8771513, 8811727, 8803326 | 44.045815 | +1.583% |

Derived hardware-cycle results: load/use minus matched base = **8.612410 cycles/iteration**; `(dot chain - matched base)/16` = **4.530988 cycles/dot**. Full-minus-ablation deltas are 14.671345 (no B load), 8.145700 (no A loads), -0.023620 (fanout=1), and 2.955880 (no dots). As before these sensitivity deltas are not additive intrinsic instruction costs.

The median wall/cycle implied frequencies were 1.590766--1.595850 GHz across cases. This slight shortfall from 1.6 GHz is expected because wall timestamps intentionally sit outside the perf enable/disable ioctls; wall time is only a consistency diagnostic and hardware cycles are the primary metric.

The removed-load validity issue is fixed by `dyn_init_operands`: it initializes every one of v2--v5 and v8--v15 to deterministic nonzero values before every warmup and every timed sample. This setup executes before the cycle event is enabled. Thus `B_no_bload` and `B_no_aloads` keep all dots/checksums live without inheriting register contents from a previous case. Their changed checksums are expected because deterministic synthetic operands replace the removed loads.

The rebuilt final ELF (`build/machine-benchmarks/dynpreunpack-inner-loop`) was re-disassembled. It contains the new 12-register initialization function and a call before each warmup/sample. Each of the nine benchmark functions still has exactly one backward `bltu`, no loop unrolling/deletion, and eight output stores after the loop; loop-body load/dot/nop counts remain those in the audit table above. Complete output including each sample's enabled/running times, wall duration, implied GHz and checksum is in `build/machine-benchmarks/dynpreunpack-inner-loop-hwcycles-results.txt`.