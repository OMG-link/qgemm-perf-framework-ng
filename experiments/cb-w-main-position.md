# Where does the cache-blocked kernel's W panel actually miss?

Experiment branch: `exp/cb-w-miss-position` (worktree `build/worktrees/cb-wmiss`, based on `f4383ad`).

## Question

`3.4` reports a single `W main` figure (`0.2853 miss/inner` for `m8b-DPU-CB`), which
collapses the whole 8 KiB W micro-panel walk into one number. It cannot say *which
part of the walk* misses. This experiment splits that one load site into per-`bk`
load sites and buckets the L1D-miss samples by panel position.

## Method

`perf` on this board has no data-address sampling (`-d` records, but every one of
16838 samples carries `addr = 0`; the PMU only exposes `event`/`firmware`
formats), so a sample cannot be tied to an address directly. Instead the *load
site* is made position-specific, which is equivalent: in the CB kernel the W
address is a pure function of the iteration index,

```text
W address = baseWQs + bk * 512 B + inner * 256 B        # 8 KiB panel = 16 bk x 2 inner
```

so replacing the single `vle8_v_i8m8(unpackedInner)` with a 17-way `switch` on
`blockWIndex` gives every `bk` its own probe label
(`ime_l1d_probe_m8b_cb_wmainpos_<bk>`), and the sample counts per label are the
per-position miss counts. Every `bk` has identical exposure (2 loads of 256 B,
once per kernel call), so the counts are directly comparable with no
normalisation.

Instrumentation notes:

- the `inner` loop must stay rolled (one label per site), whereas production
  unrolls `numKIter = 2`; the address stream and its order are unchanged;
- each probe emits a 64-NOP sled, so absolute cycle counts are not production
  values — only the *distribution* across positions is used;
- the instrumented build reproduces master's numerical result exactly
  (`337/737280`, `max_abs = 0.00774384` at index 698881), i.e. numerics are
  untouched.

Measurement: `perf record -e r5:u -c 100`, `M=480 N=1536 K=1536`, 3 repetitions,
median per label.

## Result: the distribution is not random

`kRedBatchSize = 8` (production), 16 positions, median samples:

| bk | samples | share | vs uniform |
|---:|---:|---:|---:|
| 0 | 1932 | 18.5% | **2.96x** |
| 1 | 920 | 8.8% | 1.41x |
| 2 | 545 | 5.2% | 0.84x |
| 3 | 480 | 4.6% | 0.74x |
| 4 | 441 | 4.2% | 0.68x |
| 5 | 627 | 6.0% | 0.96x |
| 6 | 509 | 4.9% | 0.78x |
| 7 | 554 | 5.3% | 0.85x |
| 8 | 1073 | 10.3% | **1.64x** |
| 9 | 940 | 9.0% | 1.44x |
| 10 | 479 | 4.6% | 0.73x |
| 11 | 467 | 4.5% | 0.72x |
| 12 | 316 | 3.0% | 0.48x |
| 13 | 305 | 2.9% | 0.47x |
| 14 | 493 | 4.7% | 0.76x |
| 15 | 359 | 3.4% | 0.55x |

- uniform expectation `652.5` samples per position;
- chi-square `3824` with 15 degrees of freedom — uniformity is rejected by a wide
  margin;
- `bk = 0,1,8,9` hold 46.6% of all W-main misses while covering only 25% of the
  panel walk.

## The code says what those four positions have in common

The reduction flush is triggered by `reductionCount`, which advances once per `bk`:

```cpp
bk++;                                                        // kernel.cpp:189
reductionCount++;                                            // kernel.cpp:190
if (reductionCount == kRedBatchSize or bk == BlockCountK) {   // kernel.cpp:191
```

With `kRedBatchSize = 8` and `BlockCountK = 16` a call flushes **twice**:

- after the `bk = 7` iteration (`reductionCount` reaches 8) → the next W load is
  `bk = 8`;
- after the `bk = 15` iteration (`bk == BlockCountK`, the final flush) → the next
  W load is `bk = 0` of the following call.

A flush reads the whole `reduction_components` buffer
(`8 x 16 x 8 x 4 B = 4096 B`), loads `acc` (`512 B`), reloads the W/A scales and
stores the accumulators back — on the order of 5.6 KiB of traffic in a burst.
The panel is only 8 KiB, so the burst is large enough to evict most of it, and the
cost shows up in the W loads that follow.

That predicts exactly the two spikes seen above: `bk = 8,9` after the mid-call
flush and `bk = 0,1` after the previous call's final flush.

## Control: remove the mid-call flush

Rebuilt the same instrumentation with `kRedBatchSize = 16`, so `reductionCount`
can no longer reach the batch size before the end of the call and only the
end-of-call flush remains. Numerics unchanged (`337/737280`, same `max_abs`).

| bk | `kRedBatchSize = 8` | `kRedBatchSize = 16` |
|---:|---:|---:|
| 0 | 2.96x | 2.75x |
| 1 | 1.41x | 1.32x |
| 2 | 0.84x | 0.79x |
| 3 | 0.74x | 0.62x |
| 4 | 0.68x | 0.77x |
| 5 | 0.96x | **1.40x** |
| 6 | 0.78x | 1.06x |
| 7 | 0.85x | 0.74x |
| 8 | **1.64x** | **0.77x** |
| 9 | 1.44x | 1.23x |
| 10 | 0.73x | 0.69x |
| 11 | 0.72x | 0.84x |
| 12 | 0.48x | 0.65x |
| 13 | 0.47x | 0.67x |
| 14 | 0.76x | 0.90x |
| 15 | 0.55x | 0.79x |
| chi-square | 3824 | 3175 |

Both predictions hold:

- the `bk = 8` spike disappears (1.64x → 0.77x) once the mid-call flush is gone;
- the `bk = 0,1` spike survives (2.96x/1.41x → 2.75x/1.32x), because the
  end-of-call flush is still there.

The distribution stays non-uniform in the control too, with a bump moving to
`bk = 5,6` (1.40x/1.06x). This experiment does not explain that one: changing
`kRedBatchSize` also doubles `reduction_components` to 8192 B, so more than one
variable moved.

## Conclusion

W-main L1D misses in `m8b-DPU-CB` are **not** spread randomly over the panel walk.
They concentrate in the W loads that immediately follow a reduction flush —
`bk = 0,1` after the previous call's final flush and `bk = 8,9` after the mid-call
flush — which together hold 46.6% of the misses on 25% of the walk. Removing the
mid-call flush removes its spike, so the flush's ~5.6 KiB burst is what evicts the
panel.

## Reproducing

```sh
BUILD_DIR=$PWD/build ./compile-and-test.sh build
tools/ime_perf/ime_perf.py profile --remote spacemit-k1 --cpu 0 \
  --kernel q4_0-q8_0-IME-m8b-DynPreUnpack-CacheBlocking \
  --m 480 --n 1536 --k 1536 --warmup 2 --samples 1 --iterations 3 \
  --repetitions 3 --event r5:u --period 100 --call-graph none \
  --binary "$PWD/build/ime-llama-bench" --out "$PWD/build/profiles/cb-wmainpos"
python3 tools/extract_probe_perf_script.py build/profiles/cb-wmainpos
python3 tools/aggregate_l1d_probe_symbols.py build/ime-llama-bench \
  build/profiles/cb-wmainpos/run-01/perf-script.txt
```

