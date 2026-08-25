# M4 immediate reduction performance — 2026-08-25

## Scope

This record measures the `m4-immediate-reduction` implementation after the
intrinsic kernel cleanup and integration into the benchmark framework.

Representative performance shape:

```text
M = 480, N = 1536, K = 1536
FMA per GEMM = M * N * K = 1,132,462,080
```

Packing, input generation, correctness reference, SSH transfer, and output
export are outside the timed region. Each remote run is pinned to CPU 0 by
`compile-and-test.sh`.

## Platform and build

```text
Host:             spacemit-k1
CPU:              Spacemit X60
Kernel:           Linux 6.6.63, riscv64
Architecture:     rv64gcv_zvfh_zicbop_zba_xsmtvdot
Build:            CMake Release, fixed 256-bit vector length
```

The benchmark binary was rebuilt from the current working tree before testing.

## Correctness preflight

The current binary was checked against the local llama.cpp reference before the
large performance runs:

```text
Shape:             M=80, N=160, K=320
Result:            PASS
Maximum abs error: 3.8147e-06
Maximum rel error: 0.00702391
```

The large shape was then run with `--no-verify` to exclude the scalar reference
pass from the timing session.

## Method

Two serial runs were executed on the same remote CPU. Each run used:

```text
Warmup iterations: 3
Samples:           10
GEMMs per sample:  3
Verification:      disabled for timing (`--no-verify`)
```

The representative result is the mean of the two run medians. The two raw
results are retained below to show run-to-run variation.

## Results

| Run | Minimum cycles | Median cycles | Time at 1.6 GHz | FMA/cycle | Peak utilization |
|---:|---:|---:|---:|---:|---:|
| 1 | 141,424,328 | 142,463,062 | 89.04 ms | 7.9492 | 6.21% |
| 2 | 141,467,754 | 143,322,023 | 89.58 ms | 7.9015 | 6.17% |
| Mean of medians | — | 142,892,543 | 89.31 ms | 7.9253 | 6.19% |

`FMA/cycle` is calculated as `M*N*K / median_cycles`. Peak utilization uses
the project peak of 128 FMA/cycle. At 1.6 GHz, the mean-of-medians result is
approximately 12.68 GFMA/s.

The median spread between the two runs is approximately 0.60%:

```text
(143,322,023 - 142,463,062) / 142,463,062 = 0.60%
```

## Comparison with M4 batch reduction

The existing M4 batch-reduction record reports a mean median of 185,375,139
cycles for the same shape. Against that value, M4 immediate reduction measured:

```text
cycle reduction = 22.92%
speedup         = 1.297x
```

The comparison is indicative rather than a controlled A/B run: the values come
from separate benchmark sessions. Both sessions use the same shape, CPU pinning,
fixed vector length, warmup count, sample count, and iteration count.

## Reproduction

Correctness preflight:

```bash
./compile-and-test.sh build
./compile-and-test.sh run \
  --kernel m4-immediate-reduction --m 80 --n 160 --k 320 \
  --warmup 3 --samples 3 --iterations 3
```

Performance run:

```bash
./compile-and-test.sh run \
  --kernel m4-immediate-reduction --m 480 --n 1536 --k 1536 \
  --warmup 3 --samples 10 --iterations 3 --no-verify
```

The performance command was executed twice serially on `spacemit-k1`.