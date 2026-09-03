# M4 batch reduction performance — 2026-08-25

## Scope

This record measures the current `q4_0-q8_0-IME-m4b` implementation after the
recent reduction-buffer and dead-code cleanup changes.

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
Governor:         performance
Current/max freq: 1.6 GHz / 1.6 GHz
Architecture:     rv64gcv_zvfh_zicbop_zba_xsmtvdot
Build:            CMake Release, fixed 256-bit vector length
```

The benchmark binary was rebuilt from the current working tree before testing.

## Correctness preflight

The current binary was checked against the local llama.cpp reference before the
large performance run:

```text
Shape:             M=80, N=160, K=320
Result:            PASS
Maximum abs error: 3.8147e-06
Maximum rel error: 0.00702391
```

## Method

Each performance run used:

```text
Warmup iterations: 3
Samples:           10
GEMMs per sample:  3
Verification:      disabled for timing (`--no-verify`)
```

The two runs were executed serially on the same remote CPU. The results show
the observed run-to-run variation rather than a statistically stabilized
multi-round baseline.

## Results

| Run | Minimum cycles | Median cycles | Time at 1.6 GHz | FMA/cycle | Peak utilization |
|---:|---:|---:|---:|---:|---:|
| 1 | 178,330,264 | 178,530,233 | 111.58 ms | 6.3433 | 4.96% |
| 2 | 188,325,515 | 192,220,044 | 120.14 ms | 5.8915 | 4.60% |
| Mean of medians | — | 185,375,139 | 115.86 ms | 6.1090 | 4.77% |

`FMA/cycle` is calculated as `M*N*K / median_cycles`. Peak utilization uses
the project peak of 128 FMA/cycle. At 1.6 GHz, the mean-of-medians result is
approximately 9.77 GFMA/s.

The median spread between the two runs is approximately 7.67%:

```text
(192,220,044 - 178,530,233) / 178,530,233 = 7.67%
```

## Reproduction

Correctness preflight:

```bash
./compile-and-test.sh build
./compile-and-test.sh run \
  --kernel q4_0-q8_0-IME-m4b --m 80 --n 160 --k 320 \
  --warmup 3 --samples 3 --iterations 3
```

Performance:

```bash
./compile-and-test.sh run \
  --kernel q4_0-q8_0-IME-m4b --m 480 --n 1536 --k 1536 \
  --warmup 3 --samples 10 --iterations 3 --no-verify
```

## M8 batch reduction

### Correctness preflight

The current `q4_0-q8_0-IME-m8b` binary was checked against the local llama.cpp
reference before the performance runs:

```text
Shape:             M=80, N=160, K=320
Result:            PASS
Maximum abs error: 3.8147e-06
Maximum rel error: 0.00702391
```

### Method

The M8 performance runs used the same representative shape and measurement
settings as the M4 batch-reduction results above:

```text
M = 480, N = 1536, K = 1536
Warmup iterations: 3
Samples:           10
GEMMs per sample:  3
Verification:      disabled for timing (`--no-verify`)
```

Two runs were executed serially on CPU 0 of the same `spacemit-k1` system.

### Results

| Run | Minimum cycles | Median cycles | Time at 1.6 GHz | FMA/cycle | Peak utilization |
|---:|---:|---:|---:|---:|---:|
| 1 | 169,110,149 | 169,587,066 | 105.99 ms | 6.6778 | 5.22% |
| 2 | 168,597,497 | 168,758,952 | 105.47 ms | 6.7105 | 5.24% |
| Mean of medians | — | 169,173,009 | 105.73 ms | 6.6941 | 5.23% |

`FMA/cycle` is calculated as `M*N*K / median_cycles`. Peak utilization uses
the project peak of 128 FMA/cycle. At 1.6 GHz, the mean-of-medians result is
approximately 10.71 GFMA/s.

The median spread between the two runs is approximately 0.49%:

```text
(169,587,066 - 168,758,952) / 168,758,952 = 0.49%
```

Using the M4 batch-reduction mean median of 185,375,139 cycles recorded above,
M8 batch reduction is approximately 1.096x faster, or 8.74% fewer cycles. The
comparison uses separate serial sessions and should be treated as indicative.

### Reproduction

Correctness preflight:

```bash
./compile-and-test.sh build
./compile-and-test.sh run \
  --kernel q4_0-q8_0-IME-m8b --m 80 --n 160 --k 320 \
  --warmup 3 --samples 3 --iterations 3
```

Performance:

```bash
./compile-and-test.sh run \
  --kernel q4_0-q8_0-IME-m8b --m 480 --n 1536 --k 1536 \
  --warmup 3 --samples 10 --iterations 3 --no-verify
```

The performance command was executed twice serially on `spacemit-k1`.
