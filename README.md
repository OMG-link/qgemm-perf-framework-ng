# perf-framework-ng

`perf-framework-ng` is a next-generation performance testing framework for
all kernels. It provides a self-contained cross-build, correctness-checking,
benchmarking, and profiling environment.

## Build and run

```bash
./compile-and-test.sh configure
./compile-and-test.sh build
./compile-and-test.sh test
./compile-and-test.sh run --list
./compile-and-test.sh run --kernel ID|all --m M --n N --k K
./compile-and-test.sh asm
./compile-and-test.sh clean
```

`configure` generates the build files and compilation database. `build`
compiles the project, while `test` configures, rebuilds, and runs the available
correctness and performance checks. `run` executes the benchmark and supports
selecting a registered kernel or all registered kernels. `asm` generates
assembly output and `clean` removes generated build artifacts.

The toolchain, compiler, deployment target, and build directory can be
overridden with `TOOLCHAIN_DIR`, `CXX`, `DEPLOYMENT_SERVER`, and `BUILD_DIR`.

The IQ2_XXS TCM variants require a RISC-V `fine_mm_*` runtime. CMake detects
`../tcm/libfine_tcm_mm.a` when present; otherwise pass
`-DIME_TCM_RUNTIME=/absolute/path/to/libfine_tcm_mm.a`. Use
`-DIME_ENABLE_TCM=OFF` to build only the non-TCM variants. The benchmark owns
the allocator lifecycle and initializes a 256 KiB TCM pool before first use.

## Benchmark options

The benchmark accepts the following general options:

```text
--list
--kernel ID|all
--m M --n N --k K
--warmup N
--samples N
--iterations N       # omitted: calibrate automatically
--threads N          # 1-4; omitted: single-threaded
--no-verify
--perf-events LIST    # replace the default cycles event with a comma-separated list
--perf-event EVENT    # append one event; may be repeated
```

`q4_0-q8_0-IME-m4i` and the three `q4_0-q8_0-IME-m8b` variants support OpenMP parallel execution.
Pass `--threads N` to select the runtime thread count. The default is one thread,
so kernels without multi-threading support retain their existing behavior. The
remote runner pins an N-thread run to CPU 0 through CPU N-1.

The default event list is `cycles`. Built-in event names are:

```text
cycles,instructions,task-clock,page-faults
l1d-access,l1d-miss,llc-access,llc-miss
dtlb-access,dtlb-miss
```

Machine-specific raw PMU events use `raw:NAME:CONFIG`, where `CONFIG` accepts
decimal or `0x` notation. For example, the SpacemiT X60 vector-load and L1D
load-miss events can be measured together with cycles using:

```bash
./compile-and-test.sh run --kernel ID --m M --n N --k K \
  --perf-events cycles,raw:vector-load:0x39,raw:l1d-miss:0x5
```

Perf counters form one pinned event group and are enabled only around each timed
kernel callback. Input preparation, verification, warmup, state reset, checksum,
and output formatting are outside the measured region. Each metric is reported
per benchmark iteration with its minimum running percentage. A result is
rejected if any counter runs for less than 99% of the enabled time; reduce the
event list if the hardware cannot schedule the complete group. Automatic
iteration calibration always uses a separate cycles counter, so a requested
event list does not need to contain `cycles`. FMA/cycle is unavailable when
`cycles` is omitted.

Each registered kernel provides its own implementation, adapter, layout, and
shape validation. The framework keeps kernel-specific details out of the
common benchmark path, verifies results before measurement by default, and
reports timing and hardware-counter metrics together with an output checksum.

## Profiling

The profiling tools support repeatable remote Linux `perf` measurements without
instrumenting kernel code. They preserve raw profiling data, the benchmark
binary, environment information, reports, and machine-readable summaries in
their configured artifact directory.

## Generated files

Generated binaries, objects, assembly, CMake files, compilation databases, and
profiling artifacts are kept in ignored build or artifact directories. Source
files contain the framework, kernel implementations, adapters, public
interfaces, and benchmark utilities.