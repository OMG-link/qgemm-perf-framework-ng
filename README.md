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
--no-verify
```

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