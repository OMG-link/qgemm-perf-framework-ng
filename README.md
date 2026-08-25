# SpacemiT IME llama.cpp kernel playground

This directory is a self-contained cross-build and benchmark playground for
the SpacemiT X60 IME i8-by-i4 kernel extracted from llama.cpp.

## Layout

```text
ime-llama/
├── include/                  # Shared types and public kernel interfaces
│   ├── ggml.h
│   ├── ime.h
│   ├── ime1.h
│   └── selected_cycles.h
├── src/                      # Production wrapper, dispatcher, and kernels
│   ├── ime.cpp
│   ├── ime1_kernels.cpp
│   └── selected_cycles.cpp
├── bench/                    # Benchmark entry point and measurement helpers
│   ├── main.cpp
│   ├── perf.h
│   └── timer.hpp
├── experiments/              # Compile-checked, non-dispatched candidates
│   ├── m4_batch_reduction.cpp
│   └── m8_batch_reduction.cpp
├── CMakeLists.txt
├── compile-and-test.sh
└── README.md
```

`include/ggml.h` is the single source of truth for GGML quantization types,
packed block layouts, fp16 conversion helpers, and common GGML macros.
`include/ime1.h` is the shared IME1 kernel API used by both production and
experimental kernels. `include/selected_cycles.h` provides the low-overhead
selected-region cycle counter used during tuning.

All generated binaries, objects, assembly, CMake files, and the compilation
database live under the ignored `build/` directory.

Generate assembly under `build/asm/` with the `asm` command.

## Build and run

The defaults select the SpacemiT LLVM tree and the `spacemit-k1` SSH target
provided for this project:

```bash
./compile-and-test.sh configure       # generates build/compile_commands.json
./compile-and-test.sh build
./compile-and-test.sh test             # smoke test: M=12, N=32, K=32
./compile-and-test.sh test 480 1536 1536
./compile-and-test.sh run 480 1536 1536
./compile-and-test.sh asm
./compile-and-test.sh clean
```

`configure` runs CMake with Ninja and exports the Clang compilation database to
`build/compile_commands.json`. `run` reuses an existing binary when present;
`test` always configures and rebuilds first.
The toolchain, compiler, target, and build directory can be overridden with
`TOOLCHAIN_DIR`, `CXX`, `DEPLOYMENT_SERVER`, and `BUILD_DIR`.

## Benchmark constraints

The active M4 dispatcher requires `M` to be divisible by 4. The benchmark's
packed layout requires `N` to be divisible by 32 and `K` by 32. Invalid or
non-positive dimensions are rejected before allocation.

The benchmark currently fills already-quantized A and packed B buffers with a
fixed byte pattern. It measures kernel performance but is not yet a numerical
correctness test against a reference GEMM.

`src/ime1_kernels.cpp` also contains an existing inline-assembly measurement around
`SQ4BIT_KERNEL_COMP_4x16x16`. It accumulates directly into `selected_cycles`;
the benchmark resets the accumulator after warmup and reports the average for
the timed iterations. Do not remove this instrumentation during optimization.

## IDE configuration

The repository contains a standalone CMake project and `.vscode/` configuration
for opening this directory as the VS Code workspace root. CMake generates
`build/compile_commands.json` from the actual targets and flags, including
`xsmtvdot`, so clangd recognizes intrinsics such as
`__riscv_smt_vmadot_i32m2` without a separate `.clangd` override.

After changing these files, run **clangd: Restart language server** from the
VS Code command palette. Microsoft C/C++ IntelliSense is disabled to avoid
duplicate diagnostics; its compatible compiler configuration is retained in
`.vscode/c_cpp_properties.json` for users who choose to re-enable it.

## Header policy

All GGML-related types must be defined in `include/ggml.h`. In particular, q4/q8
scalar blocks, interleaved `block<K, N>` layouts, q4_K/q8_K layouts, and the
IME scale32 activation blocks must not be redefined in individual kernels.
Add new shared packed layouts to `include/ggml.h` together with a `static_assert` for
their expected size/alignment. Kernel function declarations belong in
`include/ime1.h`; the llama.cpp wrapper interface belongs in `include/ime.h`.