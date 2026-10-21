# perf-framework-ng

`perf-framework-ng` is a RISC-V kernel correctness, benchmarking, and profiling
framework. Every build directory targets exactly one platform and produces a
standalone executable for that platform.

## Supported platforms

The supported target platform IDs are:

```text
spacemit-k1-a60
spacemit-k1-x60
spacemit-k3-a100
```

Each platform is an independent build target with its own profile file and
complete metadata. The two K1 profiles happen to use the same compiler,
compiler tuning name, SSH destination, and Device Tree compatible; the K3 / A100
profile differs in compiler tuning name, Device Tree compatible, CPU range, and
deployment prologue. No profile includes, inherits, or falls back to another.

| Platform | `-march` | Kernels | Allowed CPUs |
|---|---|---|---|
| `spacemit-k1-a60` | `rv64gcv_zvfh_zicbop_zba_xsmtvdot` | RVV and IME1 | 0–3 |
| `spacemit-k1-x60` | `rv64gcv_zvfh_zicbop_zba` | RVV only | 0–7 |
| `spacemit-k3-a100` | `rv64gcv_zvfh_zicbop_zba_zvl1024b_xsmtvdotii` | IME2 only | 8–15 |

The compiler tuning argument is `-mtune=spacemit-x60` for both K1 profiles
because that is the name recognized by the compiler, and `-mtune=spacemit-a100`
for K3. It is metadata declared independently in each profile and is not used as
the project platform ID.

## Build configuration

The target platform has no default and must always be explicit:

```bash
./compile-and-test.sh build --platform spacemit-k1-a60
./compile-and-test.sh build --platform spacemit-k1-x60
./compile-and-test.sh build --platform spacemit-k3-a100
```

Only the target system facts are pinned in `CMakeLists.txt`, before `project()`:

```cmake
if(NOT CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_SYSTEM_NAME Linux)
    set(CMAKE_SYSTEM_PROCESSOR riscv64)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
endif()
```

CMake needs the target system before `project()`, so these values cannot be
ordinary project options. They are stable project facts and contain no
machine-local paths.

The compiler is deliberately not pinned in the repository. It is resolved in
this order:

```text
--cxx FILE  >  .pfng-workspace.json  >  error
```

`.pfng-workspace.json` is ignored by Git and binds each platform independently.
Copy `pfng-workspace.example.json` to create it:

```json
{
  "platforms": {
    "spacemit-k1-a60": { "cxx": "/absolute/path/to/clang++" },
    "spacemit-k1-x60": { "cxx": "/absolute/path/to/clang++" },
    "spacemit-k3-a100": { "cxx": "/absolute/path/to/clang++" }
  }
}
```

All entries are independent even when they intentionally contain the same path.
The K3 / A100 profile needs a compiler whose RISC-V backend supports
`XSMTVDotII`; bind a different `cxx` for it when required.

The default build directory is derived from the explicit platform:

```text
build/spacemit-k1-a60/
build/spacemit-k1-x60/
build/spacemit-k3-a100/
```

Direct CMake use needs the platform and the compiler path, since the repository
does not provide one:

```bash
cmake -S . -B build/spacemit-k1-a60 -G Ninja \
  -DPFNG_TARGET_PLATFORM=spacemit-k1-a60 \
  -DCMAKE_CXX_COMPILER=/absolute/path/to/clang++ \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/spacemit-k1-a60 --target pfng-bench
```

A `CXX` environment variable or `-DCMAKE_TOOLCHAIN_FILE=<file>` also satisfies
the compiler requirement, for example when a full external toolchain that pins a
sysroot should take over.

## Run

The SSH target also has no default:

```bash
./compile-and-test.sh run \
  --platform spacemit-k1-a60 \
  --remote spacemit-k1 \
  --list

./compile-and-test.sh run \
  --platform spacemit-k1-a60 \
  --remote spacemit-k1 \
  --kernel q4_0-q8_0-IME-m8b --m 480 --n 1536 --k 1536

./compile-and-test.sh run \
  --platform spacemit-k3-a100 \
  --remote spacemit-k3 \
  --kernel q4_0-q8_0-IME-RR --m 480 --n 1536 --k 1536
```

`--remote` is an SSH destination and need not equal the platform ID. Before
running a benchmark, the wrapper uploads the binary and executes
`--check-platform`. The executable reads the Linux Device Tree `compatible`
property and rejects a board that does not match its build platform.

The K1 profiles use the SSH destination `spacemit-k1`. The wrapper reads the
allowed CPU range from that build's manifest: an A60 run may use at most 4
threads pinned from CPU 0, while an X60 run may use at most 8 threads pinned from
CPU 0. The K3 / A100 profile uses `spacemit-k3` and CPUs 8–15. Its profile
declares the remote prologue `echo 0 > /proc/set_ai_thread`, which the wrapper
runs before `taskset` because A100 cores are not in the default affinity mask.

Useful executable queries are:

```text
--build-info       print embedded build platform and compiler information
--platform-info    detect and print the runtime board
--check-platform   verify runtime board against the embedded build platform
--list             list kernels built for this platform
```

## Benchmark options

```text
--kernel ID|all
--m M --n N --k K
--warmup N
--samples N
--iterations N
--threads N
--no-verify
--perf-events LIST
--perf-event EVENT
```

The default event is `cycles`. Built-in event names include `cycles`,
`instructions`, `task-clock`, cache and DTLB events. Platform-specific raw PMU
events use `raw:NAME:CONFIG`.

Each kernel declares its supported platforms in its local `CMakeLists.txt`.
RVV kernels declare both K1 targets. IME1 kernels declare A60, IME2 kernels
declare A100, and kernels with an implementation for both IME generations
(`q4_0-q8_0-IME-m8b` and `q4_0-q8_0-IME-upstream`) declare A60 and A100 and
select the implementation directory in their `CMakeLists.txt`. The build includes
only kernels valid for the selected platform.

Implementation directories are named after the implementation dependency when
possible:

- `implementations/rvv-vlen256/` requires at least, and in some cases exactly,
  a 256-bit VLEN;
- `implementations/rvv/` is not restricted to a minimum 256-bit VLEN;
- `implementations/smt-ime1/` uses the IME1 (K1 / A60) instruction extension;
- `implementations/smt-ime2/` uses the IME2 (K3 / A100) extension, whose vector
  length is fixed at 1024 bits and which adds `XSMTVDotII`.

An implementation remains platform-named when it embeds board parameters; the
cache-blocked kernels therefore remain under
`implementations/spacemit-k1-a60/`.

## Profiling

`tools/pfng_perf/pfng_perf.py` profiles an unmodified benchmark binary over SSH.
It records `perf` PC samples, preserves the raw evidence, and can combine an
external `perf stat` cycle count with configured source-region ranges to report
cycles per region. It accepts `--platform`, `--remote`, and `--cpu`, validates
the binary against its build manifest, and writes artifacts under
`tools/pfng_perf/.artifacts/perf-cpu-clock/`:

```bash
python3 tools/pfng_perf/pfng_perf.py profile \
  --remote spacemit-k3 --platform spacemit-k3-a100 --cpu 8 \
  --kernel q4_0-q8_0-IME-RR --m 480 --n 1536 --k 1536 \
  --cycles-event cycles:u
```

The method and a worked K3 / A100 example are in
[2026-09-29-rr-per-region-cycles.md](docs/2026-09-29-rr-per-region-cycles.md).

## TCM status

The executable includes a shared, platform-aware TCM allocator. At benchmark
startup it maps the complete TCM once (512 KiB on A60, 3 MiB on A100) and then
serves aligned allocations from CPU-appropriate secondary arenas. A60 CPUs 0–3
share one arena; A100 CPUs 8–15 each allocate from their own 384 KiB slice. X60
has no TCM and uses a no-op backend. TCM-backed IQ2 variants remain excluded from
the active registry.