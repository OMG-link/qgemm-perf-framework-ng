# perf-framework-ng

`perf-framework-ng` is a RISC-V kernel correctness, benchmarking, and profiling
framework. Every build directory targets exactly one platform and produces a
standalone executable for that platform.

## Supported platforms

The supported target platform IDs are:

```text
spacemit-k1-a60
spacemit-k1-x60
```

They are independent build targets with separate profile files and complete
metadata. They happen to use the same compiler, compiler tuning name, SSH
destination, and Device Tree compatible, but neither profile includes,
inherits, or falls back to the other.

| Platform | `-march` | Kernels | Allowed CPUs |
|---|---|---|---|
| `spacemit-k1-a60` | `rv64gcv_zvfh_zicbop_zba_xsmtvdot` | RVV and IME | 0–3 |
| `spacemit-k1-x60` | `rv64gcv_zvfh_zicbop_zba` | RVV only | 0–7 |

The compiler tuning argument remains `-mtune=spacemit-x60` for both profiles
because that is the name recognized by the compiler. It is metadata declared
independently in each profile and is not used as the project platform ID.

## Build configuration

The target platform has no default and must always be explicit:

```bash
./compile-and-test.sh build --platform spacemit-k1-a60
./compile-and-test.sh build --platform spacemit-k1-x60
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
    "spacemit-k1-x60": { "cxx": "/absolute/path/to/clang++" }
  }
}
```

Both entries are independent even when they intentionally contain the same
path.

The default build directory is derived from the explicit platform:

```text
build/spacemit-k1-a60/
build/spacemit-k1-x60/
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
```

`--remote` is an SSH destination and need not equal the platform ID. Before
running a benchmark, the wrapper uploads the binary and executes
`--check-platform`. The executable reads the Linux Device Tree `compatible`
property and rejects a board that does not match its build platform.

Both current profiles use the SSH destination `spacemit-k1`. The wrapper reads
the allowed CPU range from that build's manifest: an A60 run may use at most 4
threads pinned from CPU 0, while an X60 run may use at most 8 threads pinned
from CPU 0.

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
All RVV kernels declare both current targets; IME kernels declare only A60. The
build includes only kernels valid for the selected platform.

Implementation directories are named after the implementation dependency when
possible:

- `implementations/rvv-vlen256/` requires at least, and in some cases exactly,
  a 256-bit VLEN;
- `implementations/rvv/` is not restricted to a minimum 256-bit VLEN;
- `implementations/smt-ime1/` uses the IME instruction extension.

An implementation remains platform-named when it embeds board parameters; the
cache-blocked kernels therefore remain under
`implementations/spacemit-k1-a60/`.

## TCM status

The executable includes a shared, platform-aware TCM allocator. At benchmark
startup it maps the complete TCM once (512 KiB on A60, 3 MiB on A100) and then
serves aligned allocations from CPU-appropriate secondary arenas. X60 has no
TCM and uses a no-op backend. TCM-backed IQ2 variants remain excluded from the
active registry.