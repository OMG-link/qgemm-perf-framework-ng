# SpacemiT K1/A60 machine benchmarks

This is an independent machine benchmark. Its CMake target is
`k1-l2-bandwidth`; it does not link the GEMM framework.

The tested K1 topology is two shared-L2 clusters: CPUs 0--3 share 512 KiB and
CPUs 4--7 share 512 KiB. Each CPU has a 32 KiB L1D and the cache line is 64 B.
The default test therefore uses CPUs 0,1,2,3, 256 KiB per single-core worker,
and 96 KiB per concurrent worker. The latter totals 384 KiB and is larger than
each L1D while remaining below the shared L2 capacity.

## Build

From the repository root:

Configure an explicit `spacemit-k1-a60` build as described in the project
README, then run:

```bash
cmake --build build/spacemit-k1-a60 --target k1-l2-bandwidth
```

## Direct run

```bash
./build/spacemit-k1-a60/machine-benchmarks/k1-l2-bandwidth --mode all --cpus 0,1,2,3
```

The program pins each worker, warms its buffer, and reports the fastest of the
requested samples. The concurrent aggregate uses the slowest worker duration
as its common interval, so its byte count is not the sum of unrelated time
windows.

## perf measurement

Use `run_k1_l2_perf.py` on the build host. It copies the exact cross-built
binary to the target and runs each perf event group separately. This avoids
silently accepting PMU multiplexing. A group is valid only if every event has
non-zero `time_running` and `time_running/time_enabled` is at least 99%.

```bash
python3 machine-benchmarks/run_k1_l2_perf.py \
  --platform spacemit-k1-a60 \
  --remote spacemit-k1 \
  --binary build/spacemit-k1-a60/machine-benchmarks/k1-l2-bandwidth
```

The runner records raw `perf stat -x ';' --no-scale` output and benchmark
output for every run. It measures these groups independently:

```text
cycles,instructions,load_inst,vector_load_inst
L1-dcache-loads,L1-dcache-load-misses,l2_load_access,l2_load_miss
l2_ar_channel_request,l2_ar_channel_stall_cycle,cycles
```

The K1 native `l2_*` events are preferred over generic `LLC-*` events. The
runner also performs an optional `--validate` pass with 16 KiB, 256 KiB and
16 MiB buffers. Native event counts are treated as evidence only after they
show the expected L1/L2/DRAM trend. `l2_load_access * 64` is not automatically
claimed to be physical L2 traffic.

The former `k1-memory-path-bandwidth` program and scaling runner were removed
from the active build because they directly depended on the TCM allocator. The
historical analysis remains archived in
`docs/2026-09-02-k1-memory-path-bandwidth.md`.